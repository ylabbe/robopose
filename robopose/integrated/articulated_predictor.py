import torch

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from robopose.lib3d.robopose_ops import (
    TCO_init_from_boxes_zup_autodepth,
)
from robopose.lib3d.camera_geometry import project_points
from robopose.lib3d.transform_ops import transform_pts, invert_T

from robopose.lib3d.articulated_mesh_database import Meshes

import robopose.utils.tensor_collection as tc

from robopose.evaluation.data_utils import data_to_pose_model_inputs

from robopose.utils.logging import get_logger
from robopose.utils.timer import Timer

logger = get_logger(__name__)


class ArticulatedObjectPredictor(torch.nn.Module):
    def __init__(self, model, bsz_objects=64):
        super().__init__()
        assert model is not None
        self.model = model
        self.can_predict_joints = self.model.cfg.predict_joints
        robots = self.model.robots
        assert len(robots) == 1
        self.robot = robots[0]
        self.urdf_layer = self.model.get_urdf_layer(self.robot.name)
        self.bsz_objects = bsz_objects
        self.cfg = model.cfg
        self.eval()

    @torch.no_grad()
    def batched_model_predictions(self, images, K, obj_data, n_iterations=1, update_obj_infos=True):
        timer = Timer()
        timer.start()

        ids = torch.arange(len(obj_data))

        ds = TensorDataset(ids)
        dl = DataLoader(ds, batch_size=self.bsz_objects)
        preds = defaultdict(list)
        for (batch_ids, ) in dl:
            obj_inputs = obj_data[batch_ids.numpy()]
            im_ids = obj_inputs.infos.batch_im_id.values
            images_ = images[im_ids]
            K_ = K[im_ids]
            TCO_input, obj_infos_input = data_to_pose_model_inputs(obj_inputs)

            outputs = self.model(images=images_, K=K_, TCO=TCO_input,
                                 obj_infos=obj_infos_input,
                                 n_iterations=n_iterations,
                                 update_obj_infos=update_obj_infos,
                                 deterministic=True)

            for n in range(1, n_iterations+1):
                iter_outputs = outputs[f'iteration={n}']
                bsz = len(images_)
                obj_outputs = iter_outputs['obj_infos_output']
                obj_inputs_ = iter_outputs['obj_infos_input']

                q_input = {k: torch.cat([obj_inputs_[n]['joints'][k] for n in range(bsz)], dim=0) for k in obj_inputs_[0]['joints'].keys()}
                q_pred = {k: torch.cat([obj_outputs[n]['joints'][k] for n in range(bsz)], dim=0) for k in obj_outputs[0]['joints'].keys()}
                q_pred = self.urdf_layer.to_tensor(q_pred)

                infos = obj_inputs.infos

                data = tc.PandasTensorCollection(infos, poses=iter_outputs['TCO_output'],
                                                 K=iter_outputs['K_input'],
                                                 joints=q_pred,
                                                 K_crop=iter_outputs['K_crop'])
                preds[f'iteration={n}'].append(data)

        for k, v in preds.items():
            preds[k] = tc.concatenate(v)
        logger.debug(f'Pose prediction on {len(obj_data)} detections (n_iterations={n_iterations}) (joint_update={update_obj_infos}): {timer.stop()}')
        return preds

    def pred_keypoints(self, pred_data):
        labels = pred_data.infos.label.values
        q = pred_data.joints
        K = pred_data.K[pred_data.infos.batch_im_id.values]
        TCO = pred_data.poses
        keypoints_3d = self.model.mesh_db.urdf_layers[0].get_keypoints(q)
        TCO_keypoints_3d = transform_pts(TCO, keypoints_3d)
        keypoints_2d = project_points(keypoints_3d, K, TCO)
        return TCO_keypoints_3d, keypoints_2d

    def make_init_obj_data(self, detections, K, resolution=None,
                           joints=None, use_known_joints=False):
        # Joint initialization
        obj_infos = []
        bsz = len(detections)
        if use_known_joints:
            tensor_joints = self.urdf_layer.to_tensor(joints)
            logger.info('Using provided joints for initialization.')
        else:
            tensor_joints = self.urdf_layer.joints_default.unsqueeze(0).repeat(bsz, 1)
            logger.info('Using default joints for initialization.')
        tensor_joints = tensor_joints.float().cuda()
        detections.infos['joint_names'] = [self.urdf_layer.joint_names.tolist() for _ in range(bsz)]
        for n, row in enumerate(detections.infos.itertuples()):
            obj_infos.append(dict(name=row.label, joints=self.urdf_layer.from_tensor(tensor_joints[[n]])))

        # Pose initialization
        boxes = detections.bboxes
        K_ = K[detections.infos.batch_im_id.values]
        meshes = self.model.mesh_db.select(obj_infos)
        _, T_offset = meshes.center_meshes()
        t_O_CENTROID = T_offset[:, :3, -1]
        centered_meshes = Meshes(meshes.labels, transform_pts(invert_T(T_offset), meshes.points))
        centered_points = centered_meshes.sample_points(2000, deterministic=True)
        T_C_CENTROID_init = TCO_init_from_boxes_zup_autodepth(boxes, centered_points, K_)
        TCO_init = T_C_CENTROID_init @ invert_T(T_offset)

        data = tc.PandasTensorCollection(infos=detections.infos,
                                         K=K_, poses=TCO_init,
                                         joints=self.urdf_layer.to_tensor(tensor_joints))
        return data

    def get_predictions(self, images, K,
                        n_iterations=1,
                        use_gt_joints=False,
                        gt_joints=None,
                        detections=None,
                        data_TCO_init=None):

        if not use_gt_joints:
            assert self.can_predict_joints
        if use_gt_joints:
            assert gt_joints is not None

        n_coarse_iterations = 1
        n_refiner_iterations = n_iterations - n_coarse_iterations

        preds = dict()
        if data_TCO_init is None:
            assert detections is not None
            assert self.model is not None
            assert n_coarse_iterations > 0
            init_with_known_joints = not self.can_predict_joints
            data_TCO_init = self.make_init_obj_data(detections, K=K,
                                                    use_known_joints=init_with_known_joints,
                                                    joints=gt_joints)

            preds['init'] = data_TCO_init
            pred_key = 'init'
            data_TCO = data_TCO_init
        else:
            assert len(K) == len(data_TCO_init)
            data_TCO = data_TCO_init
            data_TCO.register_tensor('K', K)
            pred_key = 'external_init'

        if use_gt_joints:
            data_TCO_init.joints = self.urdf_layer.to_tensor(gt_joints)

        preds[pred_key] = data_TCO

        model_updates_joints = self.can_predict_joints and not use_gt_joints
        model_preds = self.batched_model_predictions(images, K, data_TCO_init,
                                                     n_iterations=n_iterations,
                                                     update_obj_infos=model_updates_joints)

        for n in range(1, n_iterations + 1):
            preds[f'iteration={n}'] = model_preds[f'iteration={n}']
        data_TCO = preds[f'iteration={n_iterations}']
        return data_TCO, preds
