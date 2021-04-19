import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from collections import defaultdict

from robopose.datasets.samplers import DistributedSceneSampler
import robopose.utils.tensor_collection as tc
from robopose.utils.distributed import get_world_size, get_rank, get_tmp_dir
from robopose.lib3d.camera_geometry import cropresize_backtransform_points2d

from torch.utils.data import DataLoader


class ArticulatedObjectPredictionRunner:
    def __init__(self, scene_ds, batch_size=4, cache_data=False, n_workers=4):

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.tmp_dir = get_tmp_dir()

        sampler = DistributedSceneSampler(scene_ds, num_replicas=self.world_size, rank=self.rank)
        self.sampler = sampler
        dataloader = DataLoader(scene_ds, batch_size=batch_size,
                                num_workers=n_workers,
                                sampler=sampler, collate_fn=self.collate_fn)

        if cache_data:
            self.dataloader = list(tqdm(dataloader))
        else:
            self.dataloader = dataloader

    def collate_fn(self, batch):
        batch_im_id = -1
        cam_infos, K, TWC = [], [], []
        joints = defaultdict(list)
        orig_K, cropresize_bboxes, orig_wh = [], [], []
        det_infos, bboxes, poses_gt = [], [], []
        images = []
        for n, data in enumerate(batch):
            rgb, masks, obs = data
            batch_im_id += 1
            frame_info = obs['frame_info']
            im_info = {k: frame_info[k] for k in ('scene_id', 'view_id')}
            im_info.update(batch_im_id=batch_im_id)
            cam_info = im_info.copy()

            if 'orig_camera' in obs:
                orig_K_ = obs['orig_camera']['K']
                res = obs['orig_camera']['resolution']
                orig_wh_ = [max(res), min(res)]
                cropresize_bbox = obs['orig_camera']['crop_resize_bbox']
            else:
                orig_K_ = obs['camera']['K']
                orig_wh_ = [rgb.shape[1], rgb.shape[0]]
                cropresize_bbox = (0, 0, orig_wh[0]-1, orig_wh[1]-1)

            orig_K.append(torch.as_tensor(orig_K_).float())
            cropresize_bboxes.append(torch.as_tensor(cropresize_bbox))
            orig_wh.append(torch.as_tensor(orig_wh_))

            K.append(obs['camera']['K'])
            TWC.append(obs['camera']['TWC'])
            cam_infos.append(cam_info)
            images.append(rgb)

            for o, obj in enumerate(obs['objects']):
                obj_info = dict(
                    label=obj['name'],
                    score=1.0,
                )
                obj_info.update(im_info)

                h, w, _ = rgb.shape
                m = 1/5
                bbox = np.array([w*m, h*m, w-w*m, h-h*m])
                bboxes.append(bbox)

                det_infos.append(obj_info)
                assert 'joints' in obj
                for k, v in obj['joints'].items():
                    joints[k].append(torch.as_tensor(v).view(-1).float())

        detections = tc.PandasTensorCollection(
            infos=pd.DataFrame(det_infos),
            bboxes=torch.as_tensor(np.stack(bboxes)).float(),
        )
        cameras = tc.PandasTensorCollection(
            infos=pd.DataFrame(cam_infos),
            K=torch.as_tensor(np.stack(K)),
            orig_K=torch.as_tensor(np.stack(orig_K)),
            orig_wh=torch.as_tensor(np.stack(orig_wh)),
            cropresize_bboxes=torch.as_tensor(np.stack(cropresize_bboxes)),
            TWC=torch.as_tensor(np.stack(TWC)),
        )
        data = dict(
            images=torch.stack(images),
            cameras=cameras,
            detections=detections,
            joints=joints,
        )
        return data

    def get_predictions(self, obj_predictor,
                        use_gt_joints=True,
                        predict_keypoints=True,
                        n_iterations=10):

        predictions = defaultdict(list)
        for data in tqdm(self.dataloader):
            images = data['images'].cuda().float().permute(0, 3, 1, 2) / 255
            detections = data['detections']
            cameras = data['cameras'].cuda().float()
            joints = data['joints']
            stacked_joints = dict()
            for k, v in joints.items():
                stacked_joints[k] = torch.stack(v).cuda().float()
            joints = stacked_joints

            assert len(obj_predictor.model.robots) == 1
            robot = obj_predictor.model.robots[0]
            urdf_layer = obj_predictor.model.get_urdf_layer(robot.name)
            joints_tensor = urdf_layer.to_tensor(joints)
            detections.register_tensor('joints', joints_tensor)
            detections.infos['joint_names'] = urdf_layer.joint_names[None].repeat(len(detections), axis=0).tolist()

            K = cameras.K.float()
            detections = detections.cuda().float()

            _, preds = obj_predictor.get_predictions(
                images=images, K=K,
                detections=detections,
                use_gt_joints=use_gt_joints,
                gt_joints=joints_tensor,
                n_iterations=n_iterations,
            )
            if predict_keypoints:
                for pred_k, pred_v in preds.items():
                    TCO_keypoints_3d, keypoints_2d = obj_predictor.pred_keypoints(pred_v)
                    orig_wh = cameras.orig_wh[pred_v.infos.batch_im_id]
                    cropresize_bboxes = cameras.cropresize_bboxes[pred_v.infos.batch_im_id]
                    wh = torch.zeros_like(orig_wh)
                    wh[:, 1] = images.shape[2]
                    wh[:, 0] = images.shape[3]
                    keypoints_2d = cropresize_backtransform_points2d(
                        orig_wh, cropresize_bboxes, wh, keypoints_2d
                    )
                    pred_v.register_tensor('keypoints_2d', keypoints_2d)
                    pred_v.register_tensor('TCO_keypoints_3d', TCO_keypoints_3d)
                    predictions[pred_k].append(pred_v)

        for k, v in predictions.items():
            predictions[k] = tc.concatenate(predictions[k])
        return predictions
