import torch
import numpy as np
import trimesh
import numpy as np

from torch import nn

from robopose.config import DEBUG_DATA_DIR

from robopose.lib3d.camera_geometry import get_K_crop_resize, project_points, boxes_from_uv, project_points_robust
from robopose.lib3d.cropping import deepim_crops_robust
from robopose.lib3d.rotations import compute_rotation_matrix_from_ortho6d
from robopose.lib3d.transform_ops import invert_T, normalize_T, transform_pts
from robopose.lib3d.robopose_ops import pose_update_with_reference_point
from robopose.lib3d.articulated_mesh_database import Meshes

from robopose.utils.logging import get_logger
logger = get_logger(__name__)


class ArticulatedObjectRefiner(nn.Module):
    def __init__(self,
                 backbone, renderer, mesh_db,
                 render_size=(240, 320),
                 center_crop_on='centroid',
                 reference_point='centroid',
                 predict_joints=True,
                 input_anchor_mask=True,
                 possible_anchors='all_with_points'):
        super().__init__()

        self.backbone = backbone
        self.renderer = renderer
        self._mesh_db = [mesh_db]
        self.render_size = render_size
        self.pose_dim = 9
        self.predict_joints = predict_joints
        self.input_anchor_mask = input_anchor_mask
        self.reference_point = reference_point

        n_features = backbone.n_features

        self.heads = dict()
        self.pose_fc = nn.Linear(n_features, self.pose_dim, bias=True)
        self.heads['pose'] = self.pose_fc

        if predict_joints:
            assert len(self.robots) == 1
            for robot in self.robots:
                joint_names = np.array([joint.name for joint in robot.actuated_joints])
                n_dof = len(joint_names)
                robot_name = robot.name
                joints_fc = nn.Linear(n_features, n_dof, bias=True)
                setattr(self, f'{robot_name}_joints_fc', joints_fc)
                self.heads[robot_name] = joints_fc

        if not predict_joints:
            assert not input_anchor_mask
            assert possible_anchors == 'base_only', possible_anchors

        if isinstance(possible_anchors, str):
            if possible_anchors == 'all_with_points':
                self.possible_anchor_link_names = [link.name for link in self.urdf_layer.robot.links if self.urdf_layer.link_to_n_pts[link.name] > 0]
            elif possible_anchors == 'base_only':
                self.possible_anchor_link_names = [self.urdf_layer.robot.base_link.name]
            elif possible_anchors[:3] == 'top':
                volumes = []
                for link_id, pts in enumerate(self.urdf_layer.link_pts):
                    link_name = self.urdf_layer.link_names[link_id]
                    if self.urdf_layer.link_to_n_pts[link_name] > 0:
                        pts = self.urdf_layer.link_pts[link_id].cpu().numpy()
                        volumes.append((link_name, trimesh.points.PointCloud(vertices=pts).convex_hull.volume))
                n_top = int(possible_anchors[4])
                volumes = sorted(volumes, key=lambda x: -x[1])[:n_top]
                self.possible_anchor_link_names = list(map(lambda x:x[0], volumes))
        elif isinstance(possible_anchors, list):
            self.possible_anchor_link_names = possible_anchors
        else:
            raise TypeError(type(possible_anchors))
        self.possible_anchor_link_names = tuple(self.possible_anchor_link_names)
        self.possible_link_name_to_id = {link_name: n for n, link_name in enumerate(self.possible_anchor_link_names)}
        logger.debug(f'Possible anchor link names: {self.possible_anchor_link_names}')

        self.debug = False
        self.tmp_debug = dict()

    @property
    def robots(self):
        return [layer.robot for layer in self.mesh_db.urdf_layers]

    def get_urdf_layer(self, robot_name):
        return self.mesh_db.label_to_urdf_layer[robot_name]

    @property
    def urdf_layer(self):
        assert len(self.mesh_db.label_to_urdf_layer) == 1
        return list(self.mesh_db.label_to_urdf_layer.values())[0]

    @property
    def mesh_db(self):
        return self._mesh_db[0]

    def enable_debug(self):
        self.debug = True

    def disable_debug(self):
        self.debug = False

    def crop_inputs(self, images, K, TCO, obj_infos, t_C_CENTER, boxes_crop=None):
        bsz, nchannels, h, w = images.shape
        assert K.shape == (bsz, 3, 3)
        assert TCO.shape == (bsz, 4, 4)
        assert len(obj_infos) == bsz
        assert t_C_CENTER.shape == (bsz, 3)

        meshes = self.mesh_db.select(obj_infos)
        points = meshes.sample_points(2000, deterministic=True)
        uv = project_points_robust(points, K, TCO)
        boxes_rend = boxes_from_uv(uv)
        if boxes_crop is not None:
            bboxes = torch.cat([torch.arange(bsz).unsqueeze(1).to(device).float(), boxes_crop], dim=1)
            images_cropped = torchvision.ops.roi_align(
                images, bboxes,
                output_size=((min(self.render_size), max(self.render_size))), sampling_ratio=4)
        else:
            boxes_crop, images_cropped = deepim_crops_robust(
                images=images, obs_boxes=boxes_rend, K=K,
                TCO_pred=TCO, tCR_in=t_C_CENTER,
                O_vertices=points, output_size=self.render_size, lamb=1.4
            )
        K_crop = get_K_crop_resize(K=K.clone(), boxes=boxes_crop,
                                   orig_size=images.shape[-2:],
                                   crop_resize=self.render_size)
        if self.debug:
            T_C_REF = TCO.clone()
            T_C_REF[:, :3, -1] = t_C_CENTER
            self.tmp_debug.update(
                boxes_rend=boxes_rend,
                boxes_crop=boxes_crop,
                center_crop_point_uv=project_points_robust(torch.zeros(bsz, 1, 3).to(K.device), K, T_C_REF),
                origin_uv=project_points_robust(torch.zeros(bsz, 1, 3).to(K.device), K, TCO),
                origin_uv_crop=project_points_robust(torch.zeros(bsz, 1, 3).to(K.device), K_crop, TCO),
                uv=uv,
            )
        return images_cropped, K_crop.detach(), boxes_rend, boxes_crop

    def update_pose(self, TCO, K_crop, pose_outputs, t_C_REF):
        if self.pose_dim == 9:
            dR = compute_rotation_matrix_from_ortho6d(pose_outputs[:, 0:6])
            vxvyvz = pose_outputs[:, 6:9]
        else:
            raise ValueError(f'pose_dim={self.pose_dim} not supported')
        TCO_updated = pose_update_with_reference_point(TCO, K_crop, vxvyvz, dR, t_C_REF)
        return TCO_updated

    def update_joints(self, robot_name, q_input_dict, q_update_tensor, clamp=True):
        urdf_layer = self.get_urdf_layer(robot_name)
        q_input_tensor = urdf_layer.to_tensor(q_input_dict)
        q_output = q_input_tensor.detach() + q_update_tensor
        if clamp:
            q_output = self.get_urdf_layer(robot_name).clamp_q(q_output)
        return urdf_layer.from_tensor(q_output)

    def update_obj_infos(self, obj_infos, model_outputs, clamp=True):
        obj_infos_updated = []
        for n, infos_n in enumerate(obj_infos):
            name = infos_n['name']
            new_infos = dict(name=name)
            joints_update = model_outputs[name][n]
            new_joints = self.update_joints(name, infos_n['joints'], joints_update, clamp=clamp)
            new_infos['joints'] = new_joints
            obj_infos_updated.append(new_infos)
        return obj_infos_updated

    def net_forward(self, x):
        x = self.backbone(x)
        x = x.flatten(2).mean(dim=-1)
        outputs = dict()
        for k, head in self.heads.items():
            outputs[k] = head(x)
        return outputs

    def obj_infos_to_tensor(self, obj_infos):
        q = []
        for n in range(len(obj_infos)):
            q.append(self.urdf_layer.to_tensor(obj_infos[n]['joints']))
        q = torch.cat(q, dim=0)
        return q

    def detach_obj_infos(self, obj_infos):
        obj_infos_detached = []
        for obj in obj_infos:
            joints = obj['joints']
            name = obj['name']
            obj_infos_detached.append(dict(
                joints={k: v.detach() for k, v in joints.items()},
                name=name
            ))
        return obj_infos_detached

    def forward(self, images, K, obj_infos,
                TCO, n_iterations=1,
                update_obj_infos=False,
                deterministic=False):
        bsz, nchannels, h, w = images.shape
        assert K.shape == (bsz, 3, 3)
        assert TCO.shape == (bsz, 4, 4)
        assert len(obj_infos) == bsz
        if deterministic:
            seeds = images.flatten(1).sum(dim=-1).to(torch.int).cpu().numpy()

        outputs = dict()
        TCO_input = TCO
        obj_infos_input = obj_infos
        for n in range(n_iterations):
            TCO_input = normalize_T(TCO_input.detach())
            obj_infos_input = self.detach_obj_infos(obj_infos_input)

            # Crop the input image
            # Centroid is always in the center of the image
            meshes = self.mesh_db.select(obj_infos_input)
            _, T_offset = meshes.center_meshes()
            t_O_CENTROID = T_offset[:, :3, -1]
            t_C_CENTROID = TCO_input[..., :3, [-1]] + TCO_input[..., :3, :3] @ t_O_CENTROID.unsqueeze(-1)
            t_C_CENTROID = t_C_CENTROID.squeeze(-1)
            images_crop, K_crop, boxes_rend, boxes_crop = self.crop_inputs(
                images, K, TCO_input, obj_infos_input, t_C_CENTROID)

            # Choose anchor
            robot = self.urdf_layer.robot
            anchor_link_names = []
            change_mask = None
            for idx in range(bsz):
                if deterministic:
                    np_random = np.random.RandomState(seeds[idx] + n)
                else:
                    np_random = np.random
                anchor_link_name = np_random.choice(self.possible_anchor_link_names)
                anchor_link_names.append(anchor_link_name)

            # Compute inputs
            if self.input_anchor_mask:
                renders_rgb, renders_mask_int = self.renderer.render(
                    obj_infos=obj_infos_input, TCO=TCO_input,
                    K=K_crop, resolution=self.render_size,
                    render_mask=True,
                )
                # NOTE: Same order of links in urdfpy and bullet ?
                pb_anchor_link_ids = [self.urdf_layer.link_name_to_pb_link_id[link_name] for link_name in anchor_link_names]
                anchor_link_mask_ids = [0 + (link_id << 24) for link_id in pb_anchor_link_ids]
                anchor_masks = renders_mask_int == torch.tensor(anchor_link_mask_ids,
                                                                device=renders_mask_int.device,
                                                                dtype=renders_mask_int.dtype)[:, None, None]
                anchor_masks = anchor_masks.float()
                anchor_masks = anchor_masks.unsqueeze(1)
                x = torch.cat((images_crop, renders_rgb, anchor_masks), dim=1)
            else:
                renders_rgb = self.renderer.render(obj_infos=obj_infos_input,
                                                   TCO=TCO_input, K=K_crop,
                                                   resolution=self.render_size)
                x = torch.cat((images_crop, renders_rgb), dim=1)
                anchor_masks = None
            anchor_link_ids = [self.urdf_layer.link_name_to_id[link_name] for link_name in anchor_link_names]

            refiner_outputs = self.net_forward(x)

            # Update the pose of the anchor
            q_input = self.obj_infos_to_tensor(obj_infos_input)
            TOL_input = self.urdf_layer.get_TWL(q_input)
            TOA_input = TOL_input[torch.arange(bsz), anchor_link_ids]
            TCA_input = TCO_input @ TOA_input

            if self.reference_point == 'centroid':
                t_C_REF = t_C_CENTROID
            elif self.reference_point[:7] == 'on_link':
                ref_point_link_name = self.reference_point.split('on_link=')[-1]
                ref_point_link_id = self.urdf_layer.link_name_to_id[ref_point_link_name]
                t_C_REF = (TCO_input @ TOL_input[:, ref_point_link_id])[:, :3, -1]
            else:
                raise ValueError(self.reference_point)
            pose_update = refiner_outputs['pose']
            TCA_output = self.update_pose(TCA_input, K_crop, pose_update, t_C_REF)

            # Joint update
            if update_obj_infos:
                assert self.predict_joints
                obj_infos_output = self.update_obj_infos(obj_infos_input, refiner_outputs, clamp=True)
                obj_infos_output_no_clamp = self.update_obj_infos(obj_infos_input, refiner_outputs, clamp=False)
            else:
                obj_infos_output = obj_infos_input
                obj_infos_output_no_clamp = obj_infos_input

            # Update the pose of the robots root/base.
            urdf_layer = self.urdf_layer
            q_output = self.obj_infos_to_tensor(obj_infos_output)
            TCO_output = urdf_layer.compute_urdf_root_link_pose(TCA_output, anchor_link_names, q_output)

            outputs[f'iteration={n+1}'] = {
                'TCO_input': TCO_input,
                'TCO_output': TCO_output,
                'anchor_link_names': anchor_link_names,
                't_C_REF': t_C_REF,
                'TCA_input': TCA_input,
                'TCA_output': TCA_output,

                'obj_infos_input': obj_infos_input,
                'obj_infos_output': obj_infos_output,
                'obj_infos_output_no_clamp': obj_infos_output_no_clamp,

                'K_input': K,
                'K_crop': K_crop,
                'refiner_outputs': refiner_outputs,
                'boxes_rend': boxes_rend,
                'boxes_crop': boxes_crop,
            }

            if self.debug:
                self.tmp_debug.update(outputs[f'iteration={n+1}'])
                T_C_REF = TCO_input.clone().detach()
                T_C_REF[:, :3, -1] = t_C_REF
                self.tmp_debug.update(
                    uv_ref_point=project_points_robust(torch.zeros(bsz, 1, 3).to(K.device), K_crop, T_C_REF),
                    images=images,
                    images_crop=images_crop,
                    renders=renders_rgb,
                    anchor_masks=anchor_masks,
                )
                debug_data_path = DEBUG_DATA_DIR / f'debug_iter={n+1}.pth.tar'
                logger.info(debug_data_path)
                torch.save(self.tmp_debug, debug_data_path)

            TCO_input = TCO_output
            obj_infos_input = obj_infos_output


        return outputs
