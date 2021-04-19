import torch
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from robopose.utils.logging import get_logger


from robopose.lib3d.robopose_ops import loss_refiner_CO_disentangled_reference_point
from robopose.lib3d.transform_ops import add_noise, invert_T, transform_pts

from robopose.lib3d.robopose_ops import (
    TCO_init_from_boxes_zup_autodepth,
)
from robopose.lib3d.robot_ops import add_noise_joints
from robopose.lib3d.articulated_mesh_database import Meshes


logger = get_logger(__name__)

def cast(obj, dtype=None):
    if isinstance(obj, (dict, OrderedDict)):
        for k, v in obj.items():
            obj[k] = cast(torch.as_tensor(v))
            if dtype is not None:
                obj[k] = obj[k].to(dtype)
        return obj
    else:
        return obj.cuda(non_blocking=True)


def obj_infos_to_tensor(urdf_layer, obj_infos):
    q = []
    for n in range(len(obj_infos)):
        q.append(urdf_layer.to_tensor(obj_infos[n]['joints']))
    q = torch.cat(q, dim=0)
    return q

def h_pose(model, data, meters, cfg,
           n_iterations=1, mesh_db=None, train=True):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    dtype, device = torch.float32, 'cuda'
    images = cast(data.images).float() / 255.
    batch_size, _, h, w = images.shape

    TCO_gt = cast(data.TCO).float()
    K = cast(data.K).float()
    bboxes = cast(data.bboxes).float()

    # Convert joints dict to tensor
    obj_infos_gt = data.objects
    for n in range(batch_size):
        name = obj_infos_gt[n]['name']
        urdf_layer = mesh_db.urdf_layers[mesh_db.label_to_id[name]]
        obj_infos_gt[n]['joints'] = {k: torch.as_tensor(obj_infos_gt[n]['joints'][k]).view(1, -1).to(dtype) for k in urdf_layer.joint_names}

    # Compute input pose/joints by adding noise to the ground truth
    ## Joint initialization
    obj_infos_init = deepcopy(obj_infos_gt)
    if cfg.predict_joints:
        for n in range(batch_size):
            name = obj_infos_gt[n]['name']
            urdf_layer = mesh_db.urdf_layers[mesh_db.label_to_id[name]]
            if cfg.input_generator == 'gt+noise':
                q_limits = urdf_layer.joint_limits
                q0 = cast(obj_infos_gt[n]['joints'], dtype=dtype)
                q0_tensor = urdf_layer.to_tensor(q0)
                q0_tensor = add_noise_joints(q0_tensor,
                                             std_interval_ratio=cfg.joints_std_interval_ratio,
                                             q_limits=q_limits)
            elif cfg.input_generator == 'fixed':
                q_default = urdf_layer.joints_default.unsqueeze(0).to(dtype)
                q0_tensor = urdf_layer.to_tensor(q_default)
            else:
                raise ValueError
            obj_infos_init[n]['joints'] = urdf_layer.from_tensor(q0_tensor)

    # Pose initialization
    meshes = model_without_ddp.mesh_db.select(obj_infos_init)
    _, T_O_CENTROID = meshes.center_meshes()
    T_C_CENTROID_gt = TCO_gt @ T_O_CENTROID

    if cfg.input_generator == 'gt+noise':
        T_C_CENTROID_init = add_noise(T_C_CENTROID_gt,
                                      euler_deg_std=[60, 60, 60],
                                      trans_std=[0.1, 0.1, 0.1])
    elif cfg.input_generator == 'fixed':
        centered_meshes = Meshes(meshes.labels, transform_pts(invert_T(T_O_CENTROID), meshes.points))
        centered_points = centered_meshes.sample_points(2000, deterministic=True)
        T_C_CENTROID_init = TCO_init_from_boxes_zup_autodepth(bboxes, centered_points, K)
    else:
        raise ValueError

    TCO_init = T_C_CENTROID_init @ invert_T(T_O_CENTROID)

    # Cast joints to gpu
    for n in range(batch_size):
        name = obj_infos_gt[n]['name']
        urdf_layer = mesh_db.urdf_layers[mesh_db.label_to_id[name]]
        obj_infos_gt[n]['joints'] = cast(obj_infos_gt[n]['joints'], dtype=dtype)
        obj_infos_init[n]['joints'] = cast(obj_infos_init[n]['joints'], dtype=dtype)

    # Forward pass
    outputs = model(images=images, K=K, obj_infos=obj_infos_init,
                    TCO=TCO_init, n_iterations=n_iterations,
                    update_obj_infos=cfg.predict_joints)

    losses_TCO_iter = []
    losses_q_iter = []
    losses_iter = []
    q_gt = obj_infos_to_tensor(urdf_layer, obj_infos_gt)
    for n in range(n_iterations):
        iter_outputs = outputs[f'iteration={n+1}']
        K_crop = iter_outputs['K_crop']
        refiner_outputs = iter_outputs['refiner_outputs']
        obj_infos_input = iter_outputs['obj_infos_input']

        # Pose loss
        anchor_link_names = iter_outputs['anchor_link_names']
        if cfg.points_for_pose_loss == 'anchor_link':
            link_meshes = mesh_db.select(obj_infos_input, link_names=iter_outputs['anchor_link_names'], apply_fk=False)
            anchor_loss_pts = link_meshes.sample_points(min(cfg.n_points_loss, link_meshes.points.shape[1]), deterministic=False)
        elif cfg.points_for_pose_loss == 'whole_robot':
            robot_meshes = mesh_db.select(obj_infos_input, apply_fk=True)
            anchor_loss_pts = robot_meshes.sample_points(min(cfg.n_points_loss, robot_meshes.points.shape[1]), deterministic=False)
            assert all([anchor_link_names[n] == urdf_layer.robot.base_link.name for n in range(batch_size)])
        else:
            raise ValueError(cfg.points_for_pose_loss)

        TOA_gt = urdf_layer.compute_link_pose(anchor_link_names, q_gt)
        TCA_gt = TCO_gt @ TOA_gt

        TCA_input = iter_outputs['TCA_input']
        t_C_REF = iter_outputs['t_C_REF']
        refiner_pose_update = refiner_outputs['pose']

        loss_TCO_iter = loss_refiner_CO_disentangled_reference_point(
            TCO_possible_gt=TCA_gt.unsqueeze(1),
            TCO_input=TCA_input,
            refiner_outputs=refiner_pose_update,
            K_crop=K_crop,
            points=anchor_loss_pts,
            tCR=t_C_REF,
        )

        # Joints loss
        q_output = obj_infos_to_tensor(urdf_layer, iter_outputs['obj_infos_output_no_clamp'])

        if cfg.predict_joints:
            loss_q_iter = ((q_output - q_gt) ** 2).mean(dim=-1)
            meters[f'loss_q-iter={n+1}'].add(loss_q_iter.mean().item())
            losses_q_iter.append(loss_q_iter)

        if model_without_ddp.debug:
            from robopose.lib3d.camera_geometry import project_points
            model_without_ddp.tmp_debug['pts_proj_gt'] = project_points(anchor_loss_pts, K_crop, TCA_gt)
            model_without_ddp.tmp_debug['pts_proj_input'] = project_points(anchor_loss_pts, K_crop, TCA_input)

        meters[f'loss_TCO-iter={n+1}'].add(loss_TCO_iter.mean().item())
        losses_TCO_iter.append(loss_TCO_iter)

    losses_TCO_iter = torch.cat(losses_TCO_iter)
    loss_TCO = losses_TCO_iter.mean()
    if cfg.predict_joints:
        loss_q = torch.cat(losses_q_iter).mean()
        loss_q_scaled = loss_q * cfg.loss_q_lambda
        meters['loss_q'].add(loss_q.item())
        meters['loss_q_scaled'].add(loss_q_scaled.item())
        loss = loss_TCO + loss_q_scaled
    else:
        loss = loss_TCO
    meters['loss_TCO'].add(loss_TCO.item())
    meters['loss_total'].add(loss.item())
    return loss
