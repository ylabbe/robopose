import numpy as np
from pathlib import Path
import pandas as pd
import trimesh
import torch
from urdfpytorch import URDF

from robopose.utils.random import temp_numpy_seed
from robopose.config import OWI_KEYPOINTS_PATH, PANDA_KEYPOINTS_PATH
from robopose.lib3d.transform_ops import invert_T, transform_pts


def clamp(x, t_min, t_max):
    return torch.max(torch.min(x, t_max), t_min)


def pad_stack_tensors(tensor_list, fill='select_random', deterministic=True):
    n_max = max([t.shape[0] for t in tensor_list])
    if deterministic:
        np_random = np.random.RandomState(0)
    else:
        np_random = np.random
    tensor_list_padded = []
    for tensor_n in tensor_list:
        n_pad = n_max - len(tensor_n)

        if n_pad > 0:
            if isinstance(fill, torch.Tensor):
                assert isinstance(fill, torch.Tensor)
                assert fill.shape == tensor_n.shape[1:]
                pad = fill.unsqueeze(0).repeat(n_pad, *[1 for _ in fill.shape]).to(tensor_n.device).to(tensor_n.dtype)
            else:
                assert fill == 'select_random'
                ids_pad = np_random.choice(np.arange(len(tensor_n)), size=n_pad)
                pad = tensor_n[ids_pad]
            tensor_n_padded = torch.cat((tensor_n, pad), dim=0)
        else:
            tensor_n_padded = tensor_n
        tensor_list_padded.append(tensor_n_padded)
    return torch.stack(tensor_list_padded)

class UrdfLayer(torch.nn.Module):
    def __init__(self, urdf_path, n_points=20000,
                 keypoints_description_path=OWI_KEYPOINTS_PATH,
                 global_scale=1.0, sampler='per_link'):
        """
        Only supports meshes: cylinders/boxes/spheres won't be considered.
        """
        assert sampler in {'uniform', 'per_link'}
        super().__init__()
        robot = URDF.load(Path(urdf_path).as_posix())
        meshes = []
        mesh_filenames = []
        link_names = []
        face_mesh_ids = []
        TLV = []
        link_ids = []
        link_name_to_id = {}
        for link_id, link in enumerate(robot.links):
            link_name_to_id[link.name] = link_id
            for visual in link.visuals:
                if visual.geometry.mesh is not None:
                    for mesh in visual.geometry.mesh.meshes:
                        mesh_id = len(meshes)
                        TLV.append(visual.origin)
                        link_names.append(link.name)
                        face_mesh_ids.append([mesh_id for _ in range(len(mesh.faces))])
                        mesh_filenames.append(visual.geometry.mesh.filename)
                        link_ids.append(link_id)
                        meshes.append(mesh)
        self.link_name_to_id = link_name_to_id

        mesh_infos = pd.DataFrame(dict(mesh_id=mesh_id, link_id=link_ids, robot_id=0))
        groupby_key = 'link_id' if sampler == 'per_link' else 'robot_id'
        groups = mesh_infos.groupby(groupby_key).groups

        pts = []
        mesh_ids = []
        for _, group_ids in groups.items():
            meshes_ = trimesh.util.concatenate([meshes[i] for i in group_ids])
            face_mesh_ids_ = np.concatenate([face_mesh_ids[i] for i in group_ids])
            with temp_numpy_seed(0):
                pts_, face_ids_ = trimesh.sample.sample_surface(meshes_, n_points // len(groups))
            assert len(meshes_.faces) == len(face_mesh_ids_)
            pts.append(pts_)
            mesh_ids.append(face_mesh_ids_[face_ids_])
        pts = np.concatenate(pts, axis=0)
        mesh_ids = np.concatenate(mesh_ids)

        pts = pts * global_scale

        TLV = torch.as_tensor(np.stack(TLV))
        TLV[..., :3, -1] *= global_scale
        self.robot = robot
        self.link_names = np.array([link.name for link in robot.links])
        self.joint_names = np.array([joint.name for joint in robot.actuated_joints])
        self.global_scale = global_scale

        self.pts_link_names = np.array(link_names)[mesh_ids]
        self.pts_mesh_names = np.array(mesh_filenames)[mesh_ids]

        link_name_to_pb_link_id = dict()
        link_name_to_pb_link_id[self.robot.joints[0].parent] = 0
        n = 1
        for joint in self.robot.joints:
            link_name_to_pb_link_id[joint.child] = n
            n += 1
        self.link_name_to_pb_link_id = link_name_to_pb_link_id

        joint_limits = np.array([(j.limit.lower, j.limit.upper) for j in robot.actuated_joints]).transpose()
        joints_default = np.mean(joint_limits, axis=0)
        self.register_buffer('joints_default', torch.tensor(joints_default))
        self.register_buffer('joint_limits', torch.tensor(joint_limits))
        self.register_buffer('pts', torch.tensor(pts).unsqueeze(-1))
        self.register_buffer('pts_link_ids', torch.tensor(link_ids)[mesh_ids])
        self.register_buffer('pts_TLV', TLV[mesh_ids])
        self.register_buffer('TLV', TLV)

        joint_child_visual_pts = []
        for joint in robot.actuated_joints:
            child_link_name = joint.child
            child_link_id = link_name_to_id[child_link_name]
            child_point_mask = self.pts_link_ids == child_link_id
            child_point_mask = child_point_mask.nonzero().flatten().tolist()
            pts_TLV = self.pts_TLV[child_point_mask]
            pts = self.pts[child_point_mask]
            child_visual_pts = pts_TLV[:, :3, :3] @ pts + pts_TLV[:, :3, [-1]]
            joint_child_visual_pts.append(child_visual_pts.squeeze(-1))
        joint_child_visual_pts = torch.stack(joint_child_visual_pts)
        self.register_buffer('joint_child_visual_pts', joint_child_visual_pts)

        link_pts = []
        self.link_to_n_pts = dict()
        for link in robot.links:
            link_name = link.name
            link_id = link_name_to_id[link_name]
            link_mask = self.pts_link_ids == link_id
            pts_TLV = self.pts_TLV[link_mask]
            pts = self.pts[link_mask]
            visual_pts = pts_TLV[:, :3, :3] @ pts + pts_TLV[:, :3, [-1]]
            self.link_to_n_pts[link.name] = len(visual_pts)
            if len(visual_pts) == 0:
                visual_pts = torch.zeros(1, 3, 1).double() * float('nan')
            link_pts.append(visual_pts.squeeze(-1))
        link_pts = pad_stack_tensors(link_pts)
        self.register_buffer('link_pts', link_pts)

        keypoint_infos = None
        keypoint_scale = 1.0
        if robot.name == 'owi535':
            keypoint_infos = pd.read_json(OWI_KEYPOINTS_PATH)
            keypoint_scale = 1
        elif robot.name == 'panda':
            keypoint_infos = pd.read_json(PANDA_KEYPOINTS_PATH)
            keypoint_scale = 1
        elif robot.name == 'baxter':
            joint_names = [
                'torso_t0', 'left_s0', 'left_s1',
                'left_e0', 'left_e1', 'left_w0',
                'left_w1', 'left_w2', 'left_hand',
                'right_s0', 'right_s1', 'right_e0',
                'right_e1', 'right_w0', 'right_w1',
                'right_w2', 'right_hand'
            ]
            joint_name_to_joint = {joint.name: joint for joint in robot.joints}
            offsets = []
            link_names = []
            for joint_name in joint_names:
                joint = joint_name_to_joint[joint_name]
                offset = joint.origin[:3, -1]
                link_name = joint.parent
                link_names.append(link_name)
                offsets.append(offset)
            keypoint_infos = pd.DataFrame(dict(
                offset=offsets,
                link_name=link_names,
            ))
            keypoint_scale = 1
        elif robot.name == 'iiwa7':
            keypoint_names = [
                'iiwa_link_0', 'iiwa_link_1',
                'iiwa_link_2', 'iiwa_link_3',
                'iiwa_link_4', 'iiwa_link_5',
                'iiwa_link_6', 'iiwa_link_7',
            ]
            keypoint_infos = pd.DataFrame(dict(
                offset=[np.array([0, 0, 0], dtype=np.float) for _ in range(len(keypoint_names))],
                link_name=keypoint_names,
            ))
            keypoint_scale = 1
        if keypoint_infos is not None:
            offset = torch.as_tensor(np.stack(keypoint_infos['offset'])).unsqueeze(0).unsqueeze(-1) * keypoint_scale
            self.kp_link_ids = [np.where(self.link_names == n)[0].item() for n in keypoint_infos['link_name']]
            self.register_buffer('kp_offsets', offset)

    def to_tensor(self, q):
        if isinstance(q, torch.Tensor):
            assert q.shape[1] == len(self.joint_names)
        elif isinstance(q, dict):
            t_q = torch.cat([q[joint_name] for joint_name in self.joint_names], dim=1)
            assert t_q.dim() == 2
            q = t_q
        return q

    def from_tensor(self, q):
        assert isinstance(q, torch.Tensor)
        q_dict = dict()
        for n, k in enumerate(self.joint_names):
            q_dict[k] = q[:, [n]]
        return q_dict

    def get_keypoints(self, q):
        q = self.to_tensor(q)
        TWL = self.get_TWL(q)
        TWL_kp = TWL[:, self.kp_link_ids]
        pts = TWL_kp[:, :, :3, :3] @ self.kp_offsets + TWL_kp[:, :, :3, [-1]]
        return pts.squeeze(-1)

    def clamp_q(self, q):
        clamped_q = clamp(q, self.joint_limits[0] + 1e-4, self.joint_limits[1] - 1e-4)
        return clamped_q

    def get_TWL(self, q):
        q = self.to_tensor(q)
        # q = self.clamp_q(q)
        fk = self.robot.link_fk_batch(q, use_names=True)
        TWL = torch.stack([fk[link] for link in self.link_names]).permute(1, 0, 2, 3)
        TWL[..., :3, -1] *= self.global_scale
        return TWL

    def compute_urdf_root_link_pose(self, anchor_poses, anchor_link_names, q):
        # Apply forward kinematics for anchor parts
        q = self.to_tensor(q)
        bsz = len(q)
        TWL = self.get_TWL(q)
        anchor_link_ids = [np.where(anchor_link_name == self.link_names)[0].item() for anchor_link_name in anchor_link_names]
        root_link_id = np.where(self.link_names == self.robot.base_link.name)[0].item()
        TW_root = TWL[:, root_link_id]
        TW_anchor = TWL[torch.arange(bsz), anchor_link_ids]
        Tanchor_root = invert_T(TW_anchor) @ TW_root
        root_poses = anchor_poses @ Tanchor_root
        return root_poses

    def compute_link_pose(self, link_names, q):
        q = self.to_tensor(q)
        bsz = len(q)
        TWL = self.get_TWL(q)
        link_ids = [np.where(link_name == self.link_names)[0].item() for link_name in link_names]
        TW_links = TWL[torch.arange(bsz), link_ids]
        return TW_links

    def compute_per_link_fk(self, link_names, q):
        TBASE_L = self.compute_link_pose(link_names, q)
        link_ids = [self.link_name_to_id[link_name] for link_name in link_names]
        pts_link = self.link_pts[link_ids]
        pts = transform_pts(TBASE_L, pts_link)
        return pts

    def get_link_pts(self, link_names):
        link_ids = [self.link_name_to_id[link_name] for link_name in link_names]
        pts_link = self.link_pts[link_ids]
        return pts_link

    def compute_per_joint_fk(self, q):
        q = self.to_tensor(q)
        dtype, device = q.dtype, q.device
        bsz, n_dof = q.shape
        TL_Lchild = []
        for i, joint_name in enumerate(self.joint_names):
            joint = self.robot.actuated_joints[i]
            assert joint.name == joint_name
            TL_Lchild.append(joint.get_child_poses(cfg=q[:, i], n_cfgs=bsz, device=device, dtype=dtype))
        TL_Lchild = torch.stack(TL_Lchild).permute(1, 0, 2, 3)
        TL_Lchild[..., :3, -1] *= self.global_scale
        pts = TL_Lchild[:, :, :3, :3].unsqueeze(-3) @ self.joint_child_visual_pts.unsqueeze(-1).unsqueeze(0) + TL_Lchild[:, :, :3, [-1]].unsqueeze(-3)
        pts = pts.squeeze(-1)
        return pts

    def forward(self, q):
        q = self.to_tensor(q)
        TWL = self.get_TWL(q)
        pts_TWL = torch.index_select(TWL, dim=1, index=self.pts_link_ids)
        pts_TWV = pts_TWL @ self.pts_TLV
        pts = pts_TWV[:, :, :3, :3] @ self.pts.unsqueeze(0) + pts_TWV[:, :, :3, [-1]]
        return pts.squeeze(-1)
