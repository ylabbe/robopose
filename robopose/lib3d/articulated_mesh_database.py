import numpy as np
import pandas as pd
import torch
from torch import nn

from .mesh_ops import get_meshes_bounding_boxes, sample_points
from .urdf_layer import UrdfLayer
from .transform_ops import invert_T, transform_pts
from .mesh_ops import get_meshes_center

class MeshDataBase(nn.Module):
    def __init__(self, labels, urdf_layers):
        super().__init__()
        self.labels = labels
        self.label_to_id = {l: n for n, l in enumerate(labels)}
        self.urdf_layers = nn.ModuleList(urdf_layers)
        self.label_to_urdf_layer = {l: self.urdf_layers[n] for l, n in self.label_to_id.items()}

    @staticmethod
    def from_urdf_ds(urdf_ds, n_points=20000):
        labels = []
        urdf_layers = []
        for _, object_info in urdf_ds.index.iterrows():
            layer = UrdfLayer(object_info.urdf_path,
                              n_points=n_points,
                              global_scale=object_info.scale,
                              sampler='per_link')
            labels.append(object_info.to_dict()['label'])
            urdf_layers.append(layer)
        labels = np.array(labels)
        return MeshDataBase(labels, urdf_layers).float()

    def select(self, obj_infos, link_names=None, apply_fk=True):
        labels = np.array([obj['name'] for obj in obj_infos])
        groups = pd.DataFrame(dict(labels=labels)).groupby('labels').groups
        bsz = len(obj_infos)
        device = self.urdf_layers[0].pts_TLV.device
        dtype = self.urdf_layers[0].pts_TLV.dtype

        if link_names is None:
            n_pts = self.urdf_layers[0].pts_TLV.shape[0]
        else:
            n_pts = self.urdf_layers[0].link_pts.shape[1]

        points = torch.zeros(bsz, n_pts, 3, device=device, dtype=dtype)

        for label, ids in groups.items():
            label_id = self.label_to_id[label]
            layer = self.urdf_layers[label_id]
            joints = dict()
            for k in layer.joint_names:
                joints[k] = torch.cat([obj_infos[i]['joints'][k] for i in ids], dim=0).to(device).to(dtype)
            if link_names is None:
                if apply_fk:
                    points[ids] = layer(joints)
                else:
                    raise NotImplementedError
            else:
                if apply_fk:
                    points[ids] = layer.compute_per_link_fk(link_names=link_names, q=joints)
                else:
                    points[ids] = layer.get_link_pts(link_names)
        return Meshes(labels=labels, points=points)


class Meshes(nn.Module):
    def __init__(self, labels, points):
        super().__init__()
        self.labels = labels
        self.register_buffer('points', points)

    def center_meshes(self):
        T_offset = get_meshes_center(self.points)
        points = transform_pts(invert_T(T_offset), self.points)
        return Meshes(
            labels=self.labels,
            points=points,
        ), T_offset

    def __len__(self):
        return len(self.labels)

    def sample_points(self, n_points, deterministic=False):
        return sample_points(self.points, n_points, deterministic=deterministic)
