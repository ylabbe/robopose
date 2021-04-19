import json
import numpy as np
from collections import defaultdict
import pandas as pd
from PIL import Image
import pickle as pkl
from pathlib import Path
from joblib import Memory

from numpy import sin, cos
from robopose.lib3d import Transform
from robopose.datasets.utils import make_masks_from_det
from robopose.config import MEMORY
from robopose.utils.logging import get_logger
from robopose.third_party.craves.get_2d_gt import make_2d_keypoints
logger = get_logger(__name__)


def make_rotation(pitch, yaw, roll):
    # Copied from craves repo
    # Convert from degree to radius
    # pitch = pitch / 180.0 * np.pi
    # yaw = yaw / 180.0 * np.pi
    # roll = roll / 180.0 * np.pi
    pitch = pitch
    yaw = yaw
    roll = roll
    # from: http://planning.cs.uiuc.edu/node102.html
    ryaw = [
        [-cos(yaw), sin(yaw), 0],
        [-sin(yaw), -cos(yaw), 0],
        [0, 0, 1]
    ]
    rpitch = [
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)]
    ]
    rroll = [
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)]
    ]
    T = np.matrix(ryaw) * np.matrix(rpitch) * np.matrix(rroll)
    return T


def parse_name(name):
    split = name.split('_')
    if len(split) > 1:
        scene_id = '_'.join(split[:-1])
        view_id = split[-1]
    else:
        view_id = split[0]
        scene_id = view_id
    return scene_id, view_id

@MEMORY.cache
def build_frame_index(ds_dir, save_file, split):
    if 'lab' in split or 'synt' in split:
        cam_dir = ds_dir / 'FusionCameraActor3_2'
        if 'real' in split or split == 'synt_train':
            im_dir = cam_dir / 'lit'
        else:
            im_dir = cam_dir / 'syn'
    elif 'youtube' in split:
        im_dir = ds_dir / 'imgs'
    assert im_dir.exists()

    im_paths = []
    for ext in ('*.jpg', '*.png'):
        im_paths.extend(list(im_dir.glob(ext)))

    infos = defaultdict(list)
    for n, im_path in enumerate(im_paths):
        scene_id, view_id = parse_name(im_path.with_suffix('').name)
        infos['rgb_path'].append(im_path.relative_to(ds_dir).as_posix())
        infos['scene_id'].append(scene_id)
        infos['view_id'].append(view_id)

    infos = pd.DataFrame(infos)
    save_file.write_bytes(pkl.dumps(infos))
    return


class CRAVESDataset:
    def __init__(self, ds_root, split='lab_test_real'):

        ds_root = Path(ds_root)

        self.annotation_types = dict(
            segmentation=False,
            keypoints_2d=False,
            keypoints_3d=False,
            angles=False,
            cam_info=False
        )
        if 'lab_test' in split:
            ds_dir = ds_root / 'test_20181024'
            self.annotation_types['angles'] = True
            self.annotation_types['segmentation'] = True
            self.annotation_types['cam_info'] = True
            self.annotation_types['keypoints_3d'] = True
        elif split == 'youtube':
            ds_dir = ds_root / 'youtube_20181105'
            self.annotation_types['keypoints_2d'] = True
        elif split == 'synt_train':
            ds_dir = ds_root / '20181107'
            self.annotation_types['angles'] = True
            self.annotation_types['segmentation'] = True
            self.annotation_types['cam_info'] = True
            self.annotation_types['keypoints_3d'] = True
        else:
            raise NotImplementedError
        self.ds_dir = ds_dir

        # self.memory = Memory(CACHE_DIR)
        save_file = self.ds_dir / f'index_{split}.pkl'
        # self.build_frame_index = self.memory.cache(self.build_frame_index)
        build_frame_index(ds_dir=ds_dir, save_file=save_file, split=split)
        self.frame_index = pkl.loads(save_file.read_bytes())
        self.image_bbox = False
        self.focal_length_when_unknown = 500

    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, frame_id):
        row = self.frame_index.iloc[frame_id]
        rgb_path = self.ds_dir / row.rgb_path
        rgb = np.asarray(Image.open(rgb_path))[..., :3]
        h, w = rgb.shape[:2]

        mask = None
        bbox = None
        keypoints_2d = np.zeros((17, 2), dtype=np.float) * float('nan')
        if self.annotation_types['segmentation']:
            mask = np.asarray(Image.open(rgb_path.parent.parent / 'seg' / rgb_path.with_suffix('.png').name))
            mask = (mask[..., 0] == 0).astype(np.uint8) * 255
            ids = np.where(mask)
            x1, y1, x2, y2 = np.min(ids[1]), np.min(ids[0]), np.max(ids[1]), np.max(ids[0])
            bbox = np.array([x1, y1, x2, y2])

        if self.annotation_types['keypoints_2d']:
            infos = json.loads((rgb_path.parent.parent / 'd3_preds' / rgb_path.with_suffix('.json').name).read_text())
            x1, x2, y1, y2 = np.array(infos['bbox']).flatten()
            bbox = np.array([x1, y1, x2, y2])

            mask = make_masks_from_det(bbox[None], h, w).numpy().astype(np.uint8)[0] * 255

            pts = np.transpose(np.array(infos['reprojection']))
            if 'visibility' in infos:
                visibility = infos['visibility'][:-2]
                pts[np.invert(visibility), :] = -1.0
            keypoints_2d = pts

        elif self.annotation_types['keypoints_3d']:
            cam_infos = json.loads((rgb_path.parent.parent / 'caminfo' / rgb_path.with_suffix('.json').name).read_text())
            vertex_infos = json.loads((rgb_path.parent.parent.parent / 'vertex' / rgb_path.with_suffix('.json').name).read_text())
            joint_infos = json.loads((rgb_path.parent.parent.parent / 'joint' / rgb_path.with_suffix('.json').name).read_text())
            keypoints_2d = make_2d_keypoints(vertex_infos, joint_infos, cam_infos)

        if self.annotation_types['cam_info']:
            infos = json.loads((rgb_path.parent.parent / 'caminfo' / rgb_path.with_suffix('.json').name).read_text())
            H, W = infos['FilmHeight'], infos['FilmWidth']
            assert rgb.shape[:2] == (H, W)
            fov = infos['Fov'] * np.pi / 180
            f = W / (2 * np.tan(fov / 2))  # From UE, differs from pybullet.
            trans = np.array([infos['Location'][k] for k in ('X', 'Y', 'Z')]) * 0.001
            pitch, yaw, roll = np.array([infos['Rotation'][k] for k in ('Pitch', 'Yaw', 'Roll')])
            trans[1] *= -1
            yaw = - 180 + yaw
            roll = - roll
            pitch = - pitch
            pitch, yaw, roll = np.array([pitch, yaw, roll]) * np.pi / 180
            R = np.asarray(make_rotation(pitch, -yaw, -roll))
            R_NORMAL_UE = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0],
            ])
            R = R @ R_NORMAL_UE.T
            T0C = Transform(R, trans).toHomogeneousMatrix()
        else:
            f = self.focal_length_when_unknown
            T0C = np.zeros((4, 4))
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
        cam = dict(K=K, T0C=T0C, TWC=T0C, resolution=(w, h))

        angles = None
        if self.annotation_types['angles']:
            angles = json.loads((rgb_path.parent.parent.parent / 'angles' / rgb_path.with_suffix('.json').name).read_text())
            q = np.array(angles)[:4] * np.pi / 180
            q[0] *= -1
        else:
            q = np.array([np.nan for _ in range(4)])
        q = {joint_name: q[n] for n, joint_name in enumerate(['Model_Rotation', 'Rotation_Base', 'Base_Elbow', 'Elbow_Wrist'])}

        obs = dict()
        TWO = Transform((0, 0, 0, 1), (0, 0, 0)).toHomogeneousMatrix()
        robot = dict(joints=q, name='owi535', id_in_segm=255,
                     bbox=bbox, TWO=TWO, keypoints_2d=keypoints_2d)
        obs['objects'] = [robot]
        obs['camera'] = cam
        obs['frame_info'] = row.to_dict()
        return rgb, mask, obs
