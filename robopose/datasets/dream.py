import pandas as pd
import json
from tqdm import tqdm
from collections import OrderedDict
from robopose.datasets.utils import make_masks_from_det
import numpy as np
from PIL import Image

from collections import defaultdict
from pathlib import Path

from robopose.config import MEMORY
from robopose.lib3d import Transform


KUKA_SYNT_TRAIN_DR_INCORRECT_IDS = {83114, 28630, }
@MEMORY.cache
def build_frame_index(base_dir):
    im_paths = base_dir.glob('*.jpg')
    infos = defaultdict(list)
    for n, im_path in tqdm(enumerate(sorted(im_paths))):
        view_id = int(im_path.with_suffix('').with_suffix('').name)
        if 'kuka_synth_train_dr' in str(base_dir) and int(view_id) in KUKA_SYNT_TRAIN_DR_INCORRECT_IDS:
            pass
        else:
            scene_id = view_id
            infos['rgb_path'].append(im_path.as_posix())
            infos['scene_id'].append(scene_id)
            infos['view_id'].append(view_id)

    infos = pd.DataFrame(infos)
    return infos


class DreamDataset:
    def __init__(self, base_dir, image_bbox=False):
        self.base_dir = Path(base_dir)
        self.frame_index = build_frame_index(self.base_dir)

        self.joint_map = dict()
        if 'panda' in str(base_dir):
            self.keypoint_names = [
                'panda_link0', 'panda_link2', 'panda_link3',
                'panda_link4', 'panda_link6', 'panda_link7',
                'panda_hand',
            ]
            self.label = 'panda'
        elif 'baxter' in str(base_dir):
            self.keypoint_names = [
                'torso_t0', 'left_s0', 'left_s1',
                'left_e0', 'left_e1', 'left_w0',
                'left_w1', 'left_w2', 'left_hand',
                'right_s0', 'right_s1', 'right_e0',
                'right_e1', 'right_w0', 'right_w1',
                'right_w2', 'right_hand'
            ]
            self.label = 'baxter'
        elif 'kuka' in str(base_dir):
            self.keypoint_names = [
                'iiwa7_link_0', 'iiwa7_link_1',
                'iiwa7_link_2', 'iiwa7_link_3',
                'iiwa7_link_4', 'iiwa7_link_5',
                'iiwa7_link_6', 'iiwa7_link_7',
            ]
            self.label = 'iiwa7'
        else:
            raise NotImplementedError

        self.scale = 0.01 if 'synthetic' in str(self.base_dir) else 1.0
        self.all_labels = [self.label]
        self.image_bbox = image_bbox

    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, idx):
        row = self.frame_index.iloc[idx]
        rgb_path = Path(row.rgb_path)
        rgb = np.asarray(Image.open(rgb_path))
        assert rgb.shape[-1] == 3, rgb_path

        mask = None
        annotations = json.loads(rgb_path.with_suffix('').with_suffix('.json').read_text())

        # Camera
        TWC = np.eye(4)
        camera_infos_path = self.base_dir / '_camera_settings.json'
        h, w = rgb.shape[0], rgb.shape[1]
        if camera_infos_path.exists():
            cam_infos = json.loads(camera_infos_path.read_text())
            assert len(cam_infos['camera_settings']) == 1
            cam_infos = cam_infos['camera_settings'][0]['intrinsic_settings']
            fx, fy, cx, cy = [cam_infos[k] for k in ('fx', 'fy', 'cx', 'cy')]
        else:
            fx, fy = 320, 320
            cx, cy = w/2, h/2

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        camera = dict(
            TWC=TWC,
            resolution=(w, h),
            K=K,
        )
        label = self.label

        # Objects
        obj_data = annotations['objects'][0]
        if 'quaternion_xyzw' in obj_data:
            TWO = Transform(np.array(obj_data['quaternion_xyzw']),
                            np.array(obj_data['location']) * self.scale).toHomogeneousMatrix()
            R_NORMAL_UE = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0],
            ])
            TWO[:3, :3] = TWO[:3, :3] @ R_NORMAL_UE
        else:
            TWO = np.eye(4) * float('nan')
        TWO = Transform(TWO)

        joints = annotations['sim_state']['joints']
        joints = OrderedDict({d['name'].split('/')[-1]: d['position'] for d in joints})
        if self.label == 'iiwa7':
            joints = {k.replace('iiwa7_', 'iiwa_'): v for k,v in joints.items()}

        keypoints_2d = obj_data['keypoints']
        keypoints_2d = np.concatenate([np.array(kp['projected_location'])[None] for kp in keypoints_2d], axis=0)
        keypoints_2d = np.unique(keypoints_2d, axis=0)
        valid = np.logical_and(np.logical_and(keypoints_2d[:, 0] >= 0, keypoints_2d[:, 0] <= w - 1),
                               np.logical_and(keypoints_2d[:, 1] >= 0, keypoints_2d[:, 1] <= h - 1))
        keypoints_2d = keypoints_2d[valid]
        det_valid = len(keypoints_2d) >= 2
        if det_valid:
            bbox = np.concatenate([np.min(keypoints_2d, axis=0), np.max(keypoints_2d, axis=0)])
            dx, dy = bbox[[2, 3]] - bbox[[0, 1]]
            if dx <= 20 or dy <= 20:
                det_valid = False

        if not det_valid or self.image_bbox:
            m = 1/5
            keypoints_2d = np.array([[w*m, h*m], [w-w*m, h-h*m]])
        bbox = np.concatenate([np.min(keypoints_2d, axis=0), np.max(keypoints_2d, axis=0)])
        mask = make_masks_from_det(bbox[None], h, w).numpy().astype(np.uint8)[0] * 1

        keypoints = obj_data['keypoints']
        TCO_keypoints_3d = {kp['name']: np.array(kp['location']) * self.scale for kp in keypoints}
        TCO_keypoints_3d = np.array([TCO_keypoints_3d.get(k, np.nan) for k in self.keypoint_names])
        keypoints_2d = {kp['name']: kp['projected_location'] for kp in keypoints}
        keypoints_2d = np.array([keypoints_2d.get(k, np.nan) for k in self.keypoint_names])

        robot = dict(label=label, name=label, joints=joints,
                     TWO=TWO.toHomogeneousMatrix(), bbox=bbox,
                     id_in_segm=1,
                     keypoints_2d=keypoints_2d,
                     TCO_keypoints_3d=TCO_keypoints_3d)

        state = dict(
            objects=[robot],
            camera=camera,
            frame_info=row.to_dict()
        )
        return rgb, mask, state
