import pandas as pd
import numpy as np
import torch
from collections import defaultdict
import robopose.utils.tensor_collection as tc
from robopose.lib3d.transform_ops import invert_T
from robopose.datasets.utils import make_masks_from_det
from robopose.datasets.augmentations import CropResizeToAspectAugmentation
import robopose.utils.tensor_collection as tc


def make_articulated_input_infos(rgb_uint8, robot_label, bbox=None, focal=1000, resize=(640, 480)):
    rgb_uint8 = np.asarray(rgb_uint8)
    h, w, _ = rgb_uint8.shape
    K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]])
    camera = dict(K=K, T0C=np.eye(4), TWC=np.eye(4), resolution=(w, h))
    if bbox is None:
        margin = 0
        h, w, _ = np.array(rgb_uint8).shape
        keypoints_2d = np.array([[w*margin, h*margin],
                                 [w-w*margin, h-h*margin]])
        bbox = np.concatenate([np.min(keypoints_2d, axis=0), np.max(keypoints_2d, axis=0)])
    mask = make_masks_from_det(np.array(bbox)[None], h, w).numpy().astype(np.uint8)[0] * 255
    robot = dict(joints=None, name=robot_label, id_in_segm=255, bbox=bbox, TWO=np.eye(4))
    state = dict(objects=[robot], camera=camera)
    augmentation = CropResizeToAspectAugmentation(resize=resize)
    rgb, mask, state = augmentation(rgb_uint8, mask, state)
    det_infos = [dict(label=robot_label, score=1.0, batch_im_id=0)]
    detections = tc.PandasTensorCollection(
        infos=pd.DataFrame(det_infos),
        bboxes=torch.as_tensor(state['objects'][0]['bbox']).float().cuda().unsqueeze(0),
    )

    images = torch.tensor(np.array(rgb)).cuda().float().unsqueeze(0).permute(0, 3, 1, 2) / 255
    K = torch.tensor(state['camera']['K']).float().cuda().unsqueeze(0)
    return images, K, detections



def parse_obs_data(obs, parse_joints=False):
    data = defaultdict(list)
    frame_info = obs['frame_info']
    TWC = torch.as_tensor(obs['camera']['TWC']).float()
    for n, obj in enumerate(obs['objects']):
        info = dict(frame_obj_id=n,
                    label=obj['name'],
                    visib_fract=obj.get('visib_fract', 1),
                    scene_id=frame_info['scene_id'],
                    view_id=frame_info['view_id'])
        data['infos'].append(info)
        data['TWO'].append(obj['TWO'])
        data['bboxes'].append(obj['bbox'])
        data['keypoints_2d'].append(obj.get('keypoints_2d', []))
        data['TCO_keypoints_3d'].append(obj.get('TCO_keypoints_3d', []))
        data['points_3d'].append(obj.get('keypoints_2d', []))

    joints = None
    if parse_joints:
        objects = obs['objects']
        joint_names = list(objects[0]['joints'].keys())
        joints = torch.stack([torch.tensor([obj['joints'][k] for k in joint_names]) for obj in obs['objects']])

    for k, v in data.items():
        if k != 'infos':
            data[k] = torch.stack([torch.as_tensor(x) .float()for x in v])

    data['infos'] = pd.DataFrame(data['infos'])
    TCO = invert_T(TWC).unsqueeze(0) @ data['TWO']

    data = tc.PandasTensorCollection(
        infos=data['infos'],
        TCO=TCO,
        bboxes=data['bboxes'],
        keypoints_2d=data['keypoints_2d'],
        TCO_keypoints_3d=data['TCO_keypoints_3d'],
        poses=TCO,
    )
    if parse_joints:
        data.register_tensor('joints', joints)
        data.infos['joint_names'] = [joint_names for _ in range(len(data))]
    return data


def data_to_pose_model_inputs(data):
    TXO = data.poses
    has_joints = hasattr(data, 'joints')
    obj_infos = []
    for n in range(len(data)):
        obj_info = dict(name=data.infos.loc[n, 'label'])
        if has_joints:
            joint_names = data.infos.loc[n, 'joint_names']
            joints = {joint_names[i]: data.joints[n, [i]].view(1, 1) for i in range(len(joint_names))}
            obj_info.update(joints=joints)
        obj_infos.append(obj_info)
    return TXO, obj_infos
