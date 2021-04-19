import numpy as np
from tqdm import tqdm
from copy import deepcopy

from .plotter import Plotter

from robopose.datasets.wrappers.augmentation_wrapper import AugmentationWrapper
from robopose.datasets.augmentations import CropResizeToAspectAugmentation


def render_prediction_wrt_camera(renderer, pred, camera=None, resolution=(640, 480), K=None):
    assert len(pred) == 1
    pred = pred.cpu()

    if camera is not None:
        camera = deepcopy(camera)
    else:
        K = pred.K[0].numpy()
        camera = dict(
            K=K,
            resolution=resolution
        )
    camera.update(TWC=np.eye(4))

    list_objects = []
    for n in range(len(pred)):
        row = pred.infos.iloc[n]
        joint_names = row.joint_names
        joints = {joint_names[i]: pred.joints[n][i].item() for i in range(len(joint_names))}
        obj = dict(
            name=row.label,
            color=(1, 1, 1, 1),
            TWO=pred.poses[n].numpy(),
            joints=joints,
        )
        list_objects.append(obj)
    renders = renderer.render_scene(list_objects, [camera])[0]['rgb']

    # renders = renderer.render_scene(list_objects, [camera])[0]['mask_int']
    # renders[renders == (0 + 5 << 24)] = 255
    # renders[renders != 255] = 0
    # renders = renders.astype(np.uint8)
    # renders = renders[..., None]
    # renders = np.repeat(renders, 3, axis=-1)
    return renders


def render_gt(renderer, objects, camera):
    camera = deepcopy(camera)
    TWC = camera['TWC']
    for obj in objects:
        obj['TWO'] = np.linalg.inv(TWC) @ obj['TWO']
    camera['TWC'] = np.eye(4)
    rgb_rendered = renderer.render_scene(objects, [camera])[0]['rgb']
    return rgb_rendered


def make_singleview_prediction_plots(scene_ds, renderer, one_prediction):
    plotter = Plotter()

    assert len(one_prediction) == 1
    predictions = one_prediction
    scene_id, view_id = np.unique(predictions.infos['scene_id']).item(), np.unique(predictions.infos['view_id']).item()

    scene_ds_index = scene_ds.frame_index
    scene_ds_index['ds_idx'] = np.arange(len(scene_ds_index))
    scene_ds_index = scene_ds_index.set_index(['scene_id', 'view_id'])
    idx = scene_ds_index.loc[(scene_id, view_id), 'ds_idx']

    augmentation = CropResizeToAspectAugmentation(resize=(640, 480))
    scene_ds = AugmentationWrapper(scene_ds, augmentation)
    rgb_input, mask, state = scene_ds[idx]

    figures = dict()

    figures['input_im'] = plotter.plot_image(rgb_input)

    # gt_rendered = render_gt(renderer, state['objects'], state['camera'])
    # figures['gt_rendered'] = plotter.plot_image(gt_rendered)
    # figures['gt_overlay'] = plotter.plot_overlay(rgb_input, gt_rendered)

    renders = render_prediction_wrt_camera(renderer, predictions)
    # pred_rendered = renders['rgb']
    pred_rendered = renders
    # pred_rendered[renders['mask_int'] == -1] = 255
    # print(pred_rendered)
    figures['pred_rendered'] = plotter.plot_image(pred_rendered)
    figures['pred_overlay'] = plotter.plot_overlay(rgb_input, pred_rendered)
    # figures['pred_mask_int'] = renders['mask_int']
    figures['rgb_input'] = rgb_input
    figures['rgb_overlay'] = plotter.make_overlay(rgb_input, pred_rendered)
    return figures


def make_video(scene_ds, renderer, predictions, view_ids=None):
    plotter = Plotter()
    if view_ids is None:
        view_ids = np.sort(predictions.infos.view_id.tolist())
    # view_ids = view_ids[::30]

    augmentation = CropResizeToAspectAugmentation(resize=(640, 480))
    scene_ds = AugmentationWrapper(scene_ds, augmentation)

    images = []
    for view_id in tqdm(view_ids):
        scene_ds_index = scene_ds.frame_index
        scene_ds_index['ds_idx'] = np.arange(len(scene_ds_index))
        scene_ds_index = scene_ds_index.set_index(['view_id'])
        idx = scene_ds_index.loc[(view_id), 'ds_idx']

        rgb_input, mask, state = scene_ds[idx]
        pred_idx = np.where(predictions.infos['view_id'] == view_id)[0].item()
        renders = render_prediction_wrt_camera(renderer, predictions[[pred_idx]])
        overlay = plotter.make_overlay(rgb_input, renders)

        # Rotation
        images.append((np.asarray(rgb_input), np.asarray(overlay)))
    return images
