import numpy as np
import pickle as pkl
import torch
import logging
import pandas as pd
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import argparse

from robopose.datasets.datasets_cfg import make_scene_dataset, make_urdf_dataset
from robopose.datasets.wrappers.augmentation_wrapper import AugmentationWrapper
from robopose.datasets.augmentations import CropResizeToAspectAugmentation
from robopose.evaluation.runner_utils import format_results
import robopose.utils.tensor_collection as tc

from robopose.third_party.craves.load_results import load_craves_results

from robopose.lib3d.articulated_mesh_database import MeshDataBase
from robopose.lib3d.transform import Transform

from robopose.rendering.bullet_batch_renderer import BulletBatchRenderer
from robopose.training.articulated_models_cfg import create_model

from robopose.integrated.articulated_predictor import ArticulatedObjectPredictor
from robopose.evaluation.eval_runner.articulated_obj_eval import ArticulatedObjectEvaluation
from robopose.evaluation.pred_runner.articulated_predictions import ArticulatedObjectPredictionRunner
from robopose.evaluation.pred_runner.video_predictions import VideoPredictionRunner

from robopose.evaluation.meters.craves_meters import CravesErrorMeter, CravesKeypointsMeter
from robopose.evaluation.meters.dream_meters import DreamErrorMeter

from robopose.utils.distributed import get_tmp_dir, get_rank
from robopose.utils.distributed import init_distributed_mode

from robopose.training.articulated_models_cfg import check_update_config

from robopose.config import EXP_DIR, RESULTS_DIR, LOCAL_DATA_DIR
from robopose.utils.logging import get_logger
from robopose.utils.random import temp_numpy_seed

from robopose.paper_models_cfg import (
    PANDA_MODELS,
    KUKA_MODELS,
    BAXTER_MODELS,
    OWI_MODELS,
    PANDA_ABLATION_REFERENCE_POINT_MODELS,
    PANDA_ABLATION_ANCHOR_MODELS,
    PANDA_ABLATION_ITERATION_MODELS
)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = get_logger(__name__)


def patch_state_dict(state_dict):
    return {k.replace('robot_joints', 'owi535_joints'): v for k, v in state_dict.items()}


def load_model(run_id, n_plotters=8):
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config(cfg)
    urdf_ds_name = cfg.urdf_ds_name
    renderer = BulletBatchRenderer(urdf_ds_name, n_workers=n_plotters)
    urdf_ds = make_urdf_dataset(urdf_ds_name)
    mesh_db = MeshDataBase.from_urdf_ds(urdf_ds, n_points=2500).cuda().float()

    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config(cfg)
    model = create_model(cfg, renderer=renderer, mesh_db=mesh_db)
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    state_dict = patch_state_dict(ckpt['state_dict'])
    model.load_state_dict(state_dict)
    model.cfg = cfg
    model = model.cuda().eval()
    return model


def load_dream_result(ds_name, net):
    return load_dream_result_id(f'ds={ds_name}_net={net}')


def load_dream_result_id(result_id):
    dream_result_dir = LOCAL_DATA_DIR / 'dream_results' / result_id
    if not dream_result_dir.exists():
        logger.info(f'DREAM {result_id} not found ({dream_result_dir})')
        return None
    pnp_results = pd.read_csv(dream_result_dir / 'pnp_results.csv')
    x, y, z, qx, qy, qz, qw = [pnp_results.loc[:, f'pose_{k}'].values for k in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')]
    pnp_poses = []
    pnp_results['scene_id'] = pnp_results['name']
    pnp_results['view_id'] = pnp_results['name']
    scale = 1/100 if 'synt' in result_id else 1.
    for n in range(len(x)):
        T = Transform(np.array([qx[n], qy[n], qz[n], qw[n]]), np.array([x[n], y[n], z[n]]) * scale)
        pnp_poses.append(T.toHomogeneousMatrix())
    pnp_poses = torch.as_tensor(np.stack(pnp_poses))
    infos = pnp_results.loc[:, ['view_id', 'scene_id', 'pnp_success']]
    results = tc.PandasTensorCollection(
        infos=infos,
        pnp_poses=pnp_poses
    )
    return results


def load_dream_results_best(ds_name):
    ds_name_to_dream_result_id = {
        'dream.baxter.synt.dr.test': 'ds=baxter_synth_test_dr_net=baxter_dream_vgg_q',

        'dream.kuka.synt.dr.test': 'ds=kuka_synth_test_dr_net=kuka_dream_resnet_h',
        'dream.kuka.synt.photo.test': 'ds=kuka_synth_test_photo_net=kuka_dream_resnet_h',

        'dream.panda.synt.dr.test': 'ds=panda_synth_test_dr_net=panda_dream_resnet_h',
        'dream.panda.synt.photo.test': 'ds=panda_synth_test_photo_net=panda_dream_resnet_h',
        'dream.panda.real.orb': 'ds=panda-orb_net=panda_dream_vgg_f',
        'dream.panda.real.realsense': 'ds=panda-3cam_realsense_net=panda_dream_resnet_h',
        'dream.panda.real.azure': 'ds=panda-3cam_azure_net=panda_dream_vgg_f',
        'dream.panda.real.kinect360': 'ds=panda-3cam_kinect360_net=panda_dream_resnet_h',
    }
    dream_result_id = ds_name_to_dream_result_id[ds_name]
    return load_dream_result_id(dream_result_id)


def load_dream_all_results(ds_name):
    ds_name_map = {
        'dream.baxter.synt.dr.test': 'baxter_synth_test_dr',

        'dream.kuka.synt.dr.test': 'kuka_synth_test_dr',
        'dream.kuka.synt.photo.test': 'kuka_synth_test_photo',

        'dream.panda.synt.dr.test': 'panda_synth_test_dr',
        'dream.panda.synt.photo.test': 'panda_synth_test_photo',
        'dream.panda.real.orb': 'panda-orb',
        'dream.panda.real.realsense': 'panda-3cam_realsense',
        'dream.panda.real.azure': 'panda-3cam_azure',
        'dream.panda.real.kinect360': 'panda-3cam_kinect360',
    }
    dream_ds_name = ds_name_map[ds_name]
    all_results = dict()
    robot = ds_name.split('.')[1]
    for net in (f'{robot}_dream_vgg_q', f'{robot}_dream_vgg_f', f'{robot}_dream_resnet_h'):
        dream_results = load_dream_result(dream_ds_name, net)
        if dream_results is not None:
            all_results[f'known_joints/dream_net={net}'] =  dream_results
        else:
            pass
    return all_results


def get_meters(ds_name):
    meters = dict()
    if 'craves' in ds_name:
        meters.update(
            craves=CravesErrorMeter(),
            craves_keypoints=CravesKeypointsMeter()
        )
    elif 'dream' in ds_name:
        meters.update(
            dream_keypoints=DreamErrorMeter(),
        )
    else:
        pass
    return meters


def run_robot_eval(args):
    logger.info(f"{'-'*80}")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"{'-'*80}")

    scene_ds = make_scene_dataset(args.ds_name)
    if not args.online_evaluation:
        with temp_numpy_seed(0):
            scene_ds.frame_index = scene_ds.frame_index.sample(frac=1).reset_index(drop=True)
    if args.n_frames is not None:
        scene_ds.frame_index = scene_ds.frame_index.iloc[:args.n_frames]

    scene_ds.image_bbox = True
    det_prefix = 'full_image_detections'

    if args.ds_name == 'craves.youtube' and args.focal_length is not None:
        scene_ds.focal_length_when_unknown = args.focal_length

    augmentation = CropResizeToAspectAugmentation(resize=(640, 480))
    scene_ds_pred = AugmentationWrapper(scene_ds, augmentation)

    pred_kwargs = dict()
    predictor = None

    if args.online_evaluation:
        pred_runner = VideoPredictionRunner(scene_ds_pred,
                                            cache_data=False,
                                            n_workers=args.n_workers)
    else:
        pred_runner = ArticulatedObjectPredictionRunner(scene_ds_pred,
                                                        batch_size=args.pred_bsz,
                                                        cache_data=len(pred_kwargs) > 1,
                                                        n_workers=args.n_workers)

    if not args.skip_model_predictions:
        model = load_model(args.run_id, n_plotters=args.n_plotters)
        predictor = ArticulatedObjectPredictor(model, bsz_objects=64).cuda()


    if not args.skip_model_predictions:
        if args.online_evaluation:
            if model.cfg.predict_joints:
                assert model.cfg.predict_joints
                pred_kwargs.update({
                        f'{det_prefix}/unknown_joints': dict(
                            obj_predictor=predictor,
                            use_gt_joints=False,
                            n_iterations_init=10,
                            n_iterations_new_image=1,
                        )})
            else:
                pred_kwargs.update({
                    f'{det_prefix}/gt_joints': dict(
                        obj_predictor=predictor,
                        use_gt_joints=True,
                        n_iterations_init=10,
                        n_iterations_new_image=1
                    )
                })

        else:
            base_pred_kwargs = dict(
                n_iterations=args.n_iterations,
            )

            pred_kwargs.update({
                f'{det_prefix}/gt_joints': dict(
                    obj_predictor=predictor,
                    use_gt_joints=True,
                    **base_pred_kwargs
                ),
            })

            if model.cfg.predict_joints:
                assert model.cfg.predict_joints
                pred_kwargs.update({
                    f'{det_prefix}/unknown_joints': dict(
                        obj_predictor=predictor,
                        use_gt_joints=False,
                        **base_pred_kwargs
                    )})

                if not args.unknown_evaluate_known:
                    del pred_kwargs[f'{det_prefix}/gt_joints']

    logger.info(f'Runner: {pred_runner}')
    logger.info(f"Predictions kwargs: {pred_kwargs.keys()}")

    meters = get_meters(args.ds_name)
    logger.info(f"Meters: {meters}")
    eval_runner = ArticulatedObjectEvaluation(scene_ds, meters,
                                              batch_size=args.eval_bsz,
                                              cache_data=len(pred_kwargs) > 1,
                                              n_workers=args.n_workers,
                                              sampler=pred_runner.sampler)

    all_predictions = dict()

    if args.external_predictions:
        if 'craves' in args.ds_name:
            all_predictions['gt_detections/unknown_joints/craves'] = load_craves_results(args.ds_name)
        elif 'dream' in args.ds_name:
            if args.eval_all_dream_models:
                dream_all_preds = load_dream_all_results(args.ds_name)
                for k in dream_all_preds.keys():
                    args.eval_keys.add(k)
                all_predictions.update(dream_all_preds)
            else:
                all_predictions['known_joints/dream'] = load_dream_results_best(args.ds_name)

    for pred_prefix, pred_kwargs_ in pred_kwargs.items():
        logger.info(f"Running predictions: {pred_prefix}")
        preds = pred_runner.get_predictions(**pred_kwargs_)
        for preds_name, preds_n in preds.items():
            all_predictions[f'{pred_prefix}/{preds_name}'] = preds_n

    torch.distributed.barrier()
    logger.info('Done with predictions.')

    eval_metrics, eval_dfs = dict(), dict()
    if not args.skip_evaluation:
        for preds_k, preds in all_predictions.items():
            do_eval = preds_k in args.eval_keys
            if do_eval:
                logger.info(f"Evaluation of predictions: {preds_k}")
                eval_metrics[preds_k], eval_dfs[preds_k] = eval_runner.evaluate(preds)
            else:
                logger.info(f"Skipped: {preds_k}")

    torch.distributed.barrier()
    logger.info("Done with Evaluation.")

    for k, v in all_predictions.items():
        all_predictions[k] = v.gather_distributed(tmp_dir=get_tmp_dir()).cpu()

    logger.info('Gathered predictions from all processes.')

    if get_rank() == 0:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f'Finished evaluation on {args.ds_name}')
        results = format_results(all_predictions, eval_metrics, eval_dfs)
        torch.save(results, save_dir / 'results.pth.tar')
        torch.save(results.get('summary'), save_dir / 'summary.pth.tar')
        torch.save(results.get('dfs'), save_dir / 'error_dfs.pth.tar')
        torch.save(results.get('metrics'), save_dir / 'metrics.pth.tar')
        (save_dir / 'summary.txt').write_text(results.get('summary_txt', ''))
        (save_dir / 'config.yaml').write_text(yaml.dump(args))
        logger.info(f'Saved predictions+metrics in {save_dir}')
    else:
        results = None

    torch.distributed.barrier()
    logger.info("Done.")
    return results


def make_default_cfg(debug=False):
    cfg = argparse.ArgumentParser('').parse_args([])

    cfg.n_workers = 8
    cfg.n_plotters = 8
    cfg.pred_bsz = 16  # 1 it/s for DREAM.
    cfg.eval_bsz = 1
    cfg.n_frames = None
    cfg.skip_evaluation = False
    cfg.skip_model_predictions = False
    cfg.online_evaluation = False
    cfg.allow_gt_bboxes = False
    cfg.unknown_evaluate_known = False
    cfg.external_predictions = True
    cfg.eval_all_dream_models = False

    cfg.n_iterations = 10
    cfg.eval_keys = set()
    cfg.save_dir = None
    cfg.ds_name = None
    cfg.run_id = None

    if debug:
        cfg.n_workers = 0
        cfg.n_plotters = 1
        cfg.pred_bsz = 1
        cfg.eval_bsz = 1
        cfg.n_frames = 10
    return cfg


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if 'robopose' in logger.name:
            logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval_all_iter', action='store_true')
    parser.add_argument('--skip_predictions', action='store_true')
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--id', default=-1, type=int)
    parser.add_argument('--datasets', default='', type=str)
    parser.add_argument('--model', default='', type=str)
    parser.add_argument('--path_external_configs', default='', type=str)
    parser.add_argument('--path_write_done', default='', type=str)
    args = parser.parse_args()

    init_distributed_mode()

    if args.path_external_configs:
        if get_rank() == 0:
            (Path(args.path_write_done).parent / 'started.txt').write_text('started.')
        cfgs = pkl.loads(Path(args.path_external_configs).read_bytes())
        for cfg in cfgs:
            run_robot_eval(cfg)
            if get_rank() == 0:
                Path(cfg.save_dir / 'done.txt').write_text('done.')
        if get_rank() == 0:
            Path(args.path_write_done).write_text('done.')
        return

    cfg = argparse.ArgumentParser('').parse_args([])

    cfg.n_workers = 8
    cfg.n_plotters = 8
    cfg.pred_bsz = 16  # 1 it/s for DREAM.
    cfg.eval_bsz = 1
    cfg.n_frames = None
    cfg.skip_evaluation = False
    cfg.skip_model_predictions = args.skip_predictions
    cfg.online_evaluation = False
    cfg.allow_gt_bboxes = False
    cfg.unknown_evaluate_known = False
    cfg.external_predictions = True
    cfg.focal_length = None
    cfg.eval_all_dream_models = False
    eval_all_iter = args.eval_all_iter

    # Compat
    args.models = args.model
    args.config = args.datasets

    if 'dream' in cfg:
        cfg.eval_bsz = cfg.pred_bsz
    else:
        cfg.eval_bsz = 1

    if args.config == 'craves-lab':
        ds_names = ['craves.lab.real.test']
    elif args.config == 'craves-youtube':
        ds_names = ['craves.youtube']
    elif args.config == 'craves':
        ds_names = [
            'craves.youtube',
            'craves.lab.real.test'
        ]
    elif args.config == 'dream-panda-kinect':
        ds_names = [
            'dream.panda.real.kinect360',
        ]
    elif args.config == 'dream-panda-orb':
        ds_names = [
            'dream.panda.real.orb',
        ]
    elif args.config == 'dream-panda-kinect-online':
        ds_names = [
            'dream.panda.real.kinect360',
        ]
        cfg.online_evaluation = True
        cfg.n_plotters = 1
    elif args.config == 'dream-panda-synt':
        ds_names = [
            'dream.panda.synt.dr.test',
        ]
    elif args.config == 'dream-panda-azure':
        ds_names = [
            'dream.panda.real.azure',
        ]
    elif args.config == 'dream-panda-realsense':
        ds_names = [
            'dream.panda.real.realsense',
        ]
    elif args.config == 'dream-panda':
        ds_names = [
            # 'dream.panda.synt.dr.train.1000',
            'dream.panda.synt.dr.test',
            'dream.panda.synt.photo.test',
            'dream.panda.real.orb',
            'dream.panda.real.kinect360',
            'dream.panda.real.realsense',
            'dream.panda.real.azure',
        ]
    elif args.config == 'dream-kuka':
        ds_names = [
            'dream.kuka.synt.dr.test',
            'dream.kuka.synt.photo.test',
        ]
    elif args.config == 'dream-baxter':
        ds_names = [
            'dream.baxter.synt.dr.test',
        ]
    else:
        raise ValueError(args.config)

    cfg.n_iterations = 10

    if args.model == 'dream-all-models':
        cfg.run_id = None
        cfg.skip_model_predictions = True
        cfg.eval_all_dream_models = True

    elif 'dream-panda' in args.config:
        cfg.allow_gt_bboxes = False
        # Table 1
        if args.models == 'knownq':
            cfg.run_id = PANDA_MODELS['gt_joints']
        elif args.models == 'unknownq':
            cfg.run_id = PANDA_MODELS['predict_joints']
        # Online (Table 2)
        elif args.models == 'knownq-online':
            cfg.run_id = PANDA_MODELS['gt_joints']
            cfg.online_evaluation = True
            cfg.n_plotters = 1
            cfg.n_pred_bsz = 1
            cfg.eval_bsz = 1
        elif args.models == 'unknownq-online':
            cfg.run_id = PANDA_MODELS['predict_joints']
            cfg.online_evaluation = True
            cfg.n_plotters = 1
            cfg.n_pred_bsz = 1
            cfg.eval_bsz = 1

        # Ablations (Table 5)
        elif args.models == 'knownq-link0':
            cfg.run_id = PANDA_ABLATION_REFERENCE_POINT_MODELS['link0']
        elif args.models == 'knownq-link1':
            cfg.run_id = PANDA_ABLATION_REFERENCE_POINT_MODELS['link1']
        elif args.models == 'knownq-link5':
            cfg.run_id = PANDA_ABLATION_REFERENCE_POINT_MODELS['link5']
        elif args.models == 'knownq-link2':
            cfg.run_id = PANDA_ABLATION_REFERENCE_POINT_MODELS['link2']
        elif args.models == 'knownq-link4':
            cfg.run_id = PANDA_ABLATION_REFERENCE_POINT_MODELS['link4']
        elif args.models == 'knownq-link9':
            cfg.run_id = PANDA_ABLATION_REFERENCE_POINT_MODELS['link9']

        # Ablations (Table 6)
        elif args.models == 'unknownq-link1':
            cfg.run_id = PANDA_ABLATION_ANCHOR_MODELS['link1']
        elif args.models == 'unknownq-link2':
            cfg.run_id = PANDA_ABLATION_ANCHOR_MODELS['link2']
        elif args.models == 'unknownq-link5':
            cfg.run_id = PANDA_ABLATION_ANCHOR_MODELS['link5']
        elif args.models == 'unknownq-link0':
            cfg.run_id = PANDA_ABLATION_ANCHOR_MODELS['link0']
        elif args.models == 'unknownq-link4':
            cfg.run_id = PANDA_ABLATION_ANCHOR_MODELS['link4']
        elif args.models == 'unknownq-link9':
            cfg.run_id = PANDA_ABLATION_ANCHOR_MODELS['link9']
        elif args.models == 'unknownq-random_all':
            cfg.run_id = PANDA_ABLATION_ANCHOR_MODELS['random_all']
        elif args.models == 'unknownq-random_top5':
            cfg.run_id = PANDA_ABLATION_ANCHOR_MODELS['random_top5']
        elif args.models == 'unknownq-random_top3':
            cfg.run_id = PANDA_ABLATION_ANCHOR_MODELS['random_top3']

        # Sup. Mat. Iterations
        elif args.models == 'train_K=1':
            cfg.run_id = PANDA_ABLATION_ITERATION_MODELS['n_train_iter=1']
            cfg.eval_bsz = cfg.pred_bsz
            eval_all_iter = True
        elif args.models == 'train_K=2':
            cfg.run_id = PANDA_ABLATION_ITERATION_MODELS['n_train_iter=2']
            cfg.eval_bsz = cfg.pred_bsz
            eval_all_iter = True
        elif args.models == 'train_K=3':
            cfg.run_id = PANDA_MODELS['predict_joints']
            cfg.eval_bsz = cfg.pred_bsz
            eval_all_iter = True
        elif args.models == 'train_K=5':
            cfg.run_id = PANDA_ABLATION_ITERATION_MODELS['n_train_iter=5']
            cfg.eval_bsz = cfg.pred_bsz
            eval_all_iter = True
        else:
            raise ValueError('Unknown models')

    elif 'craves' in args.config:
        if 'youtube' in args.config:
            cfg.n_iterations = 20
        else:
            cfg.n_iterations = 10
        cfg.pred_bsz = 1
        cfg.run_id = OWI_MODELS['predict_joints']
        if args.models == 'unknownq':
            cfg.n_iterations = 10
        elif args.models == 'unknownq-focal=500':
            cfg.focal_length = 500
        elif args.models == 'unknownq-focal=750':
            cfg.focal_length = 750
        elif args.models == 'unknownq-focal=1000':
            cfg.focal_length = 1000
        elif args.models == 'unknownq-focal=1250':
            cfg.focal_length = 1250
        elif args.models == 'unknownq-focal=1500':
            cfg.focal_length = 1500
        elif args.models == 'unknownq-focal=1750':
            cfg.focal_length = 1750
        elif args.models == 'unknownq-focal=2000':
            cfg.focal_length = 2000
        elif args.models == 'unknownq-focal=5000':
            cfg.focal_length = 5000
        else:
            raise ValueError

    elif 'dream-kuka' in args.config:
        if args.models == 'knownq':
            cfg.run_id = KUKA_MODELS['gt_joints']
        elif args.models == 'unknownq':
            cfg.run_id = KUKA_MODELS['predict_joints']
        else:
            raise ValueError

    elif 'dream-baxter' in args.config:
        if args.models == 'knownq':
            cfg.run_id = BAXTER_MODELS['gt_joints']
        elif args.models == 'unknownq':
            cfg.run_id = BAXTER_MODELS['predict_joints']
        else:
            raise ValueError
    else:
        logger.info('No models')
        args.skip_model_predictions = True

    eval_keys = {'gt_detections/unknown_joints/craves', 'known_joints/dream',}
    for k1 in ('gt_detections', 'full_image_detections'):
        for k2 in ('gt_joints', 'unknown_joints', 'unknown_joints_with_gt_init'):
            if eval_all_iter:
                for it in range(1, cfg.n_iterations+1):
                    eval_keys.add(f'{k1}/{k2}/iteration={it}')
            else:
                eval_keys.add(f'{k1}/{k2}/iteration={cfg.n_iterations}')
            eval_keys.add(f'{k1}/{k2}/online')

    cfg.eval_keys = eval_keys
    cfg.eval_bsz = cfg.pred_bsz

    if args.id < 0:
        n_rand = np.random.randint(1e6)
        args.id = n_rand
    save_dir = RESULTS_DIR / f'{args.config}-{args.models}-{args.comment}-{args.id}'
    logger.info(f'Save dir: {save_dir}')

    if args.debug:
        cfg.n_workers = 2
        cfg.n_plotters = 1
        cfg.pred_bsz = 1
        cfg.eval_bsz = cfg.pred_bsz
        cfg.n_frames = 20

    logger.info(f'DS NAMES: {ds_names}')
    np.random.seed(0)
    torch.manual_seed(0)
    for ds_name in ds_names:
        this_cfg = deepcopy(cfg)
        this_cfg.ds_name = ds_name
        this_cfg.save_dir = save_dir / f'dataset={ds_name}'
        logger.info(f'DATASET: {ds_name}')
        run_robot_eval(this_cfg)
        logger.info(f'Done with DATASET: {ds_name}')

    (save_dir / 'config.yaml').write_text(yaml.dump(cfg))


if __name__ == '__main__':
    main()
