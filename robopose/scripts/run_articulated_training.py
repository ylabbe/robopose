import argparse
import numpy as np
import os
from colorama import Fore, Style

from robopose.training.train_articulated import train_pose
from robopose.utils.logging import get_logger
logger = get_logger(__name__)


def make_cfg(args):
    cfg = argparse.ArgumentParser('').parse_args([])
    if args.config:
        logger.info(f"{Fore.GREEN}Training with config: {args.config} {Style.RESET_ALL}")

    cfg.resume_run_id = None
    if len(args.resume) > 0:
        cfg.resume_run_id = args.resume
        logger.info(f"{Fore.RED}Resuming {cfg.resume_run_id} {Style.RESET_ALL}")

    N_CPUS = int(os.environ.get('N_CPUS', 10))
    N_WORKERS = min(N_CPUS - 2, 8)
    N_RAND = np.random.randint(1e6)

    run_comment = ''

    # Data
    cfg.urdf_ds_name = ''

    cfg.train_ds_names = []
    cfg.val_ds_names = cfg.train_ds_names
    cfg.val_epoch_interval = 10
    cfg.test_ds_names = []
    cfg.test_epoch_interval = 100
    cfg.n_test_frames = None
    cfg.input_resize = (480, 640)
    cfg.rgb_augmentation = True
    cfg.background_augmentation = True

    # Model
    cfg.backbone_str = 'resnet34'
    cfg.run_id_pretrain = None
    cfg.run_id_pretrain_backbone = None
    cfg.n_rendering_workers = N_WORKERS

    # Optimizer
    cfg.lr = 3e-4
    cfg.weight_decay = 0.
    cfg.n_epochs_warmup = 50
    cfg.lr_epoch_decay = 550
    cfg.clip_grad_norm = 10

    # Training
    cfg.batch_size = 32
    cfg.epoch_size = 115200
    cfg.n_epochs = 700
    cfg.n_dataloader_workers = N_WORKERS
    cfg.save_epoch_interval = None

    # Method
    cfg.loss_disentangled = True
    cfg.n_points_loss = 2600
    cfg.n_iterations = 3
    cfg.add_iteration_epoch_interval = 150
    cfg.center_crop_on = 'centroid'
    cfg.reference_point = 'centroid'

    cfg.input_generator = 'gt+noise'
    cfg.predict_joints = False
    cfg.points_for_pose_loss = 'whole_robot'
    cfg.possible_anchor_links = 'top_5_largest'
    cfg.joints_std_interval_ratio = 0.05
    cfg.loss_q_lambda = 1.

    # Test stuff
    cfg.test_ds_names = []
    cfg.test_n_iterations = 10
    cfg.test_n_frames = None
    cfg.test_epoch_interval = 100
    cfg.test_only_last_epoch = False

    def gt_joints_cfg():
        cfg.predict_joints = False
        cfg.possible_anchor_links = 'base_only'
        cfg.points_for_pose_loss = 'whole_robot'
        cfg.center_crop_on = 'centroid'
        cfg.reference_point = 'centroid'

    def predict_joints_cfg():
        cfg.center_crop_on = 'centroid'
        cfg.reference_point = 'centroid'
        cfg.predict_joints = True
        cfg.possible_anchor_links = 'top_5_largest'
        cfg.points_for_pose_loss = 'anchor_link'
        cfg.joints_std_interval_ratio = 0.05

    if 'dream-panda' in args.config:
        cfg.urdf_ds_name = 'panda'
        cfg.train_ds_names = [('dream.panda.synt.dr.train', 1)]
        cfg.val_ds_names = [('dream.panda.synt.dr.train', 1)]
        # DREAM datasets don't have masks for background augmentation
        cfg.background_augmentation = False

        # GT joints model
        if 'dream-panda-gt_joints' == args.config:
            gt_joints_cfg()
        # Unknown joints model
        elif 'dream-panda-predict_joints' == args.config:
            predict_joints_cfg()

        # Ablations GT joints
        elif 'dream-panda-gt_joints-reference_point=link5' == args.config:
            gt_joints_cfg()
            cfg.reference_point = 'on_link=panda_link5'
        elif 'dream-panda-gt_joints-reference_point=link2' == args.config:
            gt_joints_cfg()
            cfg.reference_point = 'on_link=panda_link2'
        elif 'dream-panda-gt_joints-reference_point=link1' == args.config:
            gt_joints_cfg()
            cfg.reference_point = 'on_link=panda_link1'
        elif 'dream-panda-gt_joints-reference_point=link4' == args.config:
            gt_joints_cfg()
            cfg.reference_point = 'on_link=panda_link4'
        elif 'dream-panda-gt_joints-reference_point=link0' == args.config:
            gt_joints_cfg()
            cfg.reference_point = 'on_link=panda_link0'
        elif 'dream-panda-gt_joints-reference_point=hand' == args.config:
            gt_joints_cfg()
            cfg.reference_point = 'on_link=panda_hand'

        # Ablations, unknown joints
        elif 'dream-panda-predict_joints-anchor=link5' == args.config:
            predict_joints_cfg()
            cfg.possible_anchor_links = ['panda_link5']
        elif 'dream-panda-predict_joints-anchor=link2' == args.config:
            predict_joints_cfg()
            cfg.possible_anchor_links = ['panda_link2']
        elif 'dream-panda-predict_joints-anchor=link1' == args.config:
            predict_joints_cfg()
            cfg.possible_anchor_links = ['panda_link1']
        elif 'dream-panda-predict_joints-anchor=link4' == args.config:
            predict_joints_cfg()
            cfg.possible_anchor_links = ['panda_link4']
        elif 'dream-panda-predict_joints-anchor=link0' == args.config:
            predict_joints_cfg()
            cfg.possible_anchor_links = ['panda_link0']
        elif 'dream-panda-predict_joints-anchor=hand' == args.config:
            predict_joints_cfg()
            cfg.possible_anchor_links = ['panda_hand']
        elif 'dream-panda-predict_joints-anchor=random_all' == args.config:
            predict_joints_cfg()
            cfg.possible_anchor_links = 'all_with_points'
        elif 'dream-panda-predict_joints-anchor=random_top_3_largest' == args.config:
            predict_joints_cfg()
            cfg.possible_anchor_links = 'top_3_largest'

        # Study number of iterations
        elif 'dream-panda-predict_joints-n_train_iter=1' == args.config:
            predict_joints_cfg()
            cfg.n_iterations = 1
        elif 'dream-panda-predict_joints-n_train_iter=2' == args.config:
            predict_joints_cfg()
            cfg.n_iterations = 2
        elif 'dream-panda-predict_joints-n_train_iter=5' == args.config:
            predict_joints_cfg()
            cfg.n_iterations = 5
        else:
            raise ValueError(args.config)

    elif 'craves-owi535' in args.config:
        cfg.urdf_ds_name = 'owi535'
        cfg.train_ds_names = [('craves.synt.train', 100)]
        cfg.val_ds_names = [('craves.synt.train', 100)]

        if 'craves-owi535-predict_joints' == args.config:
            predict_joints_cfg()
        else:
            raise ValueError(args.config)

    elif 'dream-baxter' in args.config:
        cfg.urdf_ds_name = 'baxter'
        cfg.train_ds_names = [('dream.baxter.synt.dr.train', 1)]
        cfg.val_ds_names = [('dream.baxter.synt.dr.train', 1)]

        if 'dream-baxter-gt_joints' == args.config:
            gt_joints_cfg()
        elif 'dream-baxter-predict_joints' == args.config:
            predict_joints_cfg()
        else:
            raise ValueError(args.config)

    elif 'dream-kuka' in args.config:
        cfg.urdf_ds_name = 'iiwa7'
        cfg.train_ds_names = [('dream.kuka.synt.dr.train', 1)]
        cfg.val_ds_names = [('dream.kuka.synt.dr.train', 1)]

        if 'dream-kuka-gt_joints' == args.config:
            gt_joints_cfg()
        elif 'dream-kuka-predict_joints' == args.config:
            predict_joints_cfg()
        else:
            raise ValueError(args.config)


    elif args.resume:
        pass

    else:
        raise ValueError(args.config)

    cfg.run_id = f'{args.config}-{run_comment}-{N_RAND}'

    if args.debug:
        cfg.n_epochs = 6
        cfg.val_epoch_interval = 1
        cfg.batch_size = 4
        cfg.epoch_size = 20 * cfg.batch_size
        cfg.run_id = 'debug-' + cfg.run_id
        cfg.background_augmentation = True
        cfg.n_dataloader_workers = 8
        cfg.n_rendering_workers = 1
        cfg.n_test_frames = 10
        cfg.add_iteration_epoch_interval = 1

    N_GPUS = int(os.environ.get('N_PROCS', 1))
    cfg.epoch_size = cfg.epoch_size // N_GPUS
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', default='', type=str)
    args = parser.parse_args()

    cfg = make_cfg(args)
    train_pose(cfg)
