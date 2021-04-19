import yaml
from copy import deepcopy
import argparse
import time
import torch
import simplejson as json
from tqdm import tqdm
import functools
from torchnet.meter import AverageValueMeter
from collections import defaultdict
import torch.distributed as dist

from robopose.config import EXP_DIR

from torch.utils.data import DataLoader, ConcatDataset
from robopose.utils.multiepoch_dataloader import MultiEpochDataLoader

from robopose.datasets.datasets_cfg import make_urdf_dataset, make_scene_dataset
from robopose.datasets.articulated_dataset import ArticulatedDataset
from robopose.datasets.samplers import PartialSampler

from robopose.rendering.bullet_batch_renderer import BulletBatchRenderer
from robopose.lib3d.articulated_mesh_database import MeshDataBase

from .pose_articulated_forward_loss import h_pose
from .articulated_models_cfg import create_model, check_update_config


from robopose.scripts.run_robot_eval import run_robot_eval
from robopose.utils.logging import get_logger
from robopose.utils.distributed import get_world_size, get_rank, sync_model, init_distributed_mode, reduce_dict, get_tmp_dir
from torch.backends import cudnn

cudnn.benchmark = True
logger = get_logger(__name__)


def log(config, model,
        log_dict, test_dict, epoch):
    save_dir = config.save_dir
    save_dir.mkdir(exist_ok=True)
    if log_dict is not None:
        log_dict.update(epoch=epoch)
    if not (save_dir / 'config.yaml').exists():
        (save_dir / 'config.yaml').write_text(yaml.dump(config))

    def save_checkpoint(model, postfix=None):
        ckpt_name = 'checkpoint'
        if postfix is not None:
            ckpt_name += postfix
        ckpt_name += '.pth.tar'
        path = save_dir / ckpt_name
        torch.save({'state_dict': model.module.state_dict(),
                    'epoch': epoch}, path)

    save_checkpoint(model)
    save_checkpoint(model, postfix='_epoch=last')
    if config.save_epoch_interval is not None and epoch % config.save_epoch_interval == 0:
        save_checkpoint(model, postfix=f'_epoch={epoch}')

    if log_dict is not None:
        with open(save_dir / 'log.txt', 'a') as f:
            f.write(json.dumps(log_dict, ignore_nan=True) + '\n')

    if test_dict is not None:
        for ds_name, ds_errors in test_dict.items():
            ds_errors['epoch'] = epoch
            with open(save_dir / f'errors_{ds_name}.txt', 'a') as f:
                f.write(json.dumps(test_dict[ds_name], ignore_nan=True) + '\n')

    logger.info(config.run_id)
    logger.info(log_dict)
    logger.info(test_dict)


def run_test(args, epoch):
    errors = dict()

    tmp_dir = get_tmp_dir()
    logger.info("loading.")
    args = yaml.load((tmp_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    logger.info("loaded.")

    cfg = argparse.ArgumentParser('').parse_args([])
    cfg.n_workers = 1
    cfg.n_plotters = 1
    cfg.pred_bsz = 8
    cfg.eval_bsz = 8

    cfg.skip_evaluation = False
    cfg.online_evaluation = False
    cfg.skip_model_predictions = False
    cfg.unknown_evaluate_known = True
    cfg.external_predictions = False

    test_ds_names = args.test_ds_names
    cfg.n_iterations = args.test_n_iterations
    cfg.run_id = args.run_id
    cfg.n_frames = args.test_n_frames
    cfg.focal_length = None

    eval_keys = set()
    for k1 in ('gt_detections', 'full_image_detections'):
        for k2 in ('gt_joints', 'unknown_joints', ):
            eval_keys.add(f'{k1}/{k2}/iteration={cfg.n_iterations}')
    cfg.eval_keys = eval_keys

    errors = dict()
    args.save_dir.mkdir(exist_ok=True)
    for ds_name in test_ds_names:
        this_cfg = deepcopy(cfg)
        if 'dream' in ds_name:
            this_cfg.eval_bsz = cfg.pred_bsz
        else:
            this_cfg.eval_bsz = 1
        this_cfg.ds_name = ds_name
        this_cfg.save_dir = args.save_dir / f'epoch={epoch}/dataset={ds_name}'
        this_ds_results = run_robot_eval(this_cfg)
        if get_rank() == 0:
            errors[ds_name] = this_ds_results['summary']
    return errors


def train_pose(args):
    torch.set_num_threads(1)

    if args.resume_run_id:
        resume_dir = EXP_DIR / args.resume_run_id
        resume_args = yaml.load((resume_dir / 'config.yaml').read_text())
        keep_fields = set(['resume', 'resume_path', 'epoch_size', 'resume_run_id'])
        vars(args).update({k: v for k, v in vars(resume_args).items() if k not in keep_fields})

    args.save_dir = EXP_DIR / args.run_id
    args = check_update_config(args)

    logger.info(f"{'-'*80}")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"{'-'*80}")

    # Initialize distributed
    device = torch.cuda.current_device()
    init_distributed_mode()
    if get_rank() == 0:
        tmp_dir = get_tmp_dir()
        (tmp_dir / 'config.yaml').write_text(yaml.dump(args))

    world_size = get_world_size()
    args.n_gpus = world_size
    args.global_batch_size = world_size * args.batch_size
    logger.info(f'Connection established with {world_size} gpus.')

    # Make train/val datasets
    def make_datasets(dataset_names):
        datasets = []
        for (ds_name, n_repeat) in dataset_names:
            ds = make_scene_dataset(ds_name)
            logger.info(f'Loaded {ds_name} with {len(ds)} images.')
            for _ in range(n_repeat):
                datasets.append(ds)
        return ConcatDataset(datasets)

    scene_ds_train = make_datasets(args.train_ds_names)
    scene_ds_val = make_datasets(args.val_ds_names)

    ds_kwargs = dict(
        resize=args.input_resize,
        rgb_augmentation=args.rgb_augmentation,
        background_augmentation=args.background_augmentation,
    )
    ds_train = ArticulatedDataset(scene_ds_train, **ds_kwargs)
    ds_val = ArticulatedDataset(scene_ds_val, **ds_kwargs)

    train_sampler = PartialSampler(ds_train, epoch_size=args.epoch_size)
    ds_iter_train = DataLoader(ds_train, sampler=train_sampler, batch_size=args.batch_size,
                               num_workers=args.n_dataloader_workers, collate_fn=ds_train.collate_fn,
                               drop_last=False, pin_memory=True)
    ds_iter_train = MultiEpochDataLoader(ds_iter_train)

    val_sampler = PartialSampler(ds_val, epoch_size=int(0.1 * args.epoch_size))
    ds_iter_val = DataLoader(ds_val, sampler=val_sampler, batch_size=args.batch_size,
                             num_workers=args.n_dataloader_workers, collate_fn=ds_val.collate_fn,
                             drop_last=False, pin_memory=True)
    ds_iter_val = MultiEpochDataLoader(ds_iter_val)

    # Make model
    renderer = BulletBatchRenderer(object_set=args.urdf_ds_name, n_workers=args.n_rendering_workers)
    urdf_ds = make_urdf_dataset(args.urdf_ds_name)
    mesh_db = MeshDataBase.from_urdf_ds(urdf_ds).cuda().float()

    model = create_model(cfg=args, renderer=renderer, mesh_db=mesh_db).cuda()

    if args.run_id_pretrain is not None:
        pretrain_path = EXP_DIR / args.run_id_pretrain / 'checkpoint.pth.tar'
        logger.info(f'Using pretrained model from {pretrain_path}.')
        model.load_state_dict(torch.load(pretrain_path)['state_dict'])

    if args.run_id_pretrain_backbone is not None and get_rank() == 0:
        pretrain_path = EXP_DIR / args.run_id_pretrain_backbone / 'checkpoint.pth.tar'
        logger.info(f'Using pretrained backbone from {pretrain_path}.')
        pretrain_state_dict = torch.load(pretrain_path)['state_dict']

        model_state_dict = model.state_dict()
        conv1_key = 'backbone.conv1.weight'
        if model_state_dict[conv1_key].shape[1] != pretrain_state_dict[conv1_key].shape[1]:
            logger.info('Using inflated input layer')
            logger.info(f'Original size: {pretrain_state_dict[conv1_key].shape}')
            logger.info(f'Target size: {model_state_dict[conv1_key].shape}')
            pretrain_n_inputs = pretrain_state_dict[conv1_key].shape[1]
            model_n_inputs = model_state_dict[conv1_key].shape[1]
            conv1_weight = pretrain_state_dict[conv1_key]
            weight_inflated = torch.cat([conv1_weight,
                                         conv1_weight[:, [0]].repeat(1, model_n_inputs - pretrain_n_inputs, 1, 1)], axis=1)
            pretrain_state_dict[conv1_key] = weight_inflated.clone()

        pretrain_state_dict = {k: v for k, v in pretrain_state_dict.items() if ('backbone' in k and k in model_state_dict)}
        logger.info(f"Pretrain keys: {list(pretrain_state_dict.keys())}")
        model.load_state_dict(pretrain_state_dict, strict=False)

    if args.resume_run_id:
        resume_dir = EXP_DIR / args.resume_run_id
        path = resume_dir / 'checkpoint.pth.tar'
        logger.info(f'Loading checkpoing from {path}')
        save = torch.load(path)
        state_dict = save['state_dict']
        model.load_state_dict(state_dict)
        start_epoch = save['epoch'] + 1
    else:
        start_epoch = 0
    end_epoch = args.n_epochs

    # Synchronize models across processes.
    model = sync_model(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warmup
    def get_lr_ratio(batch):
        n_batch_per_epoch = args.epoch_size // args.batch_size
        epoch_id = batch // n_batch_per_epoch

        if args.n_epochs_warmup == 0:
            lr_ratio = 1.0
        else:
            n_batches_warmup = args.n_epochs_warmup * (args.epoch_size // args.batch_size)
            lr_ratio = min(max(batch, 1) / n_batches_warmup, 1.0)

        lr_ratio /= 10 ** (epoch_id // args.lr_epoch_decay)
        return lr_ratio

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_ratio)
    lr_scheduler.last_epoch = start_epoch * args.epoch_size // args.batch_size - 1

    # Just remove the annoying warning
    optimizer._step_count = 1
    lr_scheduler.step()
    optimizer._step_count = 0

    for epoch in range(start_epoch, end_epoch + 1):
        meters_train = defaultdict(lambda: AverageValueMeter())
        meters_val = defaultdict(lambda: AverageValueMeter())
        meters_time = defaultdict(lambda: AverageValueMeter())

        if args.add_iteration_epoch_interval is None:
            n_iterations = args.n_iterations
        else:
            n_iterations = min(epoch // args.add_iteration_epoch_interval + 1, args.n_iterations)
        h = functools.partial(h_pose, model=model, cfg=args, n_iterations=n_iterations, mesh_db=mesh_db)

        def train_epoch():
            model.train()
            iterator = tqdm(ds_iter_train, ncols=80)
            t = time.time()
            for n, sample in enumerate(iterator):
                if n > 0:
                    meters_time['data'].add(time.time() - t)

                optimizer.zero_grad()

                t = time.time()
                loss = h(data=sample, meters=meters_train, train=True)
                meters_time['forward'].add(time.time() - t)
                iterator.set_postfix(loss=loss.item())
                meters_train['loss_total'].add(loss.item())

                t = time.time()
                loss.backward()
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.clip_grad_norm, norm_type=2)
                meters_train['grad_norm'].add(torch.as_tensor(total_grad_norm).item())

                optimizer.step()
                meters_time['backward'].add(time.time() - t)
                meters_time['memory'].add(torch.cuda.max_memory_allocated() / 1024. ** 2)

                t = time.time()

                lr_scheduler.step()

        @torch.no_grad()
        def validation():
            model.eval()
            for sample in tqdm(ds_iter_val, ncols=80):
                loss = h(data=sample, meters=meters_val, train=False)
                meters_val['loss_total'].add(loss.item())

        @torch.no_grad()
        def test():
            model.eval()
            return run_test(args, epoch=epoch)

        train_epoch()

        if epoch % args.val_epoch_interval == 0:
            validation()

        log_dict = dict()
        log_dict.update({
            'grad_norm': meters_train['grad_norm'].mean,
            'grad_norm_std': meters_train['grad_norm'].std,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'time_forward': meters_time['forward'].mean,
            'time_backward': meters_time['backward'].mean,
            'time_data': meters_time['data'].mean,
            'gpu_memory': meters_time['memory'].mean,
            'time': time.time(),
            'n_iterations': (epoch + 1) * len(ds_iter_train),
            'n_datas': (epoch + 1) * args.global_batch_size * len(ds_iter_train),
        })

        for string, meters in zip(('train', 'val'), (meters_train, meters_val)):
            for k in dict(meters).keys():
                log_dict[f'{string}_{k}'] = meters[k].mean

        log_dict = reduce_dict(log_dict)
        if get_rank() == 0:
            log(config=args, model=model, epoch=epoch,
                log_dict=None, test_dict=None)

        dist.barrier()

        test_dict = None
        if args.test_only_last_epoch:
            if epoch == end_epoch:
                test_dict = test()
        else:
            if epoch % args.test_epoch_interval == 0:
                test_dict = test()

        if get_rank() == 0:
            log(config=args, model=model, epoch=epoch,
                log_dict=log_dict, test_dict=test_dict)

        dist.barrier()
