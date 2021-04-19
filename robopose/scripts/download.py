import argparse
import zipfile
import wget
import logging
import subprocess
from pathlib import Path
from robopose.config import PROJECT_DIR, LOCAL_DATA_DIR
from robopose.utils.logging import get_logger
from robopose.paper_models_cfg import *

logger = get_logger(__name__)

RCLONE_CFG_PATH = (PROJECT_DIR / 'rclone.conf')
RCLONE_ROOT = 'robopose:'
DOWNLOAD_DIR = LOCAL_DATA_DIR / 'downloads'
DOWNLOAD_DIR.mkdir(exist_ok=True)


def download_robot_description(robot_name):
    gdrive_download(f'deps/{robot_name}-description', PROJECT_DIR / 'deps')

def download_craves_dataset(ds_name):
    zip_name = f'{ds_name}.zip'
    gdrive_download(f'zip_files/craves/{zip_name}', DOWNLOAD_DIR)
    logger.info(f'Extracting dataset {zip_name}...')
    zipfile.ZipFile(DOWNLOAD_DIR / zip_name).extractall(LOCAL_DATA_DIR / 'craves_datasets')

def download_dream_dataset(real_or_synt, ds_name):
    assert real_or_synt == 'synthetic' or real_or_synt == 'real'
    zip_name = f'{ds_name}.zip'
    gdrive_download(f'zip_files/dream/{real_or_synt}/{zip_name}', DOWNLOAD_DIR)
    logger.info(f'Extracting dataset {zip_name}...')
    zipfile.ZipFile(DOWNLOAD_DIR / zip_name).extractall(LOCAL_DATA_DIR / 'dream_datasets' / real_or_synt)

def download_model(run_id):
    gdrive_download(f'models/{run_id}', LOCAL_DATA_DIR / 'experiments')

def download_results(result_id):
    gdrive_download(f'results/{result_id}', LOCAL_DATA_DIR / 'results')

def download_dream_results():
    gdrive_download(f'dream_results', LOCAL_DATA_DIR / 'dream_results')

def download_craves_results():
    gdrive_download(f'craves_results', LOCAL_DATA_DIR / 'craves_results')


def main():
    parser = argparse.ArgumentParser('RoboPose download utility')
    parser.add_argument('--robot', default='', type=str)
    parser.add_argument('--datasets', default='', type=str)
    parser.add_argument('--model', default='', type=str)
    parser.add_argument('--dream_paper_results', action='store_true')
    parser.add_argument('--craves_paper_results', action='store_true')
    parser.add_argument('--models', default='', type=str)
    parser.add_argument('--results', default='', type=str)

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.robot:
        download_robot_description(args.robot)

    if args.datasets:
        if args.datasets == 'craves.test':
            download_craves_dataset('youtube_20181105')
            download_craves_dataset('test_20181024')
        elif args.datasets == 'craves.train':
            download_craves_dataset('20181107')

        elif args.datasets == 'dream.train':
            for (synt_or_real, ds_name) in [
                    ('synthetic', 'panda_synth_train_dr'),
                    ('synthetic', 'baxter_synth_train_dr'),
                    ('synthetic', 'kuka_synth_train_dr')
            ]:
                download_dream_dataset(synt_or_real, ds_name)

        elif args.datasets == 'dream.test':
            for (synt_or_real, ds_name) in [
                    ('synthetic', 'panda_synth_test_photo'),
                    ('synthetic', 'panda_synth_test_dr'),
                    ('synthetic', 'kuka_synth_test_photo'),
                    ('synthetic', 'kuka_synth_test_dr'),
                    ('synthetic', 'baxter_synth_test_dr'),

                    ('real', 'panda-orb'),
                    ('real', 'panda-3cam_realsense'),
                    ('real', 'panda-3cam_kinect360'),
                    ('real', 'panda-3cam_azure'),
            ]:
                download_dream_dataset(synt_or_real, ds_name)
        else:
            raise ValueError(args.datasets)

    if args.model:
        if args.model == 'panda-known_angles':
            download_model(PANDA_MODELS['gt_joints'])
        elif args.model == 'panda-predict_angles':
            download_model(PANDA_MODELS['predict_joints'])
        elif args.model == 'kuka-known_angles':
            download_model(KUKA_MODELS['gt_joints'])
        elif args.model == 'kuka-predict_angles':
            download_model(KUKA_MODELS['predict_joints'])
        elif args.model == 'baxter-known_angles':
            download_model(BAXTER_MODELS['gt_joints'])
        elif args.model == 'baxter-predict_angles':
            download_model(BAXTER_MODELS['predict_joints'])
        elif args.model == 'owi-predict_angles':
            download_model(OWI_MODELS['predict_joints'])
        else:
            raise ValueError(args.model)

    if args.dream_paper_results:
        download_dream_results()

    if args.craves_paper_results:
        download_craves_results()

    if args.results:
        if args.results == 'dream-paper-all-models':
            result_ids = DREAM_PAPER_RESULT_IDS
        elif args.results == 'dream-known-angles':
            result_ids = DREAM_KNOWN_ANGLES_RESULT_IDS
        elif args.results == 'dream-unknown-angles':
            result_ids = DREAM_UNKNOWN_ANGLES_RESULT_IDS
        elif args.results == 'dream-unknown-angles':
            result_ids = DREAM_UNKNOWN_ANGLES_RESULT_IDS
        elif args.results == 'panda-orb-known-angles-iterative':
            result_ids = PANDA_KNOWN_ANGLES_ITERATIVE_RESULT_IDS
        elif args.results == 'craves-lab':
            result_ids = CRAVES_LAB_RESULT_IDS
        elif args.results == 'craves-youtube':
            result_ids = CRAVES_YOUTUBE_RESULT_IDS
        elif args.results == 'panda-reference-point-ablation':
            result_ids = PANDA_KNOWN_ANGLES_ABLATION_RESULT_IDS
        elif args.results == 'panda-anchor-ablation':
            result_ids = PANDA_UNKNOWN_ANGLES_ABLATION_RESULT_IDS
        elif args.results == 'panda-train_iterations-ablation':
            result_ids = PANDA_ITERATIONS_ABLATION_RESULT_IDS
        else:
            raise ValueError
        for result_id in result_ids:
            download_results(result_id)

    if args.models:
        if args.models == 'ablation_reference_point':
            model_ids = list(PANDA_ABLATION_REFERENCE_POINT_MODELS.values())
        elif args.models == 'ablation_anchor':
            model_ids = list(PANDA_ABLATION_ANCHOR_MODELS.values())
        elif args.models == 'ablation_train_iterations':
            model_ids = list(PANDA_ABLATION_ITERATION_MODELS.values())
        else:
            raise ValueError

        for model_id in model_ids:
            download_model(model_id)

def run_rclone(cmd, args, flags):
    rclone_cmd = ['rclone', cmd] + args + flags + ['--config', str(RCLONE_CFG_PATH)]
    logger.debug(' '.join(rclone_cmd))
    subprocess.run(rclone_cmd)


def gdrive_download(gdrive_path, local_path):
    gdrive_path = Path(gdrive_path)
    if gdrive_path.name != local_path.name:
        local_path = local_path / gdrive_path.name
    rclone_path = RCLONE_ROOT+str(gdrive_path)
    local_path = str(local_path)
    logger.info(f'Copying {rclone_path} to {local_path}')
    run_rclone('copyto', [rclone_path, local_path], flags=['-P'])


def download_bop_original(ds_name, download_pbr):
    filename = f'{ds_name}_base.zip'
    wget_download_and_extract(BOP_SRC + filename, BOP_DS_DIR)

    suffixes = ['models'] + BOP_DATASETS[ds_name]['splits']
    if download_pbr:
        suffixes += ['train_pbr']
    for suffix in suffixes:
        wget_download_and_extract(BOP_SRC + f'{ds_name}_{suffix}.zip', BOP_DS_DIR / ds_name)


def download_bop_gdrive(ds_name):
    gdrive_download(f'bop_datasets/{ds_name}', BOP_DS_DIR / ds_name)


def wget_download_and_extract(url, out):
    tmp_path = DOWNLOAD_DIR / url.split('/')[-1]
    if tmp_path.exists():
        logger.info(f'{url} already downloaded: {tmp_path}...')
    else:
        logger.info(f'Download {url} at {tmp_path}...')
        wget.download(url, out=tmp_path.as_posix())
    logger.info(f'Extracting {tmp_path} at {out}.')
    zipfile.ZipFile(tmp_path).extractall(out)


if __name__ == '__main__':
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    main()
