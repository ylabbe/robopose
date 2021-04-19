import robopose
import os
import yaml
from joblib import Memory
from pathlib import Path
import getpass
import socket
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

hostname = socket.gethostname()
username = getpass.getuser()

PROJECT_ROOT = Path(robopose.__file__).parent.parent
PROJECT_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_DIR / 'data'
LOCAL_DATA_DIR = PROJECT_DIR / 'local_data'
TEST_DATA_DIR = LOCAL_DATA_DIR

EXP_DIR = LOCAL_DATA_DIR / 'experiments'
RESULTS_DIR = LOCAL_DATA_DIR / 'results'
DEBUG_DATA_DIR = LOCAL_DATA_DIR / 'debug_data'
DEPS_DIR = PROJECT_DIR / 'deps'
CACHE_DIR = LOCAL_DATA_DIR / 'joblib_cache'

assert LOCAL_DATA_DIR.exists()
CACHE_DIR.mkdir(exist_ok=True)
TEST_DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DEBUG_DATA_DIR.mkdir(exist_ok=True)

ASSET_DIR = DATA_DIR / 'assets'
MEMORY = Memory(CACHE_DIR, verbose=2)

# ROBOTS
DREAM_DS_DIR = LOCAL_DATA_DIR / 'dream_datasets'

UR_DESCRIPTION = DEPS_DIR / 'ur5-description' / 'ur_description'
UR_DESCRIPTION_NEW = DEPS_DIR / 'ur5-description' / 'package_new_visuals' / 'ur_description'

OWI_DESCRIPTION = DEPS_DIR / 'owi-description' / 'owi535_description'
OWI_KEYPOINTS_PATH = DEPS_DIR / 'owi-description' / 'keypoints.json'

PANDA_DESCRIPTION_PATH = DEPS_DIR / 'panda-description' / 'panda.urdf'
PANDA_KEYPOINTS_PATH = LOCAL_DATA_DIR / 'dream_results' / 'panda_keypoints_infos.json'

CRAVES_YOUTUBE_RESULTS_DIR = LOCAL_DATA_DIR / 'craves_results/youtube-vis/preds'
CRAVES_LAB_RESULTS_DIR = LOCAL_DATA_DIR / 'craves_results/lab/preds'

CONDA_PREFIX = os.environ['CONDA_PREFIX']
if 'CONDA_PREFIX_1' in os.environ:
    CONDA_BASE_DIR = os.environ['CONDA_PREFIX_1']
    CONDA_ENV = os.environ['CONDA_DEFAULT_ENV']
else:
    CONDA_BASE_DIR = os.environ['CONDA_PREFIX']
    CONDA_ENV = 'base'
