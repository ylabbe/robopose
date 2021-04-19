import numpy as np
import pandas as pd

from robopose.config import LOCAL_DATA_DIR, ASSET_DIR
from robopose.config import OWI_DESCRIPTION, PANDA_DESCRIPTION_PATH
from robopose.config import DREAM_DS_DIR, PROJECT_DIR
from robopose.utils.logging import get_logger

from .urdf_dataset import OneUrdfDataset
from .craves import CRAVESDataset
from .dream import DreamDataset


logger = get_logger(__name__)

def _make_craves_dataset(split):
    ds_root = LOCAL_DATA_DIR / 'craves_datasets'
    ds = CRAVESDataset(ds_root, split=split)
    return ds

def make_scene_dataset(ds_name, n_frames=None):
    # CRAVES
    if ds_name == 'craves.synt.train':
        split = 'synt_train'
        ds = _make_craves_dataset(split)
    elif ds_name == 'craves.synt.val':
        split = 'synt_val'
        ds = _make_craves_dataset(split)
    elif ds_name == 'craves.lab.real.test':
        split = 'lab_test_real'
        ds = _make_craves_dataset(split)
    elif ds_name == 'craves.youtube':
        split = 'youtube'
        ds = _make_craves_dataset(split)

    # DREAM
    # Panda
    elif ds_name == 'dream.panda.synt.dr.train':
        ds = DreamDataset(DREAM_DS_DIR / 'synthetic/panda_synth_train_dr')
    elif ds_name == 'dream.panda.synt.dr.train.1000':
        ds = DreamDataset(DREAM_DS_DIR / 'synthetic/panda_synth_train_dr')
        ds.frame_index = ds.frame_index.iloc[:1000].reset_index(drop=True)
    elif ds_name == 'dream.panda.synt.photo.test':
        ds = DreamDataset(DREAM_DS_DIR / 'synthetic/panda_synth_test_photo')
    elif ds_name == 'dream.panda.synt.dr.test':
        ds = DreamDataset(DREAM_DS_DIR / 'synthetic/panda_synth_test_dr')
    elif ds_name == 'dream.panda.real.orb':
        ds = DreamDataset(DREAM_DS_DIR / 'real/panda-orb')
    elif ds_name == 'dream.panda.real.kinect360':
        ds = DreamDataset(DREAM_DS_DIR / 'real/panda-3cam_kinect360')
    elif ds_name == 'dream.panda.real.realsense':
        ds = DreamDataset(DREAM_DS_DIR / 'real/panda-3cam_realsense')
    elif ds_name == 'dream.panda.real.azure':
        ds = DreamDataset(DREAM_DS_DIR / 'real/panda-3cam_azure')

    # Baxter
    elif ds_name == 'dream.baxter.synt.dr.train':
        ds = DreamDataset(DREAM_DS_DIR / 'synthetic/baxter_synth_train_dr')
    elif ds_name == 'dream.baxter.synt.dr.test':
        ds = DreamDataset(DREAM_DS_DIR / 'synthetic/baxter_synth_test_dr')

    # Kuka
    elif ds_name == 'dream.kuka.synt.dr.train':
        ds = DreamDataset(DREAM_DS_DIR / 'synthetic/kuka_synth_train_dr')
    elif ds_name == 'dream.kuka.synt.dr.test':
        ds = DreamDataset(DREAM_DS_DIR / 'synthetic/kuka_synth_test_dr')
    elif ds_name == 'dream.kuka.synt.photo.test':
        ds = DreamDataset(DREAM_DS_DIR / 'synthetic/kuka_synth_test_photo')

    else:
        raise ValueError(ds_name)

    if n_frames is not None:
        ds.frame_index = ds.frame_index.iloc[:n_frames].reset_index(drop=True)
    ds.name = ds_name
    return ds


def make_urdf_dataset(ds_name):
    if isinstance(ds_name, list):
        ds_index = []
        for ds_name_n in ds_name:
            dataset = make_urdf_dataset(ds_name_n)
            ds_index.append(dataset.index)
        dataset.index = pd.concat(ds_index, axis=0)
        return dataset
    # OWI
    if ds_name == 'owi535':
        ds = OneUrdfDataset(OWI_DESCRIPTION / 'owi535.urdf', label='owi535', scale=0.001)
    elif ds_name == 'panda':
        ds = OneUrdfDataset(PANDA_DESCRIPTION_PATH.parent / 'patched_urdf/panda.urdf', label='panda')
    # BAXTER
    elif ds_name == 'baxter':
        ds = OneUrdfDataset(PROJECT_DIR / 'deps/baxter-description/baxter_description/patched_urdf/baxter.urdf', label='baxter')
    # KUKA
    elif ds_name == 'iiwa7':
        ds = OneUrdfDataset(PROJECT_DIR / 'deps/kuka-description/iiwa_description/urdf/iiwa7.urdf', label='iiwa7')
    else:
        raise ValueError(ds_name)
    return ds
