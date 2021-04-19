import json
import pandas as pd
import torch
from pathlib import Path

from robopose.datasets.craves import parse_name
import robopose.utils.tensor_collection as tc
from robopose.config import CRAVES_YOUTUBE_RESULTS_DIR, CRAVES_LAB_RESULTS_DIR


def load_craves_results(ds_name):
    if 'youtube' in ds_name:
        results_dir = CRAVES_YOUTUBE_RESULTS_DIR
    else:
        results_dir = CRAVES_LAB_RESULTS_DIR

    results_json = Path(results_dir).glob('*.json')
    infos = []
    keypoints = []
    for result_json in results_json:
        result = json.loads(result_json.read_text())
        keypoints.append(torch.tensor(result['d2_key']))
        scene_id, view_id = parse_name(result_json.with_suffix('').name)
        infos.append(dict(scene_id=scene_id, view_id=view_id))
    infos = pd.DataFrame(infos)
    keypoints = torch.stack(keypoints)
    data = tc.PandasTensorCollection(infos, keypoints_2d=keypoints)
    return data
