from tqdm import tqdm

from torch.utils.data import DataLoader

from robopose.utils.distributed import get_world_size, get_rank, get_tmp_dir

import robopose.utils.tensor_collection as tc
from robopose.evaluation.data_utils import parse_obs_data
from robopose.datasets.samplers import DistributedSceneSampler


class ArticulatedObjectEvaluation:
    def __init__(self, scene_ds, meters, batch_size=64,
                 cache_data=True, n_workers=4, sampler=None):

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.tmp_dir = get_tmp_dir()

        self.scene_ds = scene_ds
        if sampler is None:
            sampler = DistributedSceneSampler(scene_ds,
                                              num_replicas=self.world_size,
                                              rank=self.rank)
        dataloader = DataLoader(scene_ds, batch_size=batch_size,
                                num_workers=n_workers,
                                sampler=sampler, collate_fn=self.collate_fn)

        if cache_data:
            self.dataloader = list(tqdm(dataloader))
        else:
            self.dataloader = dataloader

        self.meters = meters

    def collate_fn(self, batch):
        obj_data = []
        for data_n in batch:
            _, _, obs = data_n
            obj_data_ = parse_obs_data(obs, parse_joints=True)
            obj_data.append(obj_data_)
        obj_data = tc.concatenate(obj_data)
        return obj_data

    def evaluate(self, obj_predictions):
        for meter in self.meters.values():
            meter.reset()
        device = obj_predictions.device
        for obj_data_gt in tqdm(self.dataloader):
            for k, meter in self.meters.items():
                if meter.is_data_valid(obj_predictions) and meter.is_data_valid(obj_data_gt):
                    meter.add(obj_predictions, obj_data_gt.to(device))
        return self.summary()

    def summary(self):
        summary, dfs = dict(), dict()
        for meter_k, meter in self.meters.items():
            if len(meter.datas) > 0:
                meter.gather_distributed(tmp_dir=self.tmp_dir)
                summary_, df_ = meter.summary()
                dfs[meter_k] = df_
                for k, v in summary_.items():
                    summary[meter_k + '/' + k] = v
        return summary, dfs
