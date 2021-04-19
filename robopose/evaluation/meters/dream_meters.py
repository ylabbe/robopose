import numpy as np
import xarray as xr
import torch
from .base import Meter
from .utils import one_to_one_matching
from robopose.lib3d.transform_ops import transform_pts


class DreamErrorMeter(Meter):
    def __init__(self):
        self.reset()

    def is_data_valid(self, data):
        valid = False
        if not valid and hasattr(data, 'TCO_keypoints_3d'):
            valid = True
        if not valid and hasattr(data, 'pnp_poses'):
            valid = True
        return valid

    def add(self, pred_data, gt_data):
        pred_data = pred_data.float()
        gt_data = gt_data.float()

        matches = one_to_one_matching(pred_data.infos,
                                      gt_data.infos,
                                      keys=('scene_id', 'view_id'),
                                      allow_pred_missing=False)

        pred_data = pred_data[matches.pred_id]
        gt_data = gt_data[matches.gt_id]

        if not hasattr(pred_data, 'TCO_keypoints_3d'):
            pnp_poses = pred_data.pnp_poses
            kp3d_pred = transform_pts(pnp_poses, gt_data.TCO_keypoints_3d)
        else:
            kp3d_pred = pred_data.TCO_keypoints_3d

        kp3d_gt = gt_data.TCO_keypoints_3d
        bsz = kp3d_pred.shape[0]
        n_kp = kp3d_gt.shape[1]
        assert kp3d_pred.shape == (bsz, n_kp, 3)
        assert kp3d_pred.shape == (bsz, n_kp, 3)

        add = torch.norm(kp3d_pred - kp3d_gt, dim=-1, p=2).mean(dim=-1).cpu().numpy()
        df = xr.Dataset(matches).rename(dict(dim_0='match_id'))
        df['add'] = 'match_id', add
        self.datas['df'].append(df)

    def summary(self):
        df = xr.concat(self.datas['df'], dim='match_id')

        auc_threshold = 0.1
        delta_threshold = 0.00001
        add_threshold_values = np.arange(0.0, auc_threshold, delta_threshold)
        counts = []
        for value in add_threshold_values:
            under_threshold = (
                (df['add'].values <= value).mean()
            )
            counts.append(under_threshold)
        auc = np.trapz(counts, dx=delta_threshold) / auc_threshold

        summary = {
            'n_objects': len(df['match_id']),
            'ADD/mean': df['add'].values.mean().item(),
            'ADD/AUC': auc.item(),
        }
        for th_mm in (10, 20, 40, 60):
            summary[f'ADD<{th_mm}mm'] = (df['add'].values <= th_mm * 1e-3).mean() * 100
        return summary, df
