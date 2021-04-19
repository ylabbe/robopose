import torch
import xarray as xr
import numpy as np
from .base import Meter

from robopose.utils.distributed import get_rank, get_world_size
from robopose.third_party.craves.eval_utils import accuracy_from_keypoints
from .errors import loc_rot_pose_errors
from .utils import get_candidate_matches, match_poses, one_to_one_matching


class CravesKeypointsMeter(Meter):
    def __init__(self):
        self.reset()

    def is_data_valid(self, data):
        return hasattr(data, 'keypoints_2d')

    def add(self, pred_data, gt_data):
        pred_data = pred_data.float()
        gt_data = gt_data.float()

        matches = one_to_one_matching(pred_data.infos,
                                      gt_data.infos,
                                      keys=('scene_id', 'view_id'),
                                      allow_pred_missing=False)

        accuracies = accuracy_from_keypoints(pred_data.keypoints_2d[matches.pred_id],
                                             gt_data.keypoints_2d[matches.gt_id],
                                             gt_data.bboxes[matches.gt_id],
                                             thr=0.2).cpu().numpy()
        batch_df = xr.Dataset()
        batch_df['keypoint_accuracy'] = ('batch_id', 'keypoints'), accuracies[None, 1:]
        batch_df['keypoints_mean_accuracy'] = ('batch_id', ), accuracies[None, 0]
        self.datas['df'].append(batch_df)

    def summary(self):
        df = xr.concat(self.datas['df'], dim='batch_id')
        summary = {
            'PCK@0.2': df['keypoints_mean_accuracy'].mean('batch_id').values.tolist(),
        }
        return summary, df


class CravesErrorMeter(Meter):
    def __init__(self):
        self.reset()

    def is_data_valid(self, data):
        return hasattr(data, 'poses')

    def add(self, pred_data, gt_data):
        # ArticulatedObjectData
        pred_data = pred_data.float()
        gt_data = gt_data.float()

        TXO_gt = gt_data.poses
        q_gt = gt_data.joints
        gt_infos = gt_data.infos
        gt_infos['valid'] = True

        TXO_pred = pred_data.poses
        q_pred = pred_data.joints
        pred_infos = pred_data.infos

        cand_infos = get_candidate_matches(pred_infos, gt_infos)
        cand_TXO_gt = TXO_gt[cand_infos['gt_id']]
        cand_TXO_pred = TXO_pred[cand_infos['pred_id']]

        loc_errors_xyz, rot_errors = loc_rot_pose_errors(cand_TXO_pred, cand_TXO_gt)
        loc_errors_norm = torch.norm(loc_errors_xyz, dim=-1, p=2)

        error = loc_errors_norm.cpu().numpy().astype(np.float)
        error[np.isnan(error)] = 1000
        cand_infos['error'] = error
        matches = match_poses(cand_infos)

        n_gt = len(gt_infos)

        def empty_array(shape, default='nan', dtype=np.float):
            return np.empty(shape, dtype=dtype) * float(default)

        gt_infos['rank'] = get_rank()
        gt_infos['world_size'] = get_world_size()
        df = xr.Dataset(gt_infos).rename(dict(dim_0='gt_object_id'))

        scores = empty_array(n_gt)
        scores[matches.gt_id] = pred_infos.loc[matches.pred_id, 'score']
        df['pred_score'] = 'gt_object_id', scores

        rot_errors_ = empty_array((n_gt, 3))
        matches_rot_errors = rot_errors[matches.cand_id].cpu().numpy()
        matches_rot_errors = np.abs((matches_rot_errors + 180.0) % 360.0 - 180.0)
        rot_errors_[matches.gt_id] = matches_rot_errors
        df['rot_error'] = ('gt_object_id', 'ypr'), rot_errors_

        loc_errors_xyz_ = empty_array((n_gt, 3))
        loc_errors_xyz_[matches.gt_id] = loc_errors_xyz[matches.cand_id].cpu().numpy()
        df['loc_error_xyz'] = ('gt_object_id', 'xyz'), loc_errors_xyz_

        loc_errors_norm_ = empty_array((n_gt))
        loc_errors_norm_[matches.gt_id] = loc_errors_norm[matches.cand_id].cpu().numpy()
        df['loc_error_norm'] = ('gt_object_id', ), loc_errors_norm_

        q_errors = empty_array((n_gt, q_gt.shape[1]))
        matches_q_errors = (q_gt[matches.gt_id] - q_pred[matches.pred_id]).cpu().numpy() * 180 / np.pi
        matches_q_errors = np.abs((matches_q_errors + 180.0) % 360.0 - 180.0)
        q_errors[matches.gt_id] = matches_q_errors

        df['joint_error'] = ('gt_object_id', 'dofs'), q_errors
        self.datas['df'].append(df)

    def summary(self):
        df = xr.concat(self.datas['df'], dim='gt_object_id')
        n_images_top50 = int(len(df['gt_object_id']) * 0.5)
        df_top50 = df.isel(gt_object_id=np.argsort(df['joint_error'].max(axis=1).values)[:n_images_top50])
        summary = dict(
            loc_error_norm=df['loc_error_norm'].mean('gt_object_id').values.tolist(),
            loc_error_norm_top50=df_top50['loc_error_norm'].mean('gt_object_id').values.tolist(),

            loc_error_xyz_abs_mean=np.mean(np.abs(df['loc_error_xyz'].values)).item(),
            loc_error_xyz_abs_mean_top50=np.mean(np.abs(df_top50['loc_error_xyz'].values)).item(),

            rot_error_mean=df['rot_error'].mean().item(),
            rot_error_mean_top50=df_top50['rot_error'].mean().item(),

            joint_error_mean=df['joint_error'].mean(('gt_object_id', 'dofs')).values.tolist(),
            joint_error_mean_top50=df_top50['joint_error'].mean(('gt_object_id', 'dofs')).values.tolist(),
        )
        return summary, df
