import numpy as np
import torch
import transforms3d


def loc_rot_pose_errors(TXO_pred, TXO_gt):
    loc_errors = TXO_pred[:, :3, -1] - TXO_gt[:, :3, -1]
    ypr_errors = []
    for R_pred_n, R_gt_n in zip(TXO_pred[:, :3, :3], TXO_gt[:, :3, :3]):
        ypr_pred = np.array(transforms3d.euler.mat2euler(R_pred_n.cpu().numpy()))
        ypr_gt = np.array(transforms3d.euler.mat2euler(R_gt_n.cpu().numpy()))
        ypr_errors.append(np.abs(ypr_pred - ypr_gt))
    ypr_errors = torch.as_tensor(np.stack(ypr_errors), device=TXO_pred.device, dtype=TXO_pred.dtype)
    ypr_errors *= 180 / np.pi
    return loc_errors, ypr_errors
