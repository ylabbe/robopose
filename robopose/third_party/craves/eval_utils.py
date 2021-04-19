import torch
from .heatmap_utils import heatmap_from_keypoints


def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask

    return preds


def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).sum().numpy() / dists.ne(-1).sum().numpy()
    else:
        return -1


def accuracy(output, target, idxs, thr=0.2):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    preds   = get_preds(output)
    gts     = get_preds(target)
    norm    = torch.ones(preds.size(0))*output.size(3)/4.0
    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i+1] = dist_acc(dists[idxs[i]-1], thr=thr)
        if acc[i+1] >= 0:
            avg_acc = avg_acc + acc[i+1]
            cnt += 1

    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc


def accuracy_from_keypoints(kp_pred, kp_gt, bboxes_gt, thr=0.2):
    assert len(kp_pred) == len(kp_gt) == len(bboxes_gt)
    range_n = range(len(kp_pred))
    heatmaps_pred = torch.stack([heatmap_from_keypoints(bboxes_gt[n], kp_pred[n]) for n in range_n])
    heatmaps_gt = torch.stack([heatmap_from_keypoints(bboxes_gt[n], kp_gt[n]) for n in range_n])

    idxs = range(0, 17)
    acc = accuracy(heatmaps_pred, heatmaps_gt, idxs=idxs, thr=thr)
    return acc
