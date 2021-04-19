import numpy as np
import pandas as pd
from collections import OrderedDict


def one_to_one_matching(pred_infos, gt_infos,
                        keys=('scene_id', 'view_id'),
                        allow_pred_missing=False):
    keys = list(keys)
    pred_infos['pred_id'] = np.arange(len(pred_infos))
    gt_infos['gt_id'] = np.arange(len(gt_infos))
    matches = pred_infos.merge(gt_infos, on=keys)

    matches_gb = matches.groupby(keys).groups
    assert all([len(v) == 1 for v in matches_gb.values()])
    if not allow_pred_missing:
        assert len(matches) == len(gt_infos)
    return matches


def get_candidate_matches(pred_infos, gt_infos,
                          group_keys=['scene_id', 'view_id', 'label'],
                          only_valids=True):
    pred_infos['pred_id'] = np.arange(len(pred_infos))
    gt_infos['gt_id'] = np.arange(len(gt_infos))
    group_keys = list(group_keys)
    cand_infos = pred_infos.merge(gt_infos, on=group_keys)
    if only_valids:
        cand_infos = cand_infos[cand_infos['valid']].reset_index(drop=True)
    cand_infos['cand_id'] = np.arange(len(cand_infos))
    return cand_infos


def match_poses(cand_infos, group_keys=['scene_id', 'view_id', 'label']):
    assert 'error' in cand_infos

    matches = []

    def match_label_preds(group):
        gt_ids_matched = set()
        group = group.reset_index(drop=True)
        gb_pred = group.groupby('pred_id', sort=False)
        ids_sorted = gb_pred.first().sort_values('score', ascending=False)
        gb_pred_groups = gb_pred.groups
        for idx, _ in ids_sorted.iterrows():
            pred_group = group.iloc[gb_pred_groups[idx]]
            best_error = np.inf
            best_match = None
            for _, tentative_match in pred_group.iterrows():
                if tentative_match['error'] < best_error and \
                   tentative_match['gt_id'] not in gt_ids_matched:
                    best_match = tentative_match
                    best_error = tentative_match['error']

            if best_match is not None:
                gt_ids_matched.add(best_match['gt_id'])
                matches.append(best_match)

    if len(cand_infos) > 0:
        cand_infos.groupby(group_keys).apply(match_label_preds)
        matches = pd.DataFrame(matches).reset_index(drop=True)
    else:
        matches = cand_infos
    return matches
