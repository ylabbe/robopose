PANDA_MODELS = dict(
    gt_joints='dream-panda-gt_joints--495831',
    predict_joints='dream-panda-predict_joints--173472',
)

KUKA_MODELS = dict(
    gt_joints='dream-kuka-gt_joints--192228',
    predict_joints='dream-kuka-predict_joints--990681',
)

BAXTER_MODELS = dict(
    gt_joints='dream-baxter-gt_joints--510055',
    predict_joints='dream-baxter-predict_joints--519984',
)

OWI_MODELS = dict(
    predict_joints='craves-owi535-predict_joints--295440',
)

PANDA_ABLATION_REFERENCE_POINT_MODELS = dict(
    link0='dream-panda-gt_joints-reference_point=link0--864695',
    link1='dream-panda-gt_joints-reference_point=link1--499756',
    link2='dream-panda-gt_joints-reference_point=link2--905185',
    link4='dream-panda-gt_joints-reference_point=link4--913645',
    link5='dream-panda-gt_joints-reference_point=link5--669469',
    link9='dream-panda-gt_joints-reference_point=hand--588677',
)

PANDA_ABLATION_ANCHOR_MODELS = dict(
    link0='dream-panda-predict_joints-anchor=link0--90648',
    link1='dream-panda-predict_joints-anchor=link1--375503',
    link2='dream-panda-predict_joints-anchor=link2--463951',
    link4='dream-panda-predict_joints-anchor=link4--388856',
    link5='dream-panda-predict_joints-anchor=link5--249745',
    link9='dream-panda-predict_joints-anchor=link9--106543',
    random_all='dream-panda-predict_joints-anchor=random_all--116995',
    random_top3='dream-panda-predict_joints-anchor=random_top_3_largest--65378',
    random_top5=PANDA_MODELS['predict_joints'],
)

PANDA_ABLATION_ITERATION_MODELS = {
    'n_train_iter=1': 'dream-panda-predict_joints-n_train_iter=1--752826',
    'n_train_iter=2': 'dream-panda-predict_joints-n_train_iter=2--949003',
    'n_train_iter=5': 'dream-panda-predict_joints-n_train_iter=5--315150',
}


RESULT_ID = 1804

DREAM_PAPER_RESULT_IDS = [
    f'dream-{robot}-dream-all-models--{RESULT_ID}' for robot in ('panda', 'kuka', 'baxter')
]

DREAM_KNOWN_ANGLES_RESULT_IDS = [
    f'dream-{robot}-knownq--{RESULT_ID}' for robot in ('panda', 'kuka', 'baxter')
]

DREAM_UNKNOWN_ANGLES_RESULT_IDS = [
    f'dream-{robot}-unknownq--{RESULT_ID}' for robot in ('panda', 'kuka', 'baxter')
]

PANDA_KNOWN_ANGLES_ITERATIVE_RESULT_IDS = [
    f'dream-panda-orb-knownq--{RESULT_ID}',
    f'dream-panda-orb-knownq-online--{RESULT_ID}'
]

CRAVES_LAB_RESULT_IDS = [
    f'craves-lab-unknownq--{RESULT_ID}'
]

CRAVES_YOUTUBE_RESULT_IDS = [
    f'craves-youtube-unknownq-focal={focal}--{RESULT_ID}' for focal in (500, 750, 1000, 1250, 1500, 1750, 2000)
]

PANDA_KNOWN_ANGLES_ABLATION_RESULT_IDS = [
    f'dream-panda-orb-knownq-link{link_id}--{RESULT_ID}' for link_id in (0, 1, 2, 4, 5, 9)
]

PANDA_UNKNOWN_ANGLES_ABLATION_RESULT_IDS = [
    f'dream-panda-orb-unknownq-{anchor}--{RESULT_ID}'
    for anchor in ('link5', 'link2', 'link1', 'link0', 'link4', 'link9', 'random_all', 'random_top5', 'random_top3')
]

PANDA_ITERATIONS_ABLATION_RESULT_IDS = [
    f'dream-panda-orb-train_K={train_K}--{RESULT_ID}'
    for train_K in (1, 2, 3, 5)
]
