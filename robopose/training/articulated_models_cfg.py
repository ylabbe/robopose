# Backbones
from robopose.models.wide_resnet import WideResNet34

# Pose models
from robopose.models.articulated import ArticulatedObjectRefiner

from robopose.utils.logging import get_logger
logger = get_logger(__name__)


def check_update_config(config):
    # Compatibility with models trained using the pre-release code
    config.is_old_format = False
    if not hasattr(config, 'urdf_ds_name'):
        config.urdf_ds_name = 'owi535'
        config.backbone_str = 'resnet34'
        config.init_method = 'v1'
        config.is_old_format = True
    if not hasattr(config, 'possible_roots'):
        config.possible_roots = 'top_5_largest'
    if not hasattr(config, 'coordinate_system'):
        config.coordinate_system = 'base_centered'
    if not hasattr(config, 'multiroot'):
        config.multiroot = False
    if not hasattr(config, 'center_crop_on'):
        config.center_crop_on = 'com'
    if not hasattr(config, 'predict_all_links_poses'):
        config.predict_all_links_poses = False

    # Compatib names for new models.
    if not hasattr(config, 'possible_anchor_links'):
        config.possible_anchor_links = config.possible_roots
    if hasattr(config, 'center_crop_on') and config.center_crop_on == 'com':
        config.center_crop_on = 'centroid'
    if hasattr(config, 'coordinate_system'):
        if config.coordinate_system == 'link':
            assert len(config.possible_roots) == 1
            assert not config.predict_joints
            config.possible_anchor_links = 'base_only'
            config.reference_point = f'on_link={config.possible_roots[0]}'
        elif config.coordinate_system == 'camera_centered':
            config.reference_point = 'centroid'
        else:
            pass
    if not hasattr(config, 'center_crop_on'):
        config.possible_anchor_links = config.possible_roots
    return config


def create_model(cfg, renderer, mesh_db):
    n_inputs = 6
    backbone_str = cfg.backbone_str
    anchor_mask_input = cfg.predict_joints
    if anchor_mask_input:
        n_inputs += 1
    if 'resnet34' in backbone_str:
        backbone = WideResNet34(n_inputs=n_inputs)
    else:
        raise ValueError('Unknown backbone', backbone_str)

    logger.info(f'Backbone: {backbone_str}')
    backbone.n_inputs = n_inputs
    render_size = (240, 320)
    model = ArticulatedObjectRefiner(
        backbone=backbone, renderer=renderer, mesh_db=mesh_db,
        render_size=render_size, input_anchor_mask=anchor_mask_input,
        predict_joints=cfg.predict_joints, possible_anchors=cfg.possible_anchor_links,
        center_crop_on=cfg.center_crop_on, reference_point=cfg.reference_point)
    return model
