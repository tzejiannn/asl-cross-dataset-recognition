"""
The 15 conditions of the ablation study. Each one changes one thing about how poses are prepared
before they reach the model, compared with the baseline.

A condition is a list of pose transforms, applied in order (see pose_transforms.py). On top of that:
  - every run first crops each clip to its hand activity (TrimToHandSpan, --trim_hand_span);
  - at training time, shear and rotation augmentation is added at the end (except the two
    "noaug" conditions, see below); with 3D input it acts on x and y and leaves z unchanged;
  - the hand-only conditions use a different skeleton graph (GRAPHS below);
  - depth_3d and depth_3d_noaug use x, y and z instead of x and y.

Groups:
  normalisation:      norm_off, norm_frame_level, wrist_norm_27pt
  low confidence:     conf_mask, conf_mask_no_norm
  landmark subset:    hands_only, hands_only_shoulder_norm, hands_full_42,
                      hands_full_42_shoulder_norm
  temporal:           temporal_fixed64, hands_only_temporal64
  depth:              depth_3d (3D with augmentation, compared with the baseline), and
                      depth_3d_noaug with depth_2d_noaug as its control, so 2D/3D is crossed with
                      augmentation on/off
"""
from pose_config import (
    GRAPH_27,
    GRAPH_HANDS_20,
    GRAPH_HANDS_42,
    POINTS_27,
    POINTS_HANDS_20,
    POINTS_HANDS_42,
    SHOULDERS_27,
    SHOULDERS_75,
    WRISTS_27,
    WRISTS_HANDS_20,
    WRISTS_HANDS_42,
)
from pose_transforms import (
    CenterAndScaleNormalize as Normalize,
    Compose,
    ConfidenceMask,
    PoseSelect as Select,
    RotationTransform,
    ShearTransform,
    TemporalResample,
    TrimToHandSpan,
)

CONDITIONS = {
    "baseline":          [Select(POINTS_27), Normalize(SHOULDERS_27)],

    # normalisation
    "norm_off":          [Select(POINTS_27)],
    "norm_frame_level":  [Select(POINTS_27), Normalize(SHOULDERS_27, frame_level=True)],
    "wrist_norm_27pt":   [Select(POINTS_27), Normalize(WRISTS_27)],

    # low-confidence points set to zero
    "conf_mask":         [ConfidenceMask(0.5), Select(POINTS_27), Normalize(SHOULDERS_27)],
    "conf_mask_no_norm": [ConfidenceMask(0.5), Select(POINTS_27)],

    # landmark subset ("shoulder_norm": normalise on the shoulders first, then keep only the hands)
    "hands_only":                  [Select(POINTS_HANDS_20), Normalize(WRISTS_HANDS_20)],
    "hands_only_shoulder_norm":    [Normalize(SHOULDERS_75), Select(POINTS_HANDS_20)],
    "hands_full_42":               [Select(POINTS_HANDS_42), Normalize(WRISTS_HANDS_42)],
    "hands_full_42_shoulder_norm": [Normalize(SHOULDERS_75), Select(POINTS_HANDS_42)],

    # temporal: every clip resampled to 64 frames
    "temporal_fixed64":      [Select(POINTS_27), Normalize(SHOULDERS_27), TemporalResample(64)],
    "hands_only_temporal64": [Select(POINTS_HANDS_20), Normalize(WRISTS_HANDS_20), TemporalResample(64)],

    # depth: same steps as the baseline; see THREE_D and NO_AUGMENTATION
    "depth_3d":          [Select(POINTS_27), Normalize(SHOULDERS_27)],
    "depth_3d_noaug":    [Select(POINTS_27), Normalize(SHOULDERS_27)],
    "depth_2d_noaug":    [Select(POINTS_27), Normalize(SHOULDERS_27)],
}

GRAPHS = {  # conditions not listed use GRAPH_27
    "hands_only": GRAPH_HANDS_20,
    "hands_only_shoulder_norm": GRAPH_HANDS_20,
    "hands_only_temporal64": GRAPH_HANDS_20,
    "hands_full_42": GRAPH_HANDS_42,
    "hands_full_42_shoulder_norm": GRAPH_HANDS_42,
}
THREE_D = {"depth_3d", "depth_3d_noaug"}
NO_AUGMENTATION = {"depth_3d_noaug", "depth_2d_noaug"}


def build_transforms(condition, train, trim_hand_span=False):
    steps = [TrimToHandSpan()] if trim_hand_span else []
    steps += CONDITIONS[condition]
    if train and condition not in NO_AUGMENTATION:
        steps += [ShearTransform(shear_std=0.1), RotationTransform(rotation_std=0.1)]
    return Compose(steps)


def channels_for(condition):
    return 3 if condition in THREE_D else 2


def graph_for(condition):
    return GRAPHS.get(condition, GRAPH_27)
