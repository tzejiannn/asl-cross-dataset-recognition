"""
Shared code for the pose statistics scripts (trimming_stats.py, wrist_detection.py,
body_visibility.py). They read only the confidence arrays of the pose files in data/poses/.
"""
import json
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  (also puts model/ and experiments/ on the import path)
from pose_transforms import TrimToHandSpan  # noqa: E402

BODY_7 = {0: "nose", 2: "left eye", 5: "right eye", 11: "left shoulder", 12: "right shoulder",
          13: "left elbow", 14: "right elbow"}  # the body points of the 27-point subset
LEFT_WRIST, RIGHT_WRIST = 33, 54


def datasets():
    """{"ASL Citizen": [(video_id, subset)] over train/val/test, "WLASL": [(video_id, "wlasl")]}, from data/split.json."""
    out = {"ASL Citizen": [], "WLASL": []}
    for v, e in json.load(open(paths.SPLIT)).items():
        out["WLASL" if e["subset"] == "wlasl" else "ASL Citizen"].append((v, e["subset"]))
    return out


def load_conf(video_id, subset):
    """The (frames, 75) confidence array of one clip."""
    d = paths.WLASL_POSES if subset == "wlasl" else paths.CITIZEN_POSES
    return pickle.load(open(os.path.join(d, video_id + ".pkl"), "rb"))["confidences"]


def trim(conf, max_len=150):
    """The confidences after TrimToHandSpan (margin 5, cap max_len; None = no cap), as in every run."""
    c = torch.tensor(conf, dtype=torch.float32)
    data = {"frames": torch.zeros(1, c.shape[0], 75), "confidences": c}
    return TrimToHandSpan(margin=5, max_len=max_len)(data)["confidences"].numpy()


def pct(x):
    return f"{100 * x:.1f}%"
