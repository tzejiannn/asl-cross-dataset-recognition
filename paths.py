"""
Where everything lives in this project. Every script imports this file, so a folder only has to be
renamed or moved here.

Importing it also lets scripts in any folder import the shared model code (model/) and the
ablation conditions (experiments/).
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(ROOT, "data")

# the split: video_id -> {"subset": train / val / test (ASL Citizen) or wlasl, "action": [class_id, 1, -1]}
SPLIT = os.path.join(DATA, "split.json")
CLASS_LIST = os.path.join(DATA, "class_list.txt")  # class id, ASL Citizen sign, WLASL word

# WLASL clips (test only)
WLASL_VIDEOS = os.path.join(DATA, "wlasl", "videos")
WLASL_CLIPS = os.path.join(DATA, "wlasl", "wlasl_clips.json")  # word -> [{video_id, source, signer_id}]

# ASL Citizen clips (training, validation and test)
CITIZEN_VIDEOS = os.path.join(DATA, "asl_citizen", "videos")

# extracted poses
WLASL_POSES = os.path.join(DATA, "poses", "wlasl")
CITIZEN_POSES = os.path.join(DATA, "poses", "asl_citizen")

# training and evaluation
TRAINING_DIR = os.path.join(ROOT, "experiments")
CHECKPOINTS = os.path.join(TRAINING_DIR, "checkpoints")  # checkpoints/{run_name}/best.pt
EVAL_RESULTS = os.path.join(TRAINING_DIR, "eval_results")

# results
RESULTS_DIR = os.path.join(ROOT, "results")

for _d in (os.path.join(ROOT, "model"), TRAINING_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)
