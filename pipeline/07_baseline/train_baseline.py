"""
Trains the baseline ST-GCN on ASL Citizen (the 67 shared signs), once per seed (0, 1 and 2 by
default). Only the ASL Citizen train split is trained on and the val split chooses the checkpoint;
WLASL plays no part.

Baseline preprocessing (the same as the "baseline" condition in experiments/ablation_configs.py):
crop each clip to its hand activity, keep the 27-point skeleton, normalise on the shoulders, and
at training time add shear and rotation augmentation. x and y only, over the 27-point graph.

Hyperparameters follow OpenHands' `examples/configs/wlasl/st_gcn.yaml`: Adam (lr 1e-3), cosine
learning-rate schedule, batch size 32, dropout 0.05. Training stops after 80 epochs without a better
validation accuracy.

Usage:
    python train_baseline.py              # seeds 0, 1 and 2
    python train_baseline.py --seeds 0
Writes ../../experiments/checkpoints/baseline_s{seed}/best.pt and ../../experiments/logs/baseline_s{seed}.log
(the baseline runs of the ablation study are these same three models).
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.join(ROOT, "..")
SPLIT = os.path.join(PIPELINE, "05_split", "split_uncleaned.json")
CLASS_LIST = os.path.join(PIPELINE, "02_vocabulary_overlap", "class_list.txt")
CITIZEN_POSES = os.path.join(PIPELINE, "06_pose_extraction", "poses_asl_citizen")
CHECKPOINTS = os.path.join(ROOT, "..", "..", "experiments", "checkpoints")
LOGS = os.path.join(ROOT, "..", "..", "experiments", "logs")

sys.path.insert(0, os.path.join(ROOT, "..", "..", "model"))
from dataset import PoseDataset  # noqa: E402
from pose_config import GRAPH_27, POINTS_27, SHOULDERS_27  # noqa: E402
from pose_transforms import (  # noqa: E402
    CenterAndScaleNormalize,
    Compose,
    PoseSelect,
    RotationTransform,
    ShearTransform,
    TrimToHandSpan,
)
from stgcn_model import build_stgcn  # noqa: E402

# baseline configuration
GRAPH = GRAPH_27
IN_CHANNELS = 2  # x, y
BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
NUM_WORKERS = 2
LR = 1e-3
DROPOUT = 0.05
MAX_EPOCHS = 1000
EARLY_STOP_PATIENCE = 80
COSINE_T_MAX = 10


def build_transforms(train):
    steps = [TrimToHandSpan(), PoseSelect(POINTS_27), CenterAndScaleNormalize(SHOULDERS_27)]
    if train:
        steps += [ShearTransform(shear_std=0.1), RotationTransform(rotation_std=0.1)]
    return Compose(steps)


def run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass over the loader; trains if an optimizer is given. Returns (mean loss, accuracy)."""
    train = optimizer is not None
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        frames, labels = batch["frames"].to(device), batch["labels"].to(device)
        with torch.set_grad_enabled(train):
            logits = model(frames)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += labels.size(0)
    return total_loss / total_n, total_correct / total_n


def train(seed, device):
    run_name = f"baseline_s{seed}"
    out_dir = os.path.join(CHECKPOINTS, run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    log_file = open(os.path.join(LOGS, run_name + ".log"), "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_class = len(open(CLASS_LIST, encoding="utf-8").readlines())
    log(f"Using device: {device}")

    train_set = PoseDataset(SPLIT, "train", [CITIZEN_POSES], build_transforms(train=True))
    val_set = PoseDataset(SPLIT, "val", [CITIZEN_POSES], build_transforms(train=False))
    log(f"condition=baseline run={run_name} seed={seed} | classes: {num_class} | "
        f"train: {len(train_set)} clips, val: {len(val_set)} clips")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=NUM_WORKERS, collate_fn=PoseDataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=VAL_BATCH_SIZE, shuffle=False,
                                             num_workers=NUM_WORKERS, collate_fn=PoseDataset.collate_fn)

    model = build_stgcn(num_class, GRAPH, in_channels=IN_CHANNELS, dropout_ratio=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=COSINE_T_MAX)

    best_val_acc, epochs_without_improvement = 0.0, 0
    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        log(f"epoch {epoch:4d} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | {time.time() - t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc, epochs_without_improvement = val_acc, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                        "condition": "baseline", "split_file": SPLIT, "seed": seed,
                        "num_class": num_class, "trim_hand_span": True},
                       os.path.join(out_dir, "best.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                log(f"Early stopping: no val_acc improvement for {EARLY_STOP_PATIENCE} epochs.")
                break
    log(f"Best val_acc: {best_val_acc:.4f}")
    log_file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        train(seed, device)


if __name__ == "__main__":
    main()
