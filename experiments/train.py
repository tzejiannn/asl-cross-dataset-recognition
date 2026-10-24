"""
Trains the ST-GCN on ASL Citizen under one ablation condition (see ablation_configs.py).

Hyperparameters follow OpenHands' `examples/configs/wlasl/st_gcn.yaml`: Adam (lr 1e-3), cosine
learning-rate schedule, batch size 32, dropout 0.05. Training stops after 80 epochs without a better
validation accuracy; the best model is saved to checkpoints/{run_name}/best.pt.

Usage:
    python train.py --condition baseline --seed 0 --run_name baseline_s0 --trim_hand_span
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  (also puts model/ and experiments/ on the import path)

from ablation_configs import build_transforms, channels_for, graph_for
from dataset import PoseDataset
from stgcn_model import build_stgcn


def accuracy_topk(logits, labels, topk=(1,)):
    """Number of clips whose label is among the k highest-scoring classes, for each k."""
    _, pred = logits.topk(max(topk), dim=1)
    correct = pred.t().eq(labels.view(1, -1))
    return [correct[:k].reshape(-1).float().sum().item() for k in topk]


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
        total_correct += accuracy_topk(logits, labels)[0]
        total_n += labels.size(0)
    return total_loss / total_n, total_correct / total_n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--run_name", default=None, help="Checkpoint folder name (default: the condition name)")
    parser.add_argument("--seed", type=int, default=None, help="Seed python/numpy/torch for a repeatable run")
    parser.add_argument("--trim_hand_span", action="store_true", help="Crop clips to the hand-activity span first")
    parser.add_argument("--split_file", default=paths.SPLIT)
    parser.add_argument("--pose_dir", nargs="+", default=[paths.CITIZEN_POSES, paths.WLASL_POSES])
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--num_class", type=int, default=None, help="Default: number of lines in the class list")
    parser.add_argument("--in_channels", type=int, default=None, help="Default: the condition's own (2 or 3)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout_ratio", type=float, default=0.05)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=80)
    parser.add_argument("--cosine_t_max", type=int, default=10)
    args = parser.parse_args()
    if args.in_channels is None:
        args.in_channels = channels_for(args.condition)
    if args.num_class is None:
        args.num_class = len(open(paths.CLASS_LIST, encoding="utf-8").readlines())

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    run_name = args.run_name or args.condition
    out_dir = args.out_dir or os.path.join(paths.CHECKPOINTS, run_name)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_z = args.in_channels == 3
    train_set = PoseDataset(args.split_file, "train", args.pose_dir,
                            build_transforms(args.condition, train=True, trim_hand_span=args.trim_hand_span), use_z)
    val_set = PoseDataset(args.split_file, "val", args.pose_dir,
                          build_transforms(args.condition, train=False, trim_hand_span=args.trim_hand_span), use_z)
    print(f"condition={args.condition} run={run_name} seed={args.seed} | classes: {args.num_class} | "
          f"train: {len(train_set)} clips, val: {len(val_set)} clips")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, collate_fn=PoseDataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False,
                                             num_workers=args.num_workers, collate_fn=PoseDataset.collate_fn)

    model = build_stgcn(args.num_class, graph_for(args.condition), in_channels=args.in_channels,
                        dropout_ratio=args.dropout_ratio).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_t_max)

    best_val_acc, epochs_without_improvement = 0.0, 0
    for epoch in range(args.max_epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        print(f"epoch {epoch:4d} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | {time.time() - t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc, epochs_without_improvement = val_acc, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                        "condition": args.condition, "split_file": args.split_file, "seed": args.seed,
                        "num_class": args.num_class, "trim_hand_span": args.trim_hand_span},
                       os.path.join(out_dir, "best.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping: no val_acc improvement for {args.early_stop_patience} epochs.")
                break
    print(f"Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
