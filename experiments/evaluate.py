"""
Evaluates one ASL Citizen-trained checkpoint on:
    test    ASL Citizen test split (11 signers unseen in training/validation)
    wlasl   every cleaned WLASL clip of the shared vocabulary (cross-dataset)
Reports top-1/5/10 per set, per-signer accuracy on ASL Citizen test, per-source accuracy on WLASL,
per-class accuracy, and per-instance correctness (for paired comparisons between conditions).

Usage:
    python evaluate.py --condition baseline --run_name baseline_s0
Loads checkpoints/{run_name}/best.pt and writes eval_results/{run_name}.json.
"""
import argparse
import collections
import csv
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  (also puts model/ and experiments/ on the import path)

from ablation_configs import build_transforms, channels_for, graph_for
from dataset import PoseDataset
from stgcn_model import build_stgcn
from train import accuracy_topk

POSE_DIRS = [paths.CITIZEN_POSES, paths.WLASL_POSES]
SETS = ("test", "wlasl")


def group_of():
    """video_id -> ASL Citizen participant, or WLASL source."""
    g = {}
    for s in ("train", "val", "test"):
        with open(os.path.join(paths.CITIZEN_SPLITS, f"{s}.csv"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                g[os.path.splitext(r["Video file"])[0]] = r["Participant ID"]
    for insts in json.load(open(paths.WLASL_CLIPS)).values():
        for i in insts:
            g[i["video_id"]] = i["source"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--num_class", type=int, default=None)
    args = ap.parse_args()

    classes = [l.rstrip("\n").split("\t") for l in open(paths.CLASS_LIST, encoding="utf-8")]
    num_class = args.num_class or len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_stgcn(num_class, graph_for(args.condition), in_channels=channels_for(args.condition)).to(device)
    ckpt = torch.load(os.path.join(paths.CHECKPOINTS, args.run_name, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    groups = group_of()

    result = {"condition": args.condition, "run_name": args.run_name, "best_epoch": ckpt.get("epoch"),
              "trim_hand_span": ckpt.get("trim_hand_span", False),
              "best_val_acc": ckpt.get("val_acc"), "sets": {}}
    for subset in SETS:
        ds = PoseDataset(paths.SPLIT, subset, POSE_DIRS,
                         build_transforms(args.condition, train=False, trim_hand_span=ckpt.get("trim_hand_span", False)),
                         use_z=channels_for(args.condition) == 3)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, collate_fn=PoseDataset.collate_fn)
        c1 = c5 = c10 = n = 0
        per_instance, per_group, per_class = {}, collections.defaultdict(list), collections.defaultdict(list)
        with torch.no_grad():
            for b in loader:
                logits = model(b["frames"].to(device))
                labels = b["labels"].to(device)
                a, b5, b10 = accuracy_topk(logits, labels, topk=(1, 5, 10))
                c1, c5, c10, n = c1 + a, c5 + b5, c10 + b10, n + labels.size(0)
                for vid, lbl, pred in zip(b["files"], labels.tolist(), logits.argmax(1).tolist()):
                    ok = int(lbl == pred)
                    per_instance[vid] = ok
                    per_group[groups.get(vid, "?")].append(ok)
                    per_class[classes[lbl][2]].append(ok)
        grp = {g: {"acc": sum(v) / len(v), "n": len(v)} for g, v in per_group.items()}
        accs = [x["acc"] for x in grp.values() if x["n"] >= 20]
        result["sets"][subset] = {
            "n": n, "top1": c1 / n, "top5": c5 / n, "top10": c10 / n,
            "per_group": grp,
            "group_cv": statistics.pstdev(accs) / statistics.mean(accs) if len(accs) > 1 and statistics.mean(accs) else None,
            "per_class": {k: {"acc": sum(v) / len(v), "n": len(v)} for k, v in per_class.items()},
            "per_instance": per_instance,
        }
        s = result["sets"][subset]
        print(f"{subset:13s} n={n:5d}  top1={s['top1']:.3f} top5={s['top5']:.3f} top10={s['top10']:.3f}  "
              f"group CV={s['group_cv'] if s['group_cv'] is None else round(s['group_cv'], 3)}")
    os.makedirs(paths.EVAL_RESULTS, exist_ok=True)
    out = os.path.join(paths.EVAL_RESULTS, f"{args.run_name}.json")
    json.dump(result, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
