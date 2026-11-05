"""
Evaluates one ASL Citizen-trained checkpoint on:
    test    ASL Citizen test split (11 signers unseen in training/validation)
    wlasl   every cleaned WLASL clip of the shared vocabulary (cross-dataset)
Reports top-1/5/10 per set, and top-1 per source website on WLASL.

Usage:
    python evaluate.py --condition baseline --run_name baseline_s0
Loads checkpoints/{run_name}/best.pt and writes eval_results/{run_name}.json.
"""
import argparse
import collections
import json
import os
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


def source_of():
    """WLASL video_id -> source website."""
    return {i["video_id"]: i["source"] for insts in json.load(open(paths.WLASL_CLIPS)).values() for i in insts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--num_class", type=int, default=None)
    args = ap.parse_args()

    num_class = args.num_class or sum(1 for _ in open(paths.CLASS_LIST, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_stgcn(num_class, graph_for(args.condition), in_channels=channels_for(args.condition)).to(device)
    ckpt = torch.load(os.path.join(paths.CHECKPOINTS, args.run_name, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    sources = source_of()

    result = {"condition": args.condition, "run_name": args.run_name, "best_epoch": ckpt.get("epoch"),
              "trim_hand_span": ckpt.get("trim_hand_span", False),
              "best_val_acc": ckpt.get("val_acc"), "sets": {}}
    for subset in SETS:
        ds = PoseDataset(paths.SPLIT, subset, POSE_DIRS,
                         build_transforms(args.condition, train=False, trim_hand_span=ckpt.get("trim_hand_span", False)),
                         use_z=channels_for(args.condition) == 3)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, collate_fn=PoseDataset.collate_fn)
        c1 = c5 = c10 = n = 0
        per_source = collections.defaultdict(list)
        with torch.no_grad():
            for b in loader:
                logits = model(b["frames"].to(device))
                labels = b["labels"].to(device)
                a, b5, b10 = accuracy_topk(logits, labels, topk=(1, 5, 10))
                c1, c5, c10, n = c1 + a, c5 + b5, c10 + b10, n + labels.size(0)
                if subset == "wlasl":
                    for vid, lbl, pred in zip(b["files"], labels.tolist(), logits.argmax(1).tolist()):
                        per_source[sources[vid]].append(int(lbl == pred))
        result["sets"][subset] = {"n": n, "top1": c1 / n, "top5": c5 / n, "top10": c10 / n}
        if subset == "wlasl":
            result["sets"][subset]["per_source"] = {g: {"acc": sum(v) / len(v), "n": len(v)}
                                                    for g, v in per_source.items()}
        s = result["sets"][subset]
        print(f"{subset:13s} n={n:5d}  top1={s['top1']:.3f} top5={s['top5']:.3f} top10={s['top10']:.3f}")
    os.makedirs(paths.EVAL_RESULTS, exist_ok=True)
    out = os.path.join(paths.EVAL_RESULTS, f"{args.run_name}.json")
    json.dump(result, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
