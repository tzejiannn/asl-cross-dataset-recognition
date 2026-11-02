"""
Cleaning check 1: the first-round evaluation (report Section 5.1). The three baseline models in
../07_baseline (trained on ASL Citizen only; no training happens here) classify every retrieved
WLASL clip of the 67 shared words, before any clip is removed (843 clips).

Poor accuracy is only used to decide which clips to look at:
    - by source: accuracy and share of frames with no person detected (a model-independent check)
      per WLASL source. A source whose clips are never recognised stands out (northtexas).
    - by sign: accuracy per sign, northtexas excluded. Signs more than one standard deviation below
      the mean are then checked by eye against ASL Citizen clips of the same sign.

Usage (from this folder):
    python first_round_eval.py
Writes first_round_baseline.json (overall, per source, clips wrong in every run) and
first_round_per_sign.json (per sign accuracy, mean and SD over signs).
"""
import collections
import json
import os
import pickle
import statistics
import sys

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "07_baseline"))
# the baseline's checkpoints, preprocessing and graph (also puts its model code on the import path)
from train_baseline import CHECKPOINTS, GRAPH, IN_CHANNELS, build_transforms  # noqa: E402
from dataset import PoseDataset  # noqa: E402
from stgcn_model import build_stgcn  # noqa: E402

SEEDS = (0, 1, 2)
CLASS_LIST = os.path.join(THIS_DIR, "..", "02_vocabulary_overlap", "class_list.txt")
WLASL_CLIPS = os.path.join(THIS_DIR, "..", "03_wlasl_filter", "wlasl_clips.json")
SPLIT = os.path.join(THIS_DIR, "..", "05_split", "split_uncleaned.json")
WLASL_POSES = os.path.join(THIS_DIR, "..", "06_pose_extraction", "poses_wlasl")


def main():
    classes = [l.rstrip("\n").split("\t") for l in open(CLASS_LIST, encoding="utf-8")]
    source = {i["video_id"]: i["source"] for insts in json.load(open(WLASL_CLIPS)).values() for i in insts}
    split = {v: e for v, e in json.load(open(SPLIT)).items() if e["subset"] == "wlasl"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = collections.defaultdict(list)    # video_id -> [0/1 per run]
    predicted = collections.defaultdict(list)  # video_id -> [predicted word per run]
    run_acc = []
    for seed in SEEDS:
        ckpt = torch.load(os.path.join(CHECKPOINTS, f"baseline_s{seed}", "best.pt"), map_location=device)
        model = build_stgcn(len(classes), GRAPH, in_channels=IN_CHANNELS).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        ds = PoseDataset(SPLIT, "wlasl", [WLASL_POSES], build_transforms(train=False))
        n_ok = 0
        with torch.no_grad():
            for b in torch.utils.data.DataLoader(ds, batch_size=64, collate_fn=PoseDataset.collate_fn):
                preds = model(b["frames"].to(device)).argmax(1).tolist()
                for vid, lbl, p in zip(b["files"], b["labels"].tolist(), preds):
                    correct[vid].append(int(lbl == p))
                    predicted[vid].append(classes[p][2])
                    n_ok += lbl == p
        run_acc.append(n_ok / len(ds))

    # model-independent check: share of frames with no person detected (body, left and right hand all absent)
    no_person = {}
    for vid in split:
        conf = pickle.load(open(os.path.join(WLASL_POSES, vid + ".pkl"), "rb"))["confidences"]
        no_person[vid] = float((conf.max(1) <= 0).mean())

    by_src = collections.defaultdict(list)
    for vid in split:
        by_src[source[vid]].append(vid)
    per_source = {s: {"clips": len(v),
                      "top1": statistics.mean(statistics.mean(correct[x]) for x in v),
                      "no_person_frames": statistics.mean(no_person[x] for x in v)}
                  for s, v in by_src.items()}
    always_wrong = sorted(v for v in split if not any(correct[v]))
    result = {"condition": "baseline", "seeds": list(SEEDS), "clips": len(split),
              "top1_per_run": run_acc, "per_source": per_source,
              "always_wrong": [{"video_id": v, "source": source[v],
                                "label": classes[split[v]["action"][0]][2],
                                "predicted": collections.Counter(predicted[v]).most_common(1)[0][0]}
                               for v in always_wrong]}
    json.dump(result, open(os.path.join(THIS_DIR, "first_round_baseline.json"), "w"), indent=1)

    # by sign, with the northtexas source left out
    by_sign = collections.defaultdict(list)
    for vid, e in split.items():
        if source[vid] != "northtexas":
            by_sign[classes[e["action"][0]][2]].append(statistics.mean(correct[vid]))
    per_sign = {w: {"acc": statistics.mean(v), "n": len(v)} for w, v in by_sign.items()}
    accs = [d["acc"] for d in per_sign.values()]
    per_sign_result = {"mean": statistics.mean(accs), "sd": statistics.stdev(accs),
                       "per_sign": dict(sorted(per_sign.items(), key=lambda kv: kv[1]["acc"]))}
    json.dump(per_sign_result, open(os.path.join(THIS_DIR, "first_round_per_sign.json"), "w"), indent=1)

    print(f"baseline: {len(split)} uncleaned WLASL clips, top-1 per run "
          f"{[round(100 * a, 1) for a in run_acc]} -> mean {100 * np.mean(run_acc):.1f}%")
    print(f"{'source':14s} {'clips':>5s} {'top-1':>7s} {'frames with no person':>22s}")
    for s, d in sorted(per_source.items(), key=lambda kv: kv[1]["top1"]):
        print(f"{s:14s} {d['clips']:5d} {100 * d['top1']:6.1f}% {100 * d['no_person_frames']:21.1f}%")
    print(f"clips wrong in every run: {len(always_wrong)}")
    cut = per_sign_result["mean"] - per_sign_result["sd"]
    print(f"by sign (northtexas excluded): mean {100 * per_sign_result['mean']:.1f}%, SD "
          f"{100 * per_sign_result['sd']:.1f}; below {100 * cut:.1f}%: "
          f"{[w for w, d in per_sign_result['per_sign'].items() if d['acc'] < cut]}")


if __name__ == "__main__":
    main()
