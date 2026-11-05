"""
The baseline's WLASL top-1 accuracy per source website (report Sections 5.4 and 6.3): the mean over
seeds 0-2 for each source with at least 20 clips, and the lowest and highest of these.

Reads experiments/eval_results/baseline_s{0,1,2}.json.

Usage (from the repository root):
    python results/source_accuracy.py
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402

SEEDS = (0, 1, 2)
MIN_CLIPS = 20


def main():
    runs = [json.load(open(os.path.join(paths.EVAL_RESULTS, f"baseline_s{s}.json")))["sets"]["wlasl"] for s in SEEDS]
    key = "per_source" if "per_source" in runs[0] else "per_group"  # per_group in results written before the rename
    rows = [(src, d["n"], statistics.mean(r[key][src]["acc"] for r in runs)) for src, d in runs[0][key].items()]
    kept = sorted((r for r in rows if r[1] >= MIN_CLIPS), key=lambda r: -r[2])
    print(f"baseline WLASL top-1 by source, mean over seeds {list(SEEDS)}")
    for src, n, acc in kept:
        print(f"  {src:14s} {n:4d} clips  {100 * acc:5.1f}%")
    print(f"{len(kept)} of {len(rows)} sources have at least {MIN_CLIPS} clips; "
          f"lowest {100 * kept[-1][2]:.1f}%, highest {100 * kept[0][2]:.1f}%")
    print("left out:", ", ".join(f"{s} ({n})" for s, n, _ in sorted(rows) if n < MIN_CLIPS))


if __name__ == "__main__":
    main()
