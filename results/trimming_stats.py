"""
How much temporal trimming (TrimToHandSpan: first to last hand-detected frame, 5 frames either
side, clips over 150 frames subsampled to 150) shortens the clips, per dataset (report Section 4.4):
the share of frames removed, and the median clip length before and after.

Usage (from the repository root):
    python results/trimming_stats.py
"""
import numpy as np

from pose_stats import datasets, load_conf, pct, trim


def main():
    for name, clips in datasets().items():
        raw, span, capped = [], [], []
        for v, sub in clips:
            c = load_conf(v, sub)
            raw.append(len(c))
            span.append(len(trim(c, max_len=None)))
            capped.append(len(trim(c)))
        raw, span, capped = map(np.array, (raw, span, capped))
        print(f"{name}: {len(raw)} clips, {raw.sum()} frames before trimming")
        print(f"  frames removed: {pct(1 - capped.sum() / raw.sum())} "
              f"({pct(1 - span.sum() / raw.sum())} by the hand-span crop alone)")
        print(f"  median length: {np.median(raw):.0f} frames before, {np.median(capped):.0f} after")
        print(f"  clips still over 150 frames after the crop: {(span > 150).sum()}")


if __name__ == "__main__":
    main()
