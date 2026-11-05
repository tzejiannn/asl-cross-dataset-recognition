"""
Which body points the confidence mask (ConfidenceMask, threshold 0.5) removes, per dataset (report
Section 7.3): for each of the 7 body points of the 27-point subset, the share of frames with
visibility below 0.5, among the frames after trimming in which the signer's body is detected (any
of the 33 body confidences above 0). ASL Citizen's training clips are also shown on their own.

Usage (from the repository root):
    python results/body_visibility.py
"""
import numpy as np

from pose_stats import BODY_7, datasets, load_conf, trim


def main():
    sets = datasets()
    sets["ASL Citizen (train only)"] = [c for c in sets["ASL Citizen"] if c[1] == "train"]
    idx = list(BODY_7)
    for name, clips in sets.items():
        below, frames, total = np.zeros(len(idx)), 0, 0
        for v, sub in clips:
            c = trim(load_conf(v, sub))
            total += len(c)
            c = c[c[:, :33].max(1) > 0]
            frames += len(c)
            below += (c[:, idx] < 0.5).sum(0)
        print(f"{name}: {frames} of {total} frames after trimming have the body detected")
        for i, b in zip(idx, below):
            print(f"  {BODY_7[i]:15s} visibility < 0.5 in {100 * b / frames:5.1f}% ({int(b)} frames)")


if __name__ == "__main__":
    main()
