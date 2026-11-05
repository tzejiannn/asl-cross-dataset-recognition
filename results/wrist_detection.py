"""
How often the wrists, the reference points of wrist normalisation, are missing after trimming, per
dataset (report Section 4.6): the share of frames with at least one wrist missing, and the share of
clips in which the two wrists are never detected in the same frame.

A wrist is missing when its hand is not detected: MediaPipe gives a hand no per-point score, so its
confidence (index 33 left, 54 right) is 1 if the hand was detected and 0 if not.

Usage (from the repository root):
    python results/wrist_detection.py
"""
from pose_stats import LEFT_WRIST, RIGHT_WRIST, datasets, load_conf, pct, trim


def main():
    for name, clips in datasets().items():
        frames = one_missing = both_missing = never_both = 0
        for v, sub in clips:
            c = trim(load_conf(v, sub))
            left, right = c[:, LEFT_WRIST] > 0, c[:, RIGHT_WRIST] > 0
            frames += len(c)
            one_missing += (~left | ~right).sum()
            both_missing += (~left & ~right).sum()
            never_both += not (left & right).any()
        print(f"{name}: {len(clips)} clips, {frames} frames after trimming")
        print(f"  frames with at least one wrist missing: {pct(one_missing / frames)} "
              f"(both missing: {pct(both_missing / frames)})")
        print(f"  clips where the two wrists are never detected in the same frame: "
              f"{pct(never_both / len(clips))} ({never_both})")


if __name__ == "__main__":
    main()
