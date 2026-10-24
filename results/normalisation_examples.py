"""
Examples of landmarks before and after spatial normalisation, for the report.

Takes two WLASL clips of the same sign from different sources (so the signer's size and position
in the frame differ) and runs each through the baseline's
steps with and without the last one: TrimToHandSpan, PoseSelect(POINTS_27), then
CenterAndScaleNormalize(SHOULDERS_27) (shoulder midpoint to the origin, divided by the mean
shoulder width). The raw coordinates are MediaPipe's, as fractions of the image width and height.

Usage:
    python normalisation_examples.py [--sign book] [--clips 07068 68012]
Only WLASL clips are used: landmarks from ASL Citizen are derived data and aren't redistributed.
Writes normalisation_examples/{figure.png, landmarks.csv, summary.txt} next to this script.
"""
import argparse
import csv
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
from pose_config import GRAPH_27, POINTS_27, SHOULDERS_27  # noqa: E402
from pose_transforms import CenterAndScaleNormalize, Compose, PoseSelect, TrimToHandSpan  # noqa: E402

NAMES_27 = (["nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]
            + [f"left_hand_{i}" for i in range(10)] + [f"right_hand_{i}" for i in range(10)])
TABLE_POINTS = [0, 3, 4, 5, 6, 7, 17]  # nose, shoulders, elbows, wrists

OUT = os.path.join(paths.RESULTS_DIR, "normalisation_examples")


def load(video_id, pose_dir):
    pose = pickle.load(open(os.path.join(pose_dir, video_id + ".pkl"), "rb"))
    return {"frames": torch.tensor(pose["keypoints"][:, :, :2], dtype=torch.float32).permute(2, 0, 1),
            "confidences": torch.tensor(pose["confidences"], dtype=torch.float32)}


def run(video_id, pose_dir):
    """Returns (raw, normalised), each (T, 27, 2), plus the hand-confidence of each trimmed frame."""
    raw = Compose([TrimToHandSpan(), PoseSelect(POINTS_27)])(load(video_id, pose_dir))
    norm = CenterAndScaleNormalize(SHOULDERS_27)({"frames": raw["frames"].clone()})
    hands = ((raw["confidences"][:, 33] > 0) & (raw["confidences"][:, 54] > 0)).float()
    return raw["frames"].permute(1, 2, 0).numpy(), norm["frames"].permute(1, 2, 0).numpy(), hands.numpy()


def pick_frame(raw, hands):
    """The frame with both hands detected that is closest to the middle of the clip."""
    T = len(raw)
    both = [t for t in range(T) if hands[t]] or list(range(T))
    return min(both, key=lambda t: abs(t - T // 2))


def draw(ax, pts, title, color):
    for a, b in GRAPH_27["inward_edges"]:
        if (pts[a] != 0).any() and (pts[b] != 0).any():  # MediaPipe gives (0, 0) for undetected hands
            ax.plot(pts[[a, b], 0], pts[[a, b], 1], color=color, lw=1.5)
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color=color, zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign", default="book")
    ap.add_argument("--clips", nargs=2, default=["07068", "68012"], help="two WLASL video ids")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    sources = {i["video_id"]: i["source"] for i in json.load(open(paths.WLASL_CLIPS))[args.sign]}
    clips = [(f"{vid} ({sources[vid]})", vid, paths.WLASL_POSES, color)
             for vid, color in zip(args.clips, ["tab:orange", "tab:blue"])]
    results = {name: run(vid, d) for name, vid, d, _ in clips}
    frames = {name: pick_frame(raw, hands) for name, (raw, _, hands) in results.items()}

    # figure: one row per clip, raw | normalised, plus both clips overlaid in each space
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for row, (name, vid, _, color) in enumerate(clips):
        raw, norm, _ = results[name]
        t = frames[name]
        draw(axes[row, 0], raw[t], f"{name}, frame {t}: raw", color)
        axes[row, 0].set_xlim(0, 1), axes[row, 0].set_ylim(1.15, 0)  # image coordinates, y down
        draw(axes[row, 1], norm[t], f"{name}, frame {t}: normalised", color)
        axes[row, 1].invert_yaxis()
    for name, vid, _, color in clips:
        raw, norm, _ = results[name]
        draw(axes[0, 2], raw[frames[name]], "Both clips overlaid: raw", color)
        draw(axes[1, 2], norm[frames[name]], "Both clips overlaid: shoulder-normalised", color)
    axes[0, 2].set_xlim(0, 1), axes[0, 2].set_ylim(1.15, 0)
    axes[1, 2].invert_yaxis()
    for ax in axes[:, 1].tolist() + [axes[1, 2]]:
        ax.axhline(0, color="grey", lw=0.8), ax.axvline(0, color="grey", lw=0.8)
    axes[0, 2].legend(handles=[plt.Line2D([], [], color=c, label=n) for n, _, _, c in clips], fontsize=9)
    fig.suptitle(f"'{args.sign}': 27 landmarks before and after shoulder normalisation", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure.png"), dpi=150)
    plt.close(fig)

    # the same six panels as separate untitled images, normalised ones on shared axis limits
    allnorm = [results[n][1][frames[n]] for n, _, _, _ in clips]
    lo = min(p[p != 0].min() for p in allnorm) - 0.15
    hi = max(p.max() for p in allnorm) + 0.15
    for space in ("raw", "normalised"):
        for group in [[c] for c in clips] + [clips]:
            fig, ax = plt.subplots(figsize=(5, 5.5))
            for name, _, _, color in group:
                raw, norm, _ = results[name]
                draw(ax, (raw if space == "raw" else norm)[frames[name]], "", color)
            if space == "raw":
                ax.set_xlim(0, 1), ax.set_ylim(1.15, 0)
            else:
                ax.set_xlim(lo, hi), ax.set_ylim(hi, lo)
                ax.axhline(0, color="grey", lw=0.8), ax.axvline(0, color="grey", lw=0.8)
            if len(group) > 1:
                ax.legend(handles=[plt.Line2D([], [], color=c, label=n) for n, _, _, c in group], fontsize=9)
            name = "overlay" if len(group) > 1 else group[0][1]
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"{name}_{space}.png"), dpi=200)
            plt.close(fig)

    # table: selected landmarks on the plotted frame
    with open(os.path.join(OUT, "landmarks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "video_id", "frame", "landmark", "raw_x", "raw_y", "norm_x", "norm_y"])
        for name, vid, _, _ in clips:
            raw, norm, _ = results[name]
            t = frames[name]
            for p in TABLE_POINTS:
                w.writerow([name, vid, t, NAMES_27[p], *(f"{v:.4f}" for v in (*raw[t, p], *norm[t, p]))])

    # summary: where the body sits and how big it is, before and after
    lines = []
    for name, vid, _, _ in clips:
        raw, norm, _ = results[name]
        for label, x in (("raw", raw), ("normalised", norm)):
            l, r = x[:, SHOULDERS_27[0]], x[:, SHOULDERS_27[1]]
            mid = ((l + r) / 2).mean(0)
            width = ((l - r) ** 2).sum(-1) ** 0.5
            lines.append(f"{name:22s} {label:10s}  shoulder midpoint ({mid[0]:+.3f}, {mid[1]:+.3f})"
                         f"  shoulder width {width.mean():.3f} (frames {len(x)})")
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
