"""
Shows clip-level and per-frame normalisation on a few frames of one WLASL clip, for the report.

Writes three untitled images (captioned in the report):
    (a) frames_a_image.png       the frames in image coordinates (fractions of the frame), over the same
                                 region as (b) and (c), so it differs from (b) only in its tick labels
        frames_a_image_full.png  the same over the whole image
    frames_combined.png          (a), (b) and (c) side by side
    (b) frames_b_clip_level.png  after CenterAndScaleNormalize(SHOULDERS_27) (the baseline): every
                                 frame is shifted and scaled by the same amount, so the frames keep
                                 their arrangement from (a)
    (c) frames_c_frame_level.png after CenterAndScaleNormalize(SHOULDERS_27, frame_level=True)
                                 (the norm_frame_level condition): each frame has its own centre
                                 and scale, so the shoulders line up and the frames move relative to
                                 each other
Same steps as the baseline before normalising: TrimToHandSpan, then PoseSelect(POINTS_27).
The default clip (66575, 'study', aslsignbank) is the WLASL clip whose shoulders move most: they
rise and draw in as the arms come up. Only WLASL clips are used: landmarks from ASL Citizen are
derived data and aren't redistributed.

Usage:
    python normalisation_frames.py [--clip 66575] [--frames 9 24 45]
Frame numbers count from the start of the trimmed clip. Writes into normalisation_examples/.
"""
import argparse
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

OUT = os.path.join(paths.RESULTS_DIR, "normalisation_examples")


def draw(ax, pts, color, label):
    seen = (pts != 0).any(-1)  # MediaPipe gives (0, 0) for undetected points
    for a, b in GRAPH_27["inward_edges"]:
        if seen[a] and seen[b]:
            ax.plot(pts[[a, b], 0], pts[[a, b], 1], color=color, lw=1.5)
    ax.scatter(pts[seen, 0], pts[seen, 1], s=10, color=color, zorder=3, label=label)
    sh = [i for i in SHOULDERS_27 if seen[i]]  # the normalisation reference points, drawn larger
    ax.scatter(pts[sh, 0], pts[sh, 1], s=60, color=color, edgecolor="black", lw=0.8, zorder=4)


def plot(ax, frames, ts, colors, xlabel, ylabel, limits=None, axes_at_zero=False):
    for t, color in zip(ts, colors):
        draw(ax, frames[t], color, f"frame {t}")
    if limits:
        ax.set_xlim(*limits[0]), ax.set_ylim(*limits[1])
    if not ax.yaxis_inverted():
        ax.invert_yaxis()  # y points down, as in the image
    if axes_at_zero:
        ax.axhline(0, color="grey", lw=0.8), ax.axvline(0, color="grey", lw=0.8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)


def save(fig, path):
    fig.savefig(path + ".tmp.png", dpi=200, bbox_inches="tight")
    os.replace(path + ".tmp.png", path)  # an image viewer holding the old file can block writing it in place
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="66575")
    ap.add_argument("--frames", type=int, nargs="+", default=[9, 24, 45])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pose = pickle.load(open(os.path.join(paths.WLASL_POSES, args.clip + ".pkl"), "rb"))
    data = {"frames": torch.tensor(pose["keypoints"][:, :, :2], dtype=torch.float32).permute(2, 0, 1),
            "confidences": torch.tensor(pose["confidences"], dtype=torch.float32)}
    raw = Compose([TrimToHandSpan(), PoseSelect(POINTS_27)])(data)["frames"]
    center, scale = CenterAndScaleNormalize(SHOULDERS_27)._center_and_scale(raw.permute(1, 2, 0))  # clip-level
    clip = CenterAndScaleNormalize(SHOULDERS_27)({"frames": raw.clone()})["frames"]
    frame = CenterAndScaleNormalize(SHOULDERS_27, frame_level=True)({"frames": raw.clone()})["frames"]
    raw, clip, frame = (x.permute(1, 2, 0).numpy() for x in (raw, clip, frame))  # (T, 27, 2)

    colors = [plt.cm.viridis(v) for v in torch.linspace(0.1, 0.85, len(args.frames)).tolist()]

    # (b) and (c) on the same axis limits so they can be compared directly
    def fit(xs, pad):
        pts = [x[t][(x[t] != 0).any(-1)] for x in xs for t in args.frames]
        (x0, y0), (x1, y1) = (min(p.min(0)[i] for p in pts) for i in (0, 1)), (max(p.max(0)[i] for p in pts) for i in (0, 1))
        return (x0 - pad, x1 + pad), (y1 + pad, y0 - pad)
    norm_limits = fit((clip, frame), 0.15)
    # (a) shows the same window as (b) and (c), mapped back to image coordinates, so all three panels
    # share one size and scale and (a) differs from (b) only in its tick labels
    image_limits = tuple(tuple(float(v / scale + center[i]) for v in lim) for i, lim in enumerate(norm_limits))
    full_limits = ((0, 1), (max(1, image_limits[1][0]), 0))  # MediaPipe places hands just out of shot below y = 1

    image_labels = ("x (fraction of image width)", "y (fraction of image height, downwards)")
    norm_labels = ("x (shoulder widths)", "y (shoulder widths, downwards)")
    panels = [  # (file, frames, axis labels, options)
        ("frames_a_image.png", raw, image_labels, dict(limits=image_limits)),
        ("frames_b_clip_level.png", clip, norm_labels, dict(limits=norm_limits, axes_at_zero=True)),
        ("frames_c_frame_level.png", frame, norm_labels, dict(limits=norm_limits, axes_at_zero=True)),
        ("frames_a_image_full.png", raw, image_labels, dict(limits=full_limits)),
    ]

    # each panel on its own
    for name, x, labels, opts in panels:
        fig, ax = plt.subplots(figsize=(5, 5.5))
        plot(ax, x, args.frames, colors, *labels, **opts)
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
        save(fig, os.path.join(OUT, name))

    # (a), (b), (c) side by side with one legend
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 5.5))
    for ax, letter, (_, x, labels, opts) in zip(axes, "abc", panels):
        plot(ax, x, args.frames, colors, *labels, **opts)
        ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, os.path.join(OUT, "frames_combined.png"))

    for t in args.frames:
        l, r = SHOULDERS_27
        print(f"frame {t:3d}  shoulders  image {raw[t, l].round(3)} {raw[t, r].round(3)}"
              f"  clip-level {clip[t, l].round(3)} {clip[t, r].round(3)}"
              f"  per-frame {frame[t, l].round(3)} {frame[t, r].round(3)}")
    print(f"Wrote {', '.join(p[0] for p in panels)}, frames_combined.png to {OUT}")


if __name__ == "__main__":
    main()
