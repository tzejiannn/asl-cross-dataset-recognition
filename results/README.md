# Results

Scripts that turn the experiment outputs into tables and draw the normalisation figures. All
commands run from the repository root, and each script writes into `results/`.

## Ablation tables: `compile_results.py`

```
python results/compile_results.py
```

Reads every run in `experiments/eval_results/` (see [`experiments/README.md`](../experiments/README.md))
and writes `ablation_results.md`, with one row per condition. Each row gives the mean ± SD over
seeds of:

- top-1 accuracy on the ASL Citizen test set and on WLASL
- the drop from ASL Citizen to WLASL (lower means the model holds up better on a new dataset)
- how uneven accuracy is across ASL Citizen test signers and across WLASL sources (coefficient of
  variation; lower is more even)

Each value is also compared with the baseline (Δ). A difference is marked `*` when it is larger
than the seed SD of both conditions. The conditions are grouped as in the experiments, followed by
all conditions sorted by WLASL accuracy and a top-5 / top-10 table.

## Normalisation figures

Both scripts use WLASL clips only. Landmarks extracted from ASL Citizen are derived data under its
licence and aren't shown.

### Two clips, before and after normalisation: `normalisation_examples.py`

```
python results/normalisation_examples.py [--sign book] [--clips 07068 68012]
```

Takes two WLASL clips of the same sign from different source websites, where the signer is a
different size and in a different place in the frame. It plots their 27 landmarks on one frame,
before and after shoulder normalisation (the baseline's preprocessing). After normalisation the
two skeletons line up.

Writes to `normalisation_examples/`:

| File | Contents |
|---|---|
| `{clip}_raw.png`, `{clip}_normalised.png` | Each clip on its own, before and after |
| `overlay_raw.png`, `overlay_normalised.png` | Both clips overlaid |
| `figure.png` | All six panels in one figure |
| `landmarks.csv` | Raw and normalised x, y of the nose, shoulders, elbows and wrists on the plotted frame |
| `summary.txt` | Each clip's shoulder midpoint and shoulder width, before and after |

### Clip-level and per-frame normalisation: `normalisation_frames.py`

```
python results/normalisation_frames.py [--clip 66575] [--frames 9 24 45]
```

Plots a few frames of one WLASL clip three ways: (a) in image coordinates, (b) after clip-level
normalisation (the baseline), and (c) after per-frame normalisation (the `norm_frame_level`
condition). The default clip is the one whose shoulders move most: they rise and draw in as the
arms come up. Frame numbers count from the start of the clip after cropping to the hand activity.

In (b) every frame is shifted and scaled by the same amount, so the frames keep their arrangement
from (a); (a) is drawn over the same region as (b), so the two differ only in their axis labels. In
(c) each frame is normalised on its own, so the shoulders line up and the rest of each skeleton
moves.

Writes to `normalisation_examples/`:

| File | Contents |
|---|---|
| `frames_combined.png` | (a), (b) and (c) side by side, with one legend |
| `frames_a_image.png`, `frames_b_clip_level.png`, `frames_c_frame_level.png` | Each panel on its own |
| `frames_a_image_full.png` | (a) over the whole image |

The figures have no titles, so they can be captioned in the report.
