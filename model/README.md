# Model

The ST-GCN classifier and the code that loads and prepares poses for it. Nothing here is run
directly: `experiments/` and `pipeline/07_baseline/` import it.

| File | Contents |
|---|---|
| `stgcn_model.py` | The ST-GCN. `build_stgcn(num_class, graph, in_channels=2)` builds the model: an ST-GCN encoder over the given skeleton graph, followed by a fully connected classifier. Reimplemented from OpenHands, with the same layers and names |
| `graph.py` | Builds the graph adjacency matrices the ST-GCN layers use, from a skeleton graph. Ported from OpenHands |
| `pose_config.py` | The point sets (which of the 75 MediaPipe points to keep) and the matching skeleton graphs |
| `pose_transforms.py` | The preprocessing steps: cropping, point selection, normalisation, confidence masking, temporal resampling and augmentation |
| `dataset.py` | `PoseDataset`: loads the pose files of one subset of `data/split.json` and applies the transforms |

## Pose format

Each pose file (`data/poses/*/{clip}.pkl`) holds a dictionary:

- `keypoints`: an array of shape (frames, 75, 3), with x, y and z for each point. x and y are
  fractions of the image width and height, with y pointing down.
- `confidences`: an array of shape (frames, 75). MediaPipe's visibility for body points; 1 for a
  detected hand and 0 for a missing one.

The 75 points are the MediaPipe Holistic body (0–32) and the left (33–53) and right (54–74) hands,
21 points each with the wrist first. A hand that isn't detected is all zeros.

## Point sets and graphs (`pose_config.py`)

| Point set | Points | Graph |
|---|---|---|
| `POINTS_27` | Nose, eyes, shoulders, elbows, and 10 per hand (wrist, thumb tip, and the base and tip of each other finger), from OpenHands | `GRAPH_27` |
| `POINTS_HANDS_20` | The same 10 points per hand, without the body | `GRAPH_HANDS_20`, with an edge joining the two wrists |
| `POINTS_HANDS_42` | All 21 points per hand | `GRAPH_HANDS_42`, with an edge joining the two wrists |

Also defined: the shoulder and wrist pairs that normalisation uses as reference points, for each
point set (`SHOULDERS_27`, `WRISTS_27`, …, and `SHOULDERS_75` for the full 75 points).

## Transforms (`pose_transforms.py`)

Each transform takes a clip as a dictionary whose `"frames"` entry is a (channels, frames, points)
tensor, and returns it changed. `Compose([...])` chains them. Each experiment condition is a list
of these (see [`experiments/README.md`](../experiments/README.md#the-conditions)).

| Transform | What it does |
|---|---|
| `TrimToHandSpan(margin=5, max_len=150)` | Crops a clip to the frames where a hand is detected, plus `margin` frames either side, then evenly subsamples clips longer than `max_len` |
| `ConfidenceMask(threshold=0.5)` | Sets points with confidence below the threshold to (0, 0) |
| `PoseSelect(indexes)` | Keeps only the given points |
| `CenterAndScaleNormalize(reference, frame_level=False)` | Moves the midpoint of two reference points (the shoulders by default) to the origin and divides by their distance. Uses the average over the clip, or each frame's own values with `frame_level=True` |
| `TemporalResample(64)` | Stretches or squeezes every clip to a fixed number of frames |
| `ShearTransform`, `RotationTransform` | Random shear and rotation in the image plane, for training only. With 3D input, z is left unchanged |

`TrimToHandSpan` and `ConfidenceMask` read the confidences of all 75 points, so they must come
before `PoseSelect`.

## Using it in your own code

Importing `paths` (at the repository root) puts `model/` and `experiments/` on the import path.
This example runs from the repository root and classifies one WLASL clip with a trained baseline:

```python
import torch
import paths
from ablation_configs import build_transforms, channels_for, graph_for
from dataset import PoseDataset
from stgcn_model import build_stgcn

condition = "baseline"
ds = PoseDataset(paths.SPLIT, "wlasl", [paths.WLASL_POSES],
                 build_transforms(condition, train=False, trim_hand_span=True))
model = build_stgcn(num_class=67, graph=graph_for(condition), in_channels=channels_for(condition))
ckpt = torch.load("experiments/checkpoints/baseline_s0/best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

clip = ds[0]                                   # {"frames": (C, T, V) tensor, "label", "file"}
logits = model(clip["frames"].unsqueeze(0))    # (1, 67): one score per class
```

For batches, use `torch.utils.data.DataLoader` with `collate_fn=PoseDataset.collate_fn`, which pads
clips to the same length.
