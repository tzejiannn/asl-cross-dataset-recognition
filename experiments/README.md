# Experiments

Trains and evaluates the ST-GCN under each ablation condition. Every run trains on ASL Citizen
(train set, with the checkpoint picked on the val set) and is then tested on both the ASL Citizen
test set and the cleaned WLASL clips. It needs the built dataset in `data/`, including the poses
(see [`pipeline/README.md`](../pipeline/README.md)).

All commands below run from the repository root.

## Running the ablation study

```
bash experiments/run_experiments.sh "<conditions>" "<seeds>"
```

For example, the whole study (the baseline and 14 conditions, 3 seeds each = 45 runs):

```
bash experiments/run_experiments.sh "baseline norm_off norm_frame_level wrist_norm_27pt conf_mask conf_mask_no_norm hands_only hands_only_shoulder_norm hands_full_42 hands_full_42_shoulder_norm temporal_fixed64 hands_only_temporal64 depth_3d depth_3d_noaug depth_2d_noaug" "0 1 2"
```

For each condition and seed, the script trains a model (`train.py`), then evaluates it
(`evaluate.py`). Runs are named `{condition}_s{seed}` and go one at a time, because two at once can
run out of memory on a 12 GB GPU. A run whose eval result already exists is skipped, so the queue
can be stopped and restarted. A failed training run is retried once before the queue stops.

A run takes about 20–60 minutes on one GPU: about 7 s per epoch, and 169–515 epochs before early
stopping.

| Output | Contents |
|---|---|
| `checkpoints/{run}/best.pt` | The model with the best validation accuracy, plus its epoch, condition and seed |
| `logs/{run}.log` | Loss and accuracy for every epoch |
| `eval_results/{run}.json` | Test results (see [Evaluating a run](#evaluating-a-run)) |

To compile the results into tables, see [`results/README.md`](../results/README.md).

## The conditions

Most conditions change one thing about how poses are prepared before they reach the model; a few
combine two changes. They're defined in `ablation_configs.py`.

Every run first crops each clip to the frames where a hand is detected, plus 5 frames either side,
capped at 150 frames. At training time, random shear and rotation are applied after the condition's
own steps (except in the two `noaug` conditions).

| Group | Condition | Change from the baseline |
|---|---|---|
| | `baseline` | 27 points (nose, eyes, shoulders, elbows and 10 per hand), centred on the shoulders and scaled by shoulder width over the whole clip |
| Spatial normalisation | `norm_off` | No normalisation |
| | `norm_frame_level` | Shoulder normalisation computed for each frame instead of once per clip |
| | `wrist_norm_27pt` | Centred and scaled on the two wrists instead of the shoulders |
| Low-confidence landmarks | `conf_mask` | Body points with MediaPipe visibility below 0.5 set to (0, 0) |
| | `conf_mask_no_norm` | The same, with no normalisation |
| Keypoint selection | `hands_only` | Only the 20 hand points, normalised on the wrists |
| | `hands_only_shoulder_norm` | Normalised on the shoulders first, then only the 20 hand points kept |
| | `hands_full_42` | All 21 points per hand, normalised on the wrists |
| | `hands_full_42_shoulder_norm` | Normalised on the shoulders first, then all 42 hand points kept |
| Temporal representation | `temporal_fixed64` | Every clip resampled to 64 frames |
| | `hands_only_temporal64` | `hands_only`, with every clip resampled to 64 frames |
| Depth and augmentation | `depth_3d` | x, y and z instead of x and y |
| | `depth_3d_noaug` | x, y and z, without augmentation |
| | `depth_2d_noaug` | x and y, without augmentation (the control for `depth_3d_noaug`) |

The hand-only conditions use a skeleton graph of the hands alone, with an edge joining the two
wrists. With 3D input, shear and rotation act on x and y and leave z unchanged, so `depth_3d` gets
the same augmentation as the baseline.

To add a condition, add an entry to `CONDITIONS` in `ablation_configs.py` (a list of transforms from
`model/pose_transforms.py`). If it uses a different point set, add its graph to `GRAPHS`. If it
uses z, add it to `THREE_D`.

## Training a single run

```
python experiments/train.py --condition baseline --seed 0 --run_name baseline_s0 --trim_hand_span
```

| Option | Default | Meaning |
|---|---|---|
| `--condition` | (required) | A condition from the table above |
| `--seed` | none | Seeds Python, NumPy and PyTorch, for a repeatable run |
| `--run_name` | the condition | Folder name under `checkpoints/` |
| `--trim_hand_span` | off | Crop clips to the hand activity first. Every run in the study uses it |
| `--max_epochs` | 1000 | Upper limit on epochs |
| `--early_stop_patience` | 80 | Stop after this many epochs without a better validation accuracy |
| `--batch_size`, `--lr`, `--dropout_ratio`, `--cosine_t_max` | 32, 1e-3, 0.05, 10 | Hyperparameters, from OpenHands' ST-GCN config for WLASL |
| `--split_file`, `--pose_dir`, `--out_dir` | `data/` paths, `checkpoints/{run_name}` | Where to read the split and poses, and where to save |

Training uses Adam with a cosine learning-rate schedule. The log goes to the terminal;
`run_experiments.sh` saves it to `logs/`.

## Evaluating a run

```
python experiments/evaluate.py --condition baseline --run_name baseline_s0
```

Loads `checkpoints/{run_name}/best.pt` and tests it on the ASL Citizen test set (`test`) and on
WLASL (`wlasl`). It prints top-1, top-5 and top-10 accuracy for each, and writes
`eval_results/{run_name}.json` with, for each set:

- `top1`, `top5`, `top10`: accuracy over all clips
- `per_source` (WLASL only): top-1 accuracy and clip count for each source website

`--condition` must be the one the run was trained with, because it sets the preprocessing and the
model's input shape.
