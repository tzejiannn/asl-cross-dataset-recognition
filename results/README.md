# Results

The scripts that turn the experiment outputs into the results tables, and that compute the
accuracy by source and the pose statistics quoted in the report. They run from the repository root.

## Ablation tables: `compile_results.py`

```
python results/compile_results.py
```

Reads every run in `experiments/eval_results/` (see [`experiments/README.md`](../experiments/README.md))
and writes `ablation_results.md`, with one row per condition. Each row gives the mean ± SD over
seeds of:

- top-1 accuracy on the ASL Citizen test set and on WLASL
- the drop from ASL Citizen to WLASL. A smaller drop is an improvement only if WLASL accuracy is
  equal or higher, since the drop also narrows when ASL Citizen accuracy falls

Each value is also compared with the baseline (Δ). A difference is marked `*` when it is larger
than the seed SD of both conditions; smaller differences aren't treated as findings. The
conditions are grouped by category as in the report, followed by all conditions sorted by WLASL
accuracy and the baseline's top-1, top-5 and top-10 on both test sets.

## Baseline accuracy by WLASL source: `source_accuracy.py`

```
python results/source_accuracy.py    # report Section 6.3
```

Reads the baseline's three runs in `experiments/eval_results/` and prints its WLASL top-1 accuracy
for each source website with at least 20 clips, averaged over the three runs, with the lowest and
highest.

## Pose statistics

```
python results/trimming_stats.py     # report Section 4.4
python results/wrist_detection.py    # report Section 4.6
python results/body_visibility.py    # report Section 7.3
```

Each reads only the confidence arrays of the pose files in `data/poses/` (see
[`model/README.md`](../model/README.md#pose-format)) for every clip in `data/split.json`, trims
them with `TrimToHandSpan` as every run does, and prints its results. `pose_stats.py` holds the code
they share.

| Script | Prints, for ASL Citizen and WLASL |
|---|---|
| `trimming_stats.py` | The share of frames that trimming removes, and the median clip length before and after |
| `wrist_detection.py` | The share of trimmed frames with at least one wrist missing (its hand not detected), and the share of clips in which the two wrists are never detected in the same frame |
| `body_visibility.py` | For each of the 7 body points of the 27-point subset, the share of trimmed frames with the body detected in which its visibility is below 0.5, the threshold of the confidence mask. ASL Citizen's training clips are also shown on their own |
