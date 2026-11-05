# WLASL × ASL Citizen — cross-dataset sign recognition

An ST-GCN sign classifier trained on ASL Citizen and tested on WLASL, over the 67 signs the two
datasets share. Ablations look at which pose-preprocessing choices make the model hold up when it is
moved to a different dataset.

**Final dataset:** 67 signs, 2,050 ASL Citizen clips (train 973 / val 258 / test 819, ASL Citizen's
own signer-independent split) and 791 cleaned WLASL clips (test only).

## Setup

Python 3.11: `pip install -r requirements.txt`. For a GPU build of torch, install it first from
<https://pytorch.org>.

## Data

The repository contains no data: the WLASL and ASL Citizen licences don't allow it to be
redistributed. To get the data:

- **[`pipeline/README.md`](pipeline/README.md)**: step-by-step instructions to download both
  datasets and rebuild the final dataset, including how the vocabulary was chosen and how the WLASL
  test clips were cleaned.
- **[`data/README.md`](data/README.md)**: which clips to keep, what `data/` contains once built, and
  the final 67-sign vocabulary.

## Repository layout

```
data/          the rebuilt dataset goes here; README.md and the list of WLASL test clips
pipeline/      the scripts that build the dataset, stages 01–09
model/         the ST-GCN model and the pose preprocessing
experiments/   ablation conditions, training and evaluation; runs write their checkpoints, logs and eval results here
results/       the scripts for the results tables, accuracy by source and the pose statistics
paths.py       where everything lives; the scripts import it
requirements.txt  the Python packages, with the versions used
```

Each folder has a README explaining how to use it.

## Experiments

The baseline and 14 conditions, 3 seeds each = 45 runs. Every run trains on ASL Citizen and is tested on both the ASL
Citizen test set and WLASL.

```
bash experiments/run_experiments.sh "baseline hands_only ..." "0 1 2"   # train + evaluate each run
python results/compile_results.py                                       # → results/ablation_results.md
```

- **[`experiments/README.md`](experiments/README.md)**: the conditions, running the study, and
  training or evaluating a single run.
- **[`model/README.md`](model/README.md)**: the ST-GCN, the pose format, the point sets and the
  preprocessing transforms.
- **[`results/README.md`](results/README.md)**: the results tables, accuracy by WLASL source and the pose statistics.

Headline results (top-1, mean ± SD over 3 seeds):

| Condition | ASL Citizen | WLASL |
|---|---:|---:|
| baseline | 74.4 ± 2.8 | 44.1 ± 1.3 |
| `hands_only_temporal64` (highest mean) | 81.5 ± 0.7 | 58.2 ± 2.0 |

On WLASL, the five best conditions differ by less than their seed-to-seed SD, so they can't be
ranked against each other (report Section 8.6).

## Licences

- WLASL: C-UDA 1.0 (<https://dxli94.github.io/WLASL/>).
- ASL Citizen: Microsoft Research License Terms. Use is non-commercial research only, nothing
  derived from it may be redistributed, and the data must be deleted when the research is
  complete.
