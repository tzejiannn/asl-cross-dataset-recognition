# WLASL × ASL Citizen — cross-dataset sign recognition

An ST-GCN sign classifier trained on ASL Citizen and tested on WLASL, over the 67 signs the two
datasets share. Ablations look at which pose-preprocessing choices make the model hold up when it is
moved to a different dataset.

**Final dataset:** 67 signs, 2,050 ASL Citizen clips (train 973 / val 258 / test 819, ASL Citizen's
own signer-independent split) and 791 cleaned WLASL clips (test only).

## Data

This repository contains no data: the WLASL and ASL Citizen licences don't allow it to be
redistributed. **[`data/README.md`](data/README.md)** explains how to obtain both datasets, gives
the preprocessing and cleaning steps that rebuild the dataset, and lists the final 67-sign
vocabulary.

In short:

1. WLASL100 (the first 100 words of WLASL) is downloaded with WLASL's own scripts, and the missing
   clips are filled in from the Hugging Face copy.
2. A word is kept only if ASL Citizen has exactly one sign for it: 100 words → 67.
3. Only the ASL Citizen clips of those 67 signs are downloaded, and ASL Citizen's own
   signer-independent train / val / test split is kept.
4. MediaPipe Holistic poses are extracted from both datasets with the same settings.
5. The WLASL test clips are cleaned (843 → 791). Low accuracy decides which clips to look at;
   whether a clip is removed is decided by watching it. The whole northtexas source is removed
   (clips cut at the wrong points), and so are 21 clips that show a different sign from ASL
   Citizen's.

## Repository layout

```
data/          how to obtain and rebuild the data (README.md), and the list of WLASL test clips; the rebuilt dataset goes here
pipeline/      how the dataset is built, stages 01–09 in order (scripts and the small files that record each decision)
model/         the ST-GCN model code (used by the baseline and by every experiment)
experiments/   ablation conditions, training and evaluation; runs write their checkpoints, logs and eval results here
results/       scripts that compile the ablation tables and draw the normalisation figures
paths.py       where everything lives; experiments/ and results/ scripts import it
```

## Experiments

15 conditions × 3 seeds = 45 runs. Every run trains on ASL Citizen and is tested on both the ASL
Citizen test split and WLASL. The conditions are defined in `experiments/ablation_configs.py`.

```
bash experiments/run_experiments.sh "baseline hands_only ..." "0 1 2"   # train + evaluate; skips finished runs
python results/compile_results.py                                       # → results/ablation_results.md
```

Headline results (top-1, mean ± SD over 3 seeds):

| Condition | ASL Citizen | WLASL |
|---|---:|---:|
| baseline | 74.4 ± 2.8 | 44.1 ± 1.3 |
| `hands_only_temporal64` (best) | 81.5 ± 0.7 | 58.2 ± 2.0 |

`compile_results.py` writes the table for every condition to `results/ablation_results.md`.

## Licences

- WLASL: C-UDA 1.0 (<https://dxli94.github.io/WLASL/>).
- ASL Citizen: Microsoft Research License Terms. Use is non-commercial research only, nothing
  derived from it may be redistributed, and the data must be deleted when the research is
  complete.
