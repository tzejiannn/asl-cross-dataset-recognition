# WLASL × ASL Citizen — cross-dataset sign recognition

An ST-GCN sign classifier trained on ASL Citizen and tested on WLASL, over the 67 signs the two
datasets share. Ablations look at which pose-preprocessing choices make the model hold up when it is
moved to a different dataset.

**Final dataset:** 67 signs, 2,050 ASL Citizen clips (train 973 / val 258 / test 819, ASL Citizen's
own signer-independent split) and 791 cleaned WLASL clips (test only).

## Data

This repository contains no data: the WLASL and ASL Citizen licences don't allow it to be
redistributed. Instead, the dataset is rebuilt from the original sources with the scripts in
`pipeline/`, one folder per stage, run in order. **[`data/README.md`](data/README.md)** has the
download steps, what each stage reads and writes, the commands, and the final 67-sign vocabulary.

| Stage | What it does |
|---|---|
| `01_wlasl_download/` | Downloads WLASL100 (the first 100 words of WLASL) with WLASL's own scripts, filling in dead links from the Hugging Face copy: 1,265 clips |
| `02_vocabulary_overlap/` | Keeps a word only if ASL Citizen has exactly one sign for it: 100 words → 67 |
| `03_wlasl_filter/` | Keeps the WLASL clips of those 67 words: 843 clips |
| `04_asl_citizen_download/` | Downloads only the ASL Citizen clips of the 67 signs: 2,050 clips |
| `05_split/` | ASL Citizen's own signer-independent split (train 973 / val 258 / test 819); WLASL is test only |
| `06_pose_extraction/` | MediaPipe Holistic poses for both datasets, with the same settings |
| `07_baseline/` | Three baseline models trained on ASL Citizen, used only to decide which WLASL clips to check |
| `08_wlasl_clean/` | WLASL cleaning, 843 → 791: the northtexas source is removed (clips cut at the wrong points), and so are 21 clips that show a different sign from ASL Citizen's. Low accuracy decides which clips to look at; whether a clip is removed is decided by watching it |
| `09_build_dataset/` | Assembles the final dataset in `data/`; the stage 06 scripts then extract its poses for the experiments |

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
