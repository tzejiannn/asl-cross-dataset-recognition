# Results

The script that turns the experiment outputs into the results tables. It runs from the repository
root and writes into `results/`.

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
- how uneven accuracy is across ASL Citizen test signers and across WLASL sources (coefficient of
  variation; lower is more even)

Each value is also compared with the baseline (Δ). A difference is marked `*` when it is larger
than the seed SD of both conditions; smaller differences aren't treated as findings. The
conditions are grouped as in the experiments, followed by all conditions sorted by WLASL accuracy
and a top-5 / top-10 table.
