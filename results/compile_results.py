"""
Compiles every run's evaluation (eval_results/{condition}_s{seed}.json) into
per-condition tables for report Chapter 7: mean ± SD over seeds of top-1/5/10 on the ASL Citizen
test set and on WLASL, the cross-dataset drop, and the coefficient of variation across ASL Citizen
test signers and across WLASL sources. A difference from the baseline is marked as a finding (*)
only if it is larger than the seed standard deviation of both conditions (report Section 5.5).

Usage:
    python compile_results.py
Reads experiments/eval_results/ and writes ablation_results.md next to this script.
"""
import collections
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  (also puts model/ and experiments/ on the import path)

TYPES = [
    ("Spatial normalisation", ["norm_off", "norm_frame_level", "wrist_norm_27pt"]),
    ("Low-confidence landmarks", ["conf_mask", "conf_mask_no_norm"]),
    ("Landmark subset", ["hands_only", "hands_only_shoulder_norm", "hands_full_42", "hands_full_42_shoulder_norm"]),
    ("Temporal representation", ["temporal_fixed64", "hands_only_temporal64"]),
    ("Depth (with augmentation control)", ["depth_3d", "depth_2d_noaug", "depth_3d_noaug"]),
]


def load():
    runs = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(paths.EVAL_RESULTS, "*.json"))):
        r = json.load(open(f))
        runs[r["condition"]].append(r)
    return runs


def stats(rs):
    s = {}
    for sub in ("test", "wlasl"):
        for k in ("top1", "top5", "top10"):
            xs = [100 * r["sets"][sub][k] for r in rs]
            s[f"{sub}_{k}"] = (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)
        cv = [r["sets"][sub]["group_cv"] for r in rs]
        s[f"{sub}_cv"] = (st.mean(cv), st.stdev(cv) if len(cv) > 1 else 0.0)
    drops = [100 * (r["sets"]["test"]["top1"] - r["sets"]["wlasl"]["top1"]) for r in rs]
    s["drop"] = (st.mean(drops), st.stdev(drops) if len(drops) > 1 else 0.0)
    s["n"] = len(rs)
    return s


def diff(s, b, key, higher_is_better=True):
    d = s[key][0] - b[key][0]
    mark = "*" if abs(d) > max(s[key][1], b[key][1]) else ""
    return f"{d:+.1f}{mark}"


def row(name, s, b):
    f = lambda k, p=1: f"{s[k][0]:.{p}f} ± {s[k][1]:.{p}f}"
    return (f"| {name} | {s['n']} | {f('test_top1')} | {diff(s, b, 'test_top1')} | {f('wlasl_top1')} | {diff(s, b, 'wlasl_top1')} | "
            f"{f('drop')} | {diff(s, b, 'drop')} | {f('test_cv', 2)} | {f('wlasl_cv', 2)} |")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # the tables contain ± and Δ, which the Windows console can't print by default
    runs = load()
    S = {c: stats(rs) for c, rs in runs.items()}
    b = S["baseline"]
    head = ("| Condition | Seeds | ASL Citizen top-1 (%) | Δ | WLASL top-1 (%) | Δ | Drop (pts) | Δ | CV signers | CV sources |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    out = ["# Ablation results (ASL Citizen study, 67 signs)", "",
           "Mean ± SD over seeds. Δ = difference from baseline; * = larger than the seed SD of both conditions "
           "(report Section 5.5). Drop = ASL Citizen top-1 − WLASL top-1; a smaller drop is an improvement only if "
           "WLASL top-1 is equal or higher (report Section 5.3). CV = coefficient "
           "of variation of top-1 across ASL Citizen test signers / WLASL sources with ≥ 20 clips (lower is more even).", ""]
    out += ["## Baseline", "", head, row("baseline", b, b), ""]
    for title, conds in TYPES:
        out += [f"## {title}", "", head, row("baseline", b, b)]
        out += [row(f"`{c}`", S[c], b) for c in conds if c in S]
        out.append("")
    out += ["## All conditions, sorted by WLASL top-1", "", head]
    out += [row(f"`{c}`" if c != "baseline" else "baseline", S[c], b)
            for c in sorted(S, key=lambda c: -S[c]["wlasl_top1"][0])]
    out += ["", "## Top-5 / top-10", "",
            "| Condition | ASL Citizen top-5 | ASL Citizen top-10 | WLASL top-5 | WLASL top-10 |", "|---|---:|---:|---:|---:|"]
    out += [f"| {c} | {S[c]['test_top5'][0]:.1f} | {S[c]['test_top10'][0]:.1f} | {S[c]['wlasl_top5'][0]:.1f} | {S[c]['wlasl_top10'][0]:.1f} |"
            for c in sorted(S, key=lambda c: -S[c]["wlasl_top1"][0])]
    path = os.path.join(paths.RESULTS_DIR, "ablation_results.md")
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out[out.index("## All conditions, sorted by WLASL top-1"):]))
    print(f"\nwrote {os.path.abspath(path)}; runs per condition: { {c: S[c]['n'] for c in S} }")


if __name__ == "__main__":
    main()
