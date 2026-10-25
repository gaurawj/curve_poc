"""Run the full POC on synthetic data: train per material, score labelled test curves at
several fault severities, report precision / recall by layer and fault type, and render
example plots.

    python run_poc.py                       # default sizes, writes ./results
    python run_poc.py --n-per-fault 60 --seed 3
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from detect import CurveAnomalyDetector, explain  # noqa: E402
from features import extract  # noqa: E402
from synth import FAULTS, MATERIALS, make_dataset  # noqa: E402

# chart palette: slots 1-6 of the validated reference palette (light mode, fixed order)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUE, ORANGE = SERIES[0], SERIES[1]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LAYERS = [("rule_flag", "Rules only"), ("ml_flag", "ML only"), ("flagged", "Rules + ML")]


def featurize(curves):
    rows, locs = [], []
    for c in curves:
        f, l = extract(c)
        rows.append(f)
        locs.append(l)
    return pd.DataFrame(rows), locs


def overlap(a, b, tol=0.003):
    return a[0] <= b[1] + tol and b[0] <= a[1] + tol


def prf(df: pd.DataFrame, col: str) -> dict:
    y, p = df.label != "normal", df[col]
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, fpr=fp / (fp + tn))


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def main(a):
    os.makedirs(a.out, exist_ok=True)
    sevs = [float(x) for x in a.severities.split(",")]
    rows, keep = [], []
    for mi, mat in enumerate(MATERIALS):
        train = make_dataset(mat, a.n_train, 0, seed=a.seed + mi, contamination=a.contamination, tag="h")
        ftr, _ = featurize(train)
        det = CurveAnomalyDetector(ml_quantile=a.ml_quantile).fit(ftr)
        for si, sev in enumerate(sevs):
            test = make_dataset(mat, a.n_test_normal, a.n_per_fault, seed=a.seed + 100 * (si + 1) + mi,
                                tag=f"t{si}", severity=sev)
            fte, lte = featurize(test)
            for c, (_, f), loc in zip(test, fte.iterrows(), lte):
                rep = det.analyze(f, loc, c)
                p = rep.primary
                tseg = c["seg"]
                rows.append(dict(
                    id=c["id"], material=mat, severity=sev, label=c["label"], flagged=rep.flagged,
                    rule_flag=rep.rule_flag, ml_flag=rep.ml_flag, ml_score=round(rep.ml_score, 4),
                    primary=p.key if p else "", seg_start=p.segment[0] if p else np.nan,
                    seg_end=p.segment[1] if p else np.nan,
                    cause_ok=bool(p and p.key == c["label"]),
                    seg_hit=bool(p and tseg and p.key != "unexplained_shape" and overlap(p.segment, tseg)),
                    findings="; ".join(f"{x.key}: {x.detail}" for x in rep.findings),
                ))
                if sev == sevs[0]:
                    keep.append((c, rep))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "test_flags.csv"), index=False)

    # ---- metrics at the reference severity (first in the list)
    ref = df[df.severity == sevs[0]]
    res = {"reference_severity": sevs[0],
           "pooled": {name: prf(ref, col) for col, name in LAYERS}}
    y = (ref.label != "normal").to_numpy()
    res["ml_score_roc_auc"] = float(roc_auc_score(y, ref.ml_score))
    res["ml_score_average_precision"] = float(average_precision_score(y, ref.ml_score))
    faults = ref[ref.label != "normal"]
    rec = pd.DataFrame({name: faults.groupby("label")[col].mean() for col, name in LAYERS}).loc[FAULTS]
    tp = faults[faults.flagged]
    attr = tp.groupby("label").agg(flagged=("id", "size"), cause_ok=("cause_ok", "mean"),
                                   seg_hit=("seg_hit", "mean")).reindex(FAULTS)
    res["cause_accuracy_on_true_positives"] = float(tp.cause_ok.mean())
    res["segment_hit_on_true_positives"] = float(tp.seg_hit.mean())

    # ---- severity sweep (rules + ML)
    sweep_rows, rec_sev = [], {}
    for sev in sevs:
        d = df[df.severity == sev]
        m = prf(d, "flagged")
        yy = (d.label != "normal").to_numpy()
        sweep_rows.append(dict(severity=sev, precision=m["precision"], recall=m["recall"], fpr=m["fpr"],
                               ml_auc=float(roc_auc_score(yy, d.ml_score))))
        rec_sev[sev] = d[d.label != "normal"].groupby("label")["flagged"].mean().reindex(FAULTS)
    sweep = pd.DataFrame(sweep_rows)
    rec_by_sev = pd.DataFrame(rec_sev)
    rec_by_sev.columns = [f"severity {s:g}" for s in sevs]
    res["severity_sweep"] = sweep.to_dict(orient="records")
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    pooled = pd.DataFrame(res["pooled"]).T.reset_index().rename(columns={"index": "layer"})
    pooled[["tp", "fp", "fn", "tn"]] = pooled[["tp", "fp", "fn", "tn"]].astype(int)
    rec_t = rec.reset_index().rename(columns={"label": "fault"})
    attr_t = attr.reset_index().rename(columns={"label": "fault", "cause_ok": "cause correct", "seg_hit": "segment hit"})
    sev_t = rec_by_sev.reset_index().rename(columns={"label": "fault"})
    n_norm, n_fault = int((ref.label == "normal").sum()), int((ref.label != "normal").sum())
    md = f"""# POC results on synthetic data

Reference test set (severity {sevs[0]:g}): {n_norm} normal and {n_fault} faulty curves across {len(MATERIALS)} materials
({a.n_per_fault} per fault type per material). Models trained per material on
{a.n_train} historical curves with {a.contamination:.0%} unlabelled faults mixed in.

**These numbers come from synthetic curves written by the same person who wrote the detector.
They show the pipeline works end to end. They are not evidence of performance on real tests.**

## Flagging quality (pooled, severity {sevs[0]:g})

{md_table(pooled)}

ML score on its own (threshold-free): ROC AUC {res['ml_score_roc_auc']:.3f}, average precision {res['ml_score_average_precision']:.3f}.

## Recall by fault type (severity {sevs[0]:g})

{md_table(rec_t)}

## Cause and segment attribution (flagged faults only, severity {sevs[0]:g})

Overall cause correct: {res['cause_accuracy_on_true_positives']:.3f}; segment overlaps the true one: {res['segment_hit_on_true_positives']:.3f}
(ML-only flags carry no specific cause or segment by design, so they count as incorrect here).

{md_table(attr_t)}

## What happens when faults get subtler

Severity scales fault magnitude (1 = clear, 0.25 = subtle). Rules + ML.

{md_table(sweep)}

Recall by fault type:

{md_table(sev_t)}
"""
    with open(os.path.join(a.out, "metrics.md"), "w") as fh:
        fh.write(md)

    # ---- example annotations + plot (reference severity, S235)
    picks = {}
    for c, rep in keep:
        if c["material"] == "S235" and c["label"] in FAULTS and c["label"] not in picks and rep.flagged and rep.primary:
            picks[c["label"]] = (c, rep)
        if c["material"] == "S235" and c["label"] == "normal" and "normal" not in picks and not rep.flagged:
            picks["normal"] = (c, rep)
    with open(os.path.join(a.out, "example_annotations.txt"), "w") as fh:
        fh.write("\n\n".join(f"[true cause: {k}]\n{explain(r)}" for k, (c, r) in picks.items()))

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE})
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.4))

    def style(ax):
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8)

    for ax, key in zip(axs.flat, ["normal"] + FAULTS):
        if key not in picks:
            ax.axis("off")
            continue
        c, rep = picks[key]
        ax.plot(c["strain"] * 100, c["stress"], color=BLUE, lw=1.0)
        p = rep.primary
        if p:
            lo, hi = p.segment[0] * 100, p.segment[1] * 100
            if hi - lo < 0.6:                       # keep very short segments visible
                mid = (lo + hi) / 2
                lo, hi = mid - 0.3, mid + 0.3
            ax.axvspan(lo, hi, color=ORANGE, alpha=0.30, lw=0)
        ax.set_title(f"{key.replace('_', ' ')}" + (f"\nflag: {p.key.replace('_', ' ')}" if p else "\nno flag"),
                     fontsize=9, color=INK, loc="left")
        style(ax)
        ax.set_xlabel("Strain (%)", fontsize=8)
        ax.set_ylabel("Stress (MPa)", fontsize=8)

    ax = axs[1, 3]
    xs = np.arange(len(sevs))
    for k, fault in enumerate(FAULTS):
        ax.plot(xs, [rec_sev[s][fault] for s in sevs], color=SERIES[k], lw=1.6, marker="o", ms=6,
                label=fault.replace("_", " "))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s:g}" for s in sevs], fontsize=8)
    ax.set_xlim(-0.2, len(sevs) - 0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Fault severity (1 = clear, lower = subtler)", fontsize=8)
    ax.set_ylabel("Recall (rules + ML)", fontsize=8)
    ax.set_title("Recall as faults get subtler\n(synthetic data)", fontsize=9, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Curve plausibility POC: flagged segments (shaded) on synthetic S235 curves",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(a.out, "examples.png"), dpi=130)

    print(md)
    print("Wrote:", ", ".join(sorted(os.listdir(a.out))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-test-normal", type=int, default=300)
    ap.add_argument("--n-per-fault", type=int, default=30)
    ap.add_argument("--contamination", type=float, default=0.03)
    ap.add_argument("--ml-quantile", type=float, default=0.97)
    ap.add_argument("--severities", default="1.0,0.5,0.25")
    main(ap.parse_args())
