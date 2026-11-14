"""Sprint work (14 Nov). Compare methods on the eval split.

Two ways of comparing, so no method wins just by flagging more:
  1. AUROC / average precision: no threshold needed.
  2. Recall at a fixed false-positive rate (10%), threshold set on dev.
The F1-best threshold (also set on dev) is reported too.

Short sentences are a confound (a 3-word fragment can have 100% word overlap
but low cosine to a 120-word chunk), so the overlap analysis is also run on
sentences with at least --min_words words.

    python evaluate.py --scores results/scores.jsonl --sentences prepared/sentences.jsonl --out results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

METHODS = {   # support score: higher = more supported
    "cosine": lambda d: d.cos_max,
    "nli": lambda d: d.nli_entail,
    # exploratory heuristic: max entailment and max contradiction can come from different chunks
    "nli_e_minus_c": lambda d: d.nli_entail - d.nli_contra,
}
OVERLAP_BINS = [0, 0.5, 0.7, 0.85, 1.0001]
OVERLAP_NAMES = ["<50%", "50-70%", "70-85%", ">=85%"]
TARGET_FPR = 0.10


def f1_threshold(s, y):
    best_f, best_t = -1, None
    for t in np.unique(np.quantile(s, np.linspace(0.005, 0.995, 400))):
        flag = s < t
        tp = (flag & (y == 1)).sum()
        p = tp / max(flag.sum(), 1)
        r = tp / max((y == 1).sum(), 1)
        f = 2 * p * r / max(p + r, 1e-12)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def fpr_threshold(s, y, target):
    """Largest threshold whose false-positive rate on these rows is <= target."""
    neg = np.sort(s[y == 0])
    return float(neg[int(np.floor(target * len(neg)))]) if len(neg) else None


def at_threshold(s, y, t):
    flag = s < t
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(pos, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
            "fpr": fp / max(neg, 1), "flagged_share": float(flag.mean())}


def ranking(s, y):
    if len(set(y)) < 2:
        return {"auroc": None, "avg_precision": None}
    return {"auroc": float(roc_auc_score(y, -s)), "avg_precision": float(average_precision_score(y, -s))}


def group_table(ev, key, thr_fpr):
    out = {}
    for g, d in ev.groupby(key, observed=False):
        y = d.y.values
        row = {"n": int(len(d)), "positives": int(y.sum()),
               "prevalence": float(y.mean()) if len(d) else None}
        for m in METHODS:
            s = d[f"s_{m}"].values
            r = ranking(s, y) if len(d) else {"auroc": None, "avg_precision": None}
            r["recall_at_fpr10"] = at_threshold(s, y, thr_fpr[m])["recall"] if y.sum() else None
            row[m] = r
        out[str(g)] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores.jsonl")
    ap.add_argument("--sentences", default="prepared/sentences.jsonl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--min_words", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(a.scores, lines=True, dtype={"source_id": str, "example_id": str})
    sent = pd.read_json(a.sentences, lines=True, dtype={"source_id": str, "example_id": str})
    sent = sent[["example_id", "sentence"]]
    df = df.merge(sent, on="example_id", how="left", validate="one_to_one")
    assert df.sentence.notna().all(), "some scores have no matching sentence"
    df["n_words"] = df.sentence.str.split().str.len()
    df["overlap_bin"] = pd.cut(df.overlap_best, OVERLAP_BINS, labels=OVERLAP_NAMES, right=False)
    df["len_bin"] = pd.cut(df.n_words, [0, 6, 12, 20, 35, 10_000],
                           labels=["1-5", "6-11", "12-19", "20-34", "35+"], right=False)
    for m, f in METHODS.items():
        df[f"s_{m}"] = f(df)

    dev, ev = df[df.split == "dev"], df[df.split == "eval"].copy()
    thr_f1 = {m: f1_threshold(dev[f"s_{m}"].values, dev.y.values) for m in METHODS}
    thr_fpr = {m: fpr_threshold(dev[f"s_{m}"].values, dev.y.values, TARGET_FPR) for m in METHODS}

    rep = {"n_eval": int(len(ev)), "eval_prevalence": float(ev.y.mean()),
           "thresholds": {"f1_best_on_dev": thr_f1, f"fpr{int(TARGET_FPR*100)}_on_dev": thr_fpr},
           "overall": {}}
    for m in METHODS:
        s, y = ev[f"s_{m}"].values, ev.y.values
        rep["overall"][m] = {**ranking(s, y),
                             "at_f1_threshold": at_threshold(s, y, thr_f1[m]),
                             "at_fpr10_threshold": at_threshold(s, y, thr_fpr[m])}

    rep["by_task"] = group_table(ev, "task_type", thr_fpr)
    rep["by_length"] = group_table(ev, "len_bin", thr_fpr)
    rep["by_overlap_all"] = group_table(ev, "overlap_bin", thr_fpr)
    long_ev = ev[ev.n_words >= a.min_words]
    rep[f"by_overlap_min{a.min_words}words"] = group_table(long_ev, "overlap_bin", thr_fpr)

    # Each RAGTruth label type vs all supported sentences.
    neg = ev[ev.y == 0]
    rep["by_label_type"] = {}
    for lt in ["Evident Conflict", "Subtle Conflict", "Evident Baseless Info", "Subtle Baseless Info"]:
        pos = ev[ev.label_types.apply(lambda x: lt in x)]
        d = pd.concat([pos, neg])
        row = {"n": int(len(pos))}
        for m in METHODS:
            s, y = d[f"s_{m}"].values, d.y.values
            row[m] = {**ranking(s, y),
                      "recall_at_fpr10": at_threshold(pos[f"s_{m}"].values, pos.y.values, thr_fpr[m])["recall"]}
        rep["by_label_type"][lt] = row

    # Examples for the report: hallucinated, high overlap, cosine says fine, NLI flags it.
    ex = long_ev[(long_ev.y == 1) & (long_ev.overlap_best >= 0.7)
                 & (long_ev.s_cosine >= thr_fpr["cosine"]) & (long_ev.s_nli < thr_fpr["nli"])]
    ex = ex.sort_values("s_cosine", ascending=False).head(15)
    rep["examples_cosine_fooled_nli_catches"] = ex[[
        "example_id", "task_type", "label", "label_types", "sentence", "overlap_best",
        "cos_max", "nli_entail", "nli_contra"]].to_dict("records")

    json.dump(rep, open(out / "report.json", "w"), indent=2, default=str)

    # Short summary to print
    print(f"eval sentences: {len(ev)}, hallucinated: {ev.y.mean():.1%}")
    print("\nOVERALL            AUROC   AP     recall@10%FPR")
    for m in METHODS:
        o = rep["overall"][m]
        print(f"  {m:<15} {o['auroc']:.3f}  {o['avg_precision']:.3f}  {o['at_fpr10_threshold']['recall']:.3f}")
    for key in ["by_overlap_all", f"by_overlap_min{a.min_words}words", "by_length", "by_label_type"]:
        print(f"\n{key}  (AUROC per method; n / positives)")
        for g, row in rep[key].items():
            au = "  ".join(f"{m}={row[m]['auroc']:.3f}" if row[m]["auroc"] is not None else f"{m}=  -  "
                           for m in METHODS)
            print(f"  {g:<22} n={row['n']:<6} pos={row.get('positives', row['n']):<5} {au}")
    print(f"\n{len(rep['examples_cosine_fooled_nli_catches'])} example sentences saved in report.json")

    # Plot: AUROC by overlap, length-controlled
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tab = rep[f"by_overlap_min{a.min_words}words"]
    x = np.arange(len(OVERLAP_NAMES))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w = 0.27
    for i, m in enumerate(METHODS):
        vals = [tab[b][m]["auroc"] or 0 for b in OVERLAP_NAMES]
        ax.bar(x + (i - 1) * w, vals, w, label=m)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.text(len(x) - 0.5, 0.51, "chance", color="grey", fontsize=8, ha="right")
    ax.set_xticks(x, [f"{b}\n(n={tab[b]['n']}, pos={tab[b]['positives']})" for b in OVERLAP_NAMES], fontsize=8)
    ax.set_xlabel(f"Word overlap with closest source chunk (sentences with ≥{a.min_words} words)")
    ax.set_ylabel("AUROC (unsupported vs supported)")
    ax.set_ylim(0.3, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "auroc_by_overlap.png", dpi=200)
    print("saved", out / "auroc_by_overlap.png")


if __name__ == "__main__":
    main()
