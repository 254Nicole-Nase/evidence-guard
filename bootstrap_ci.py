"""Sprint work (14 Nov). 95% confidence intervals for AUROC, resampling whole
sources (all sentences from a resampled source come along together), because
sentences from the same source are not independent.

    python bootstrap_ci.py --scores results/scores.jsonl --sentences prepared/sentences.jsonl --out results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

METHODS = {"cosine": lambda d: d.cos_max, "nli": lambda d: d.nli_entail,
           "nli_e_minus_c": lambda d: d.nli_entail - d.nli_contra}


def auroc(y, s):
    return roc_auc_score(y, -s) if 0 < y.sum() < len(y) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--min_words", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_json(a.scores, lines=True, dtype={"source_id": str, "example_id": str})
    sent = pd.read_json(a.sentences, lines=True, dtype={"source_id": str, "example_id": str})
    df = df.merge(sent[["example_id", "sentence"]], on="example_id", how="left", validate="one_to_one")
    df = df[df.split == "eval"].reset_index(drop=True)
    df["n_words"] = df.sentence.str.split().str.len()
    for m, f in METHODS.items():
        df[f"s_{m}"] = f(df)

    long_ = df.n_words >= a.min_words
    ov = df.overlap_best
    neg = df.y == 0
    groups = {
        "overall": np.ones(len(df), bool),
        f"min{a.min_words}words": long_.values,
        f"min{a.min_words}w_overlap<50%": (long_ & (ov < .5)).values,
        f"min{a.min_words}w_overlap50-70%": (long_ & (ov >= .5) & (ov < .7)).values,
        f"min{a.min_words}w_overlap70-85%": (long_ & (ov >= .7) & (ov < .85)).values,
        f"min{a.min_words}w_overlap>=85%": (long_ & (ov >= .85)).values,
    }
    for lt in ["Evident Conflict", "Subtle Conflict", "Evident Baseless Info", "Subtle Baseless Info"]:
        groups[lt + " vs supported"] = (neg | df.label_types.apply(lambda x: lt in x)).values

    y = df.y.values
    S = {m: df[f"s_{m}"].values for m in METHODS}
    src = df.source_id.values
    uniq, inv = np.unique(src, return_inverse=True)
    rows_of = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(a.seed)

    point = {g: {m: auroc(y[mask], S[m][mask]) for m in METHODS} for g, mask in groups.items()}
    boots = {g: {m: [] for m in list(METHODS) + ["nli_minus_cosine"]} for g in groups}
    for b in range(a.n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[i] for i in pick])
        for g, mask in groups.items():
            ii = idx[mask[idx]]
            vals = {m: auroc(y[ii], S[m][ii]) for m in METHODS}
            for m, v in vals.items():
                boots[g][m].append(v)
            boots[g]["nli_minus_cosine"].append(vals["nli"] - vals["cosine"])
        if (b + 1) % 100 == 0:
            print(f"  {b + 1}/{a.n_boot}")

    res = {}
    for g in groups:
        res[g] = {"n": int(groups[g].sum()), "positives": int(y[groups[g]].sum())}
        for m, vals in boots[g].items():
            v = np.array(vals, float)
            v = v[~np.isnan(v)]
            est = point[g]["nli"] - point[g]["cosine"] if m == "nli_minus_cosine" else point[g][m]
            res[g][m] = {"auroc": float(est), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    json.dump({"n_boot": a.n_boot, "resampled_unit": "source", "results": res},
              open(Path(a.out) / "bootstrap_ci.json", "w"), indent=2)

    print(f"\nAUROC with 95% CI ({a.n_boot} source-level resamples)")
    for g, r in res.items():
        line = "  ".join(f"{m}={r[m]['auroc']:.3f} [{r[m]['ci95'][0]:.3f},{r[m]['ci95'][1]:.3f}]"
                         for m in ["cosine", "nli", "nli_minus_cosine"])
        print(f"{g:<34} n={r['n']:<6} pos={r['positives']:<5} {line}")


if __name__ == "__main__":
    main()
