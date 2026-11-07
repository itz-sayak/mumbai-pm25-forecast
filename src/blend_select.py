"""Blend selection from clean experiment OOFs (data/processed/exp/<exp>/).

python src/blend_select.py "<exp>:<model>,<exp>:<model>,..." [lo hi]
Prints per-component RMSE, leave-one-fold-out NNLS blend RMSE, and writes data/processed/blend_spec.json:
{"components": {"<col>": {"exp":..., "model":...}}, "weights": {...}, "lo":.., "hi":..}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "processed" / "exp"
KEY = ["sid", "ts0", "ts", "h"]


def load(spec):
    gt = None
    cols = {}
    frames = []
    for item in spec:
        exp, model = item.split(":")
        p = EXP / exp
        f = p / ("oof_gbm.parquet" if (p / "oof_gbm.parquet").exists() else "oof_tsfm.parquet")
        O = pd.read_parquet(f)
        col = f"{exp}__{model}"
        cols[col] = {"exp": exp, "model": model}
        if "y" in O.columns and gt is None:
            gt = O[KEY + ["y", "fold"]]
        frames.append(O[KEY + [model]].rename(columns={model: col}))
    if gt is None:
        gt = pd.read_parquet(EXP / "gbm_base" / "oof_gbm.parquet")[KEY + ["y", "fold"]]
    M = gt
    for F in frames:
        M = M.merge(F, on=KEY, how="inner")
    return M.dropna(), cols


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


SHRINK = 0.5  # block weights = SHRINK * per-block NNLS + (1 - SHRINK) * global NNLS (LOFO-selected)


def fit_blocks(df, names, block=3, shrink=None):
    """NNLS weights per block of target hour-of-day, shrunk toward the global NNLS weights
    (fixes the flattened diurnal cycle of a global blend without over-fitting any block)."""
    shrink = SHRINK if shrink is None else shrink
    wg, _ = nnls(df[names].values, df.y.values)
    out = {}
    for b in range(0, 24, block):
        m = (df.ts.dt.hour // block * block == b).values
        w, _ = nnls(df[names].values[m], df.y.values[m])
        out[b] = shrink * w + (1 - shrink) * wg
    return out


def apply_blocks(df, names, W, block=3):
    p = np.zeros(len(df))
    hb = (df.ts.dt.hour // block * block).values
    for b, w in W.items():
        m = hb == int(b)
        p[m] = df[names].values[m] @ np.asarray(w)
    return p


def main(spec, lo=4, hi=52, write=True, blocks=True):
    M, cols = load(spec)
    B = M[(M.h >= lo) & (M.h <= hi)]
    names = list(cols)
    print(f"rows {len(B)}  lead {lo}-{hi}")
    T = B.groupby("fold").apply(lambda g: pd.Series({c: rmse(g.y, g[c]) for c in names}))
    T.loc["mean"] = T.mean()
    print(T.T.round(2).to_string())
    lofo, lofo_b = {}, {}
    for f in B.fold.unique():
        tr, va = B[B.fold != f], B[B.fold == f]
        w, _ = nnls(tr[names].values, tr.y.values)
        lofo[f] = rmse(va.y, va[names].values @ w)
        lofo_b[f] = rmse(va.y, apply_blocks(va, names, fit_blocks(tr, names)))
    lofo["mean"] = float(np.mean(list(lofo.values())))
    lofo_b["mean"] = float(np.mean(list(lofo_b.values())))
    print("LOFO blend", {k: round(v, 2) for k, v in lofo.items()})
    print("LOFO blend (3h blocks)", {k: round(v, 2) for k, v in lofo_b.items()})
    w, _ = nnls(B[names].values, B.y.values)
    W = {c: float(x) for c, x in zip(names, w) if x > 1e-4}
    print("weights", {k: round(v, 3) for k, v in W.items()}, "sum", round(sum(W.values()), 3))
    use_blocks = blocks and lofo_b["mean"] < lofo["mean"]
    WB = {str(b): [float(x) for x in w] for b, w in fit_blocks(B, names).items()} if use_blocks else None
    if write:
        comps = {c: cols[c] for c in (names if use_blocks else W)}
        json.dump({"components": comps, "weights": W, "names": names, "blocks": WB, "lo": lo, "hi": hi,
                   "lofo": lofo, "lofo_blocks": lofo_b},
                  open(ROOT / "data" / "processed" / "blend_spec.json", "w"), indent=1)
    return (lofo_b if use_blocks else lofo), W


if __name__ == "__main__":
    spec = sys.argv[1].split(",")
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else 52
    main(spec, lo, hi)
