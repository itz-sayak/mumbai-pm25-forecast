"""Blend OOF predictions: NNLS weights per horizon bucket, leave-one-fold-out evaluation."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def load_oof():
    g = pd.read_parquet(PROC / "oof_gbm.parquet")
    t = pd.read_parquet(PROC / "oof_tsfm.parquet")
    O = g.merge(t.drop(columns=["fold"]), on=["sid", "ts0", "ts", "h"], how="left")
    return O


def rmse(a, b):
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def fit_w(df, cols):
    X = df[cols].values
    w, _ = nnls(X, df.y.values)
    return w


def main(cols=None, lo=16, hi=63):
    O = load_oof()
    base = [c for c in O.columns if c.startswith(("b_", "gbm_", "chronos2", "timesfm3", "timestral", "pibm"))]
    O = O.dropna(subset=base)
    cols = cols or base
    B = O[(O.h >= lo) & (O.h <= hi)]
    print("rows", len(B), "folds", B.fold.value_counts().to_dict())
    rows = []
    for f, g in B.groupby("fold"):
        rows.append({"fold": f, **{c: rmse(g.y, g[c]) for c in base}})
    T = pd.DataFrame(rows).set_index("fold")
    T.loc["mean"] = T.mean()
    print(T.round(2).T.to_string())
    # leave-one-fold-out NNLS
    res = {}
    for f in B.fold.unique():
        tr, va = B[B.fold != f], B[B.fold == f]
        w = fit_w(tr, cols)
        res[f] = (rmse(va.y, va[cols].values @ w), dict(zip(cols, w.round(3))))
    for f, (r, w) in res.items():
        print("LOFO", f, round(r, 2), {k: v for k, v in w.items() if v > 0})
    w = fit_w(B, cols)
    W = {k: float(v) for k, v in zip(cols, w) if v > 0}
    print("ALL-fold weights", W, "sum", round(sum(W.values()), 3))
    json.dump(W, open(PROC / "blend_weights.json", "w"), indent=1)
    return W


if __name__ == "__main__":
    cols = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else None
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else 63
    main(cols, lo, hi)
