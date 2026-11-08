"""Out-of-sample check of a past run against the OpenAQ hours observed since its origin (validation only; nothing
here is written into a submission). Compares the final central forecast (`blend`, with the festival tilt when the
origin is 0-4 days after Diwali) with the plain NNLS blend (`blend_nnls`) and each member, by day and station.

python src/live_check.py <tag>      (reads data/processed/<tag>_detail.parquet; refresh openaq_live.py first)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def main(tag):
    P = pd.read_parquet(PROC / f"{tag}_detail.parquet")
    P = P[P.src == "T0"].copy()
    T0 = P.ts0.iloc[0]
    L = pd.read_parquet(ROOT / "data" / "external" / "openaq_live.parquet")
    L["sid"] = L.location_id.astype(int).astype(str)
    P["sid"] = P.sid.astype(str)
    M = P.merge(L[["sid", "ts", "y"]].rename(columns={"y": "obs"}), on=["sid", "ts"], how="inner")
    M = M[M.ts > T0].dropna(subset=["obs"])
    cols = [c for c in ["blend", "blend_nnls"] if c in M] + \
           [c for c in M.columns if "__" in c and M[c].notna().all()]
    print(f"origin {T0}  observed rows {len(M)}  last obs {M.ts.max()}")
    r = lambda g, c: np.sqrt(((g[c] - g.obs) ** 2).mean())
    out = pd.DataFrame({c: [r(M, c), M[c].mean() / M.obs.mean()] for c in cols}, index=["rmse", "mean p/obs"]).T
    print(out.round(3).to_string())
    by_day = M.groupby(M.ts.dt.date).apply(lambda g: pd.Series({"n": len(g), "obs": g.obs.mean(),
                                                                 **{c: r(g, c) for c in cols[:2]}}))
    print(by_day.round(2).to_string())
    print(M.groupby("sid").apply(lambda g: pd.Series({c: r(g, c) for c in cols[:2]})).round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
