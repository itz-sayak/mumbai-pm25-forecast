"""Local replica of the public-leaderboard scorer: RMSE of a submission against the OpenAQ hourly observations
(the same upstream source as Train.csv; data/external/openaq_live.parquet) on a 2-day public window.
Calibrated on 7 Nov 2026: reproduces the 3-4 Nov public scores of 7 submissions within 0.003 (92 scored hours,
as announced). Uses only past observations to SCORE files; nothing here is ever written into a submission.

python src/local_score.py <sub.csv> [<sub.csv> ...] [--windows 2026-11-03,2026-11-05]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_test  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "external" / "openaq_live.parquet"
WINDOWS = ["2026-11-03", "2026-11-05"]  # 5 Nov update: public 3-4 Nov; 9 Nov update: public 5-6 Nov


def truth(start):
    """Observed targets for the 2-day window starting at `start` (hour-end labels, IST)."""
    te = load_test()
    L = pd.read_parquet(LIVE)
    t0 = pd.Timestamp(start)
    w = te[(te.ts > t0) & (te.ts <= t0 + pd.Timedelta(days=2))].copy()
    w["sid"] = w.location_id.astype(str)
    L["sid"] = L.location_id.astype(str)
    return w.merge(L[["sid", "ts", "y"]], on=["sid", "ts"], how="inner")[["ID", "sid", "ts", "y"]]


def score(sub, start):
    s = pd.read_csv(sub) if not isinstance(sub, pd.DataFrame) else sub
    m = truth(start).merge(s[["ID", "Target"]], on="ID")
    return float(np.sqrt(((m.Target - m.y) ** 2).mean())), len(m)


if __name__ == "__main__":
    args = sys.argv[1:]
    wins = WINDOWS
    if "--windows" in args:
        i = args.index("--windows")
        wins = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    for f in args:
        print(Path(f).name, "  ".join(f"{w[5:]}+2d: {r:.3f} (n={n})" for w in wins for r, n in [score(f, w)]))
