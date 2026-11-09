"""Track C (research): long-format training data for from-scratch neuralforecast models (src/nf_cv.py, venv_nf).

Same series and covariates as the fine-tuned Chronos-2 (tsfm.Exo.full): every Mumbai series except the 2018-20 MPCB
history, lead-0 weather, CAMS, hour sin/cos, fireworks flag (known future) and nb_med / nb_near (past only).
Writes data/processed/nf/nf_long.parquet and nf_targets.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402
import experiments as E  # noqa: E402
import tsfm as TS  # noqa: E402
from neighbors import build_signals  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "nf"


def main():
    E.set_flags("tsfm_fw_nb")
    TS.USE_FW = TS.USE_NB = True
    d = D.load_all()
    d = d[~d.sid.str.startswith("k18")]
    grid = D.to_grid(d)
    grid = grid.reindex(pd.date_range(grid.index.min(), grid.index.max() + pd.Timedelta(hours=TS.HMAX), freq="h"))
    exo = TS.Exo(grid.index, nb=build_signals(d, grid.index))
    locs = d.groupby("sid").location_id.first()
    parts, targets = [], []
    for sid, loc in TS.all_sid_locs(d, grid):
        y = grid[sid]
        v = y.dropna()
        if len(v) < 24 * 21:
            continue
        cov = exo.full(loc)
        df = pd.DataFrame({"unique_id": sid, "ds": grid.index, "y_obs": y.values.astype(np.float32),
                           **{k: v_.astype(np.float32) for k, v_ in cov.items()}})
        parts.append(df[df.ds >= v.index[0]])
        if int(locs[sid]) in D.TARGETS:
            targets.append(sid)
    L = pd.concat(parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    L.to_parquet(OUT / "nf_long.parquet")
    json.dump({"targets": targets, "data_end": str(d[d.src == "train"].ts.max())}, open(OUT / "nf_targets.json", "w"))
    print(L.shape, L.unique_id.nunique(), "series;", len(targets), "target series;", list(L.columns))


if __name__ == "__main__":
    main()
