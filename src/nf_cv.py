"""Track C (research): from-scratch neural forecasters (neuralforecast; fixed hyper-parameters, no Auto* tuning, seeded).

N-HiTS, TiDE and NBEATSx, global over all Mumbai series (src/nf_prep.py), 14-day input, 84-h horizon, MSE loss.
Known-future covariates: weather (lead-0), CAMS, hour sin/cos, fireworks flag; past-only: neighbours.
Per fold (train.FOLDS): trained once on data before the first origin, then rolling 23:00 origins (step 24 h) through
the fold with the true history as context (cross_validation, refit=False).
Writes data/processed/exp/nf_scratch/oof_tsfm.parquet (members nhits, tide, nbeatsx).
Runs in /dev/shm/pm25_work/venv_nf:  python src/nf_cv.py [max_steps]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
FOLDS = [("recent26", "2026-09-15", "2026-12-31"), ("fall25", "2025-10-01", "2025-12-01"),
         ("fall24", "2024-10-10", "2024-12-10")]
FUT = ["blh", "ws", "rh", "t2", "pr", "u", "v", "cams", "hs", "hc", "fw"]
HIST = ["nb_med", "nb_near"]
H, INP = 84, 336
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500


def models():
    from neuralforecast.losses.pytorch import MSE
    from neuralforecast.models import NBEATSx, NHITS, TiDE
    kw = dict(h=H, input_size=INP, futr_exog_list=FUT, hist_exog_list=HIST, loss=MSE(), max_steps=STEPS,
              learning_rate=1e-3, batch_size=32, windows_batch_size=256, scaler_type="robust", random_seed=0,
              enable_progress_bar=False, accelerator="gpu", devices=1)
    return [NHITS(**kw), TiDE(**kw), NBEATSx(**kw)]


def main():
    from neuralforecast import NeuralForecast
    torch.manual_seed(0)
    np.random.seed(0)
    L = pd.read_parquet(ROOT / "data" / "processed" / "nf" / "nf_long.parquet")
    meta = json.load(open(ROOT / "data" / "processed" / "nf" / "nf_targets.json"))
    targets, data_end = set(meta["targets"]), pd.Timestamp(meta["data_end"])
    outs = []
    for name, a, b in FOLDS:
        first = pd.Timestamp(a) + pd.Timedelta(hours=23)
        last = min(pd.Timestamp(b), data_end).normalize() + pd.Timedelta(hours=23)
        if last > data_end:
            last -= pd.Timedelta(days=1)
        end = last + pd.Timedelta(hours=H)
        n_win = int((last - first) / pd.Timedelta(hours=24)) + 1
        df = L[L.ds <= end].copy()
        # series need training history before the first origin
        st = df[df.y_obs.notna()].groupby("unique_id").ds.agg(["min", "max"])
        keep = st[(st["min"] <= first - pd.Timedelta(hours=INP + H)) & (st["max"] >= first - pd.Timedelta(days=30))].index
        df = df[df.unique_id.isin(keep)]
        parts = []
        for uid, g in df.groupby("unique_id"):  # pad every series to the common end (windows are end-aligned)
            g = g.set_index("ds").reindex(pd.date_range(g.ds.min(), end, freq="h"))
            g["unique_id"] = uid
            g[FUT + HIST] = g[FUT + HIST].interpolate(limit_direction="both")
            g["y"] = g.y_obs.interpolate(limit_direction="both")
            parts.append(g.rename_axis("ds").reset_index())
        df = pd.concat(parts, ignore_index=True)
        print(f"{name}: {df.unique_id.nunique()} series, origins {first} .. {last} ({n_win}), steps {STEPS}", flush=True)
        nf = NeuralForecast(models=models(), freq="h")
        cv = nf.cross_validation(df=df[["unique_id", "ds", "y"] + FUT + HIST], n_windows=n_win, step_size=24,
                                 refit=False)
        cv = cv.reset_index() if "unique_id" not in cv.columns else cv
        cv = cv[cv.unique_id.isin(targets)]
        yo = df[["unique_id", "ds", "y_obs"]]
        cv = cv.merge(yo, on=["unique_id", "ds"], how="left")
        o = pd.DataFrame({"sid": cv.unique_id, "ts0": cv.cutoff, "ts": cv.ds,
                          "h": ((cv.ds - cv.cutoff) / pd.Timedelta(hours=1)).astype(int),
                          "nhits": cv.NHITS.clip(lower=0), "tide": cv.TiDE.clip(lower=0),
                          "nbeatsx": cv.NBEATSx.clip(lower=0), "y": cv.y_obs, "fold": name})
        o = o[o.ts0.dt.hour == 23]
        r = {m: float(np.sqrt(np.nanmean((o.y - o[m]) ** 2))) for m in ("nhits", "tide", "nbeatsx")}
        print(name, "RMSE all leads", {k: round(v, 2) for k, v in r.items()}, flush=True)
        outs.append(o)
    O = pd.concat(outs, ignore_index=True)
    out = ROOT / "data" / "processed" / "exp" / "nf_scratch"
    out.mkdir(parents=True, exist_ok=True)
    O.to_parquet(out / "oof_tsfm.parquet")
    print("NF DONE", O.shape)


if __name__ == "__main__":
    main()
