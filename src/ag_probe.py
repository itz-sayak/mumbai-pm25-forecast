"""RESEARCH ONLY (W5) - never imported by the submission pipeline. AutoML is not permitted in the solution, so
AutoGluon-TimeSeries is used offline as a probe: which model families work on our folds? Anything adopted is
re-implemented by hand with fixed settings and validated through src/experiments.py.

Three steps, two environments:
  python src/ag_probe.py prep                         (main venv)  same items/covariates/origins as tsfm.cv
  python src/ag_probe.py fit <fold> [time_limit_s]    (venv_ag: autogluon.timeseries; no imports from src/)
  python src/ag_probe.py collect                      (main venv)  -> data/processed/exp/ag_probe/oof_tsfm.parquet
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "exp" / "ag_probe"
H = 51          # private-window lead range is 3-51 from the final origin
CTX = 2048      # context given to every prediction item
TRAIN_DAYS = 730


def prep():
    sys.path.insert(0, str(ROOT / "src"))
    import tsfm as TS
    from data import TARGETS, load_all, to_grid
    from neighbors import build_signals
    from train import FOLDS
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_all()
    d = d[~d.sid.str.startswith("k18")]
    grid = to_grid(d)
    grid = grid.reindex(pd.date_range(grid.index.min(), grid.index.max() + pd.Timedelta(hours=TS.HMAX), freq="h"))
    TS.CTX, TS.USE_FW, TS.USE_NB = CTX, True, True
    exo = TS.Exo(grid.index, nb=build_signals(d, grid.index))
    locs = d.groupby("sid").location_id.first()
    tsids = [(s, str(int(locs[s]))) for s in grid.columns if int(locs[s]) in TARGETS]
    meta = {}
    for name, a, b in FOLDS:
        cut = int(np.searchsorted(grid.index, pd.Timestamp(a)))
        lo = max(0, cut - TRAIN_DAYS * 24)
        # training frame: every Mumbai series up to the fold start, lead-0 covariates (as in chronos_finetune)
        rows = []
        for sid, loc in TS.all_sid_locs(d, grid):
            y = grid[sid].values[lo:cut]
            if np.isfinite(y).sum() < 24 * 21:
                continue
            f = pd.DataFrame({k: v[lo:cut] for k, v in exo.full(loc).items()})
            f.insert(0, "target", y)
            f.insert(0, "timestamp", grid.index[lo:cut])
            f.insert(0, "item_id", sid)
            rows.append(f)
        tr = pd.concat(rows, ignore_index=True)
        known = [c for c in tr.columns if c not in ("item_id", "timestamp", "target") and not c.startswith(("nb_", "cp_"))]
        past = [c for c in tr.columns if c.startswith(("nb_", "cp_"))]
        # prediction items: one per (target station, 23:00 origin), exactly the tsfm.cv origins
        org = np.where((grid.index >= a) & (grid.index < b) & (grid.index.hour == 23))[0]
        items = TS.build_items(grid, exo, tsids, org)
        ctx_rows, fut_rows = [], []
        for it in items:
            iid = f"{it['sid']}|{grid.index[it['t0']]:%Y-%m-%d %H:%M}"
            ts = grid.index[it["t0"] - CTX + 1:it["t0"] + 1]
            c = pd.DataFrame({"item_id": iid, "timestamp": ts, "target": it["y"]})
            for k in known:
                c[k] = it["cp"][k] if k in it["cp"] else 0.0
            for k in past:
                c[k] = it["po"].get(k, np.zeros(CTX))
            ctx_rows.append(c)
            ft = grid.index[it["t0"] + 1:it["t0"] + 1 + H]
            fu = pd.DataFrame({"item_id": iid, "timestamp": ft})
            for k in known:
                fu[k] = it["cf"][k][:H] if k in it["cf"] else 0.0
            fut_rows.append(fu)
        tr.to_parquet(OUT / f"{name}_train.parquet")
        pd.concat(ctx_rows, ignore_index=True).to_parquet(OUT / f"{name}_ctx.parquet")
        pd.concat(fut_rows, ignore_index=True).to_parquet(OUT / f"{name}_fut.parquet")
        meta[name] = {"known": known, "past": past, "items": len(items), "train_rows": len(tr)}
        print(name, meta[name]["items"], "items", len(tr), "train rows", len(known), "known", len(past), "past", flush=True)
    json.dump(meta, open(OUT / "meta.json", "w"), indent=1)


def fit(fold, time_limit=1800):
    import torch
    # probe-only workaround: Lightning queries the legacy matmul-precision API after AutoGluon set the new one
    torch.get_float32_matmul_precision = lambda: "highest"
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    meta = json.load(open(OUT / "meta.json"))[fold]
    tr = TimeSeriesDataFrame.from_data_frame(pd.read_parquet(OUT / f"{fold}_train.parquet"))
    ctx = TimeSeriesDataFrame.from_data_frame(pd.read_parquet(OUT / f"{fold}_ctx.parquet"))
    fut = TimeSeriesDataFrame.from_data_frame(pd.read_parquet(OUT / f"{fold}_fut.parquet"))
    pred = TimeSeriesPredictor(prediction_length=H, target="target", known_covariates_names=meta["known"],
                               eval_metric="RMSE", freq="h", path=str(OUT / f"ag_{fold}"), verbosity=2)
    pred.fit(tr, presets="best_quality", time_limit=int(time_limit), random_seed=0)
    lb = pred.leaderboard(silent=True)
    lb.to_csv(OUT / f"{fold}_leaderboard.csv", index=False)
    try:
        w = pred._trainer.load_model(pred.model_best).model_to_weight  # ensemble weights, if the best is an ensemble
        json.dump({k: float(v) for k, v in w.items()}, open(OUT / f"{fold}_ens_weights.json", "w"), indent=1)
    except Exception as e:  # noqa
        print("no ensemble weights:", repr(e)[:200])
    outs = []
    for m in lb.model.tolist():
        try:
            p = pred.predict(ctx, known_covariates=fut, model=m)
        except Exception as e:  # noqa
            print(m, "predict FAILED", repr(e)[:200], flush=True)
            continue
        p = p["mean"].rename(f"ag_{m}").reset_index()
        outs.append(p.set_index(["item_id", "timestamp"]))
        print(m, "ok", flush=True)
    P = pd.concat(outs, axis=1).reset_index()
    P.to_parquet(OUT / f"{fold}_pred.parquet")
    print(lb.to_string())


def collect():
    frames = []
    for f in sorted(OUT.glob("*_pred.parquet")):
        fold = f.name.replace("_pred.parquet", "")
        P = pd.read_parquet(f)
        sid_ts0 = P.item_id.str.split("|", expand=True)
        P["sid"], P["ts0"] = sid_ts0[0], pd.to_datetime(sid_ts0[1])
        P["ts"] = pd.to_datetime(P.timestamp)
        P["h"] = ((P.ts - P.ts0) / pd.Timedelta(hours=1)).astype(int)
        P["fold"] = fold
        frames.append(P.drop(columns=["item_id", "timestamp"]))
    O = pd.concat(frames, ignore_index=True)
    cols = [c for c in O.columns if c.startswith("ag_")]
    O[cols] = O[cols].clip(lower=0)
    O.to_parquet(OUT / "oof_tsfm.parquet")
    print(O.shape, cols)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
    elif cmd == "fit":
        fit(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 1800)
    elif cmd == "collect":
        collect()
