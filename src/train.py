"""Global direct multi-horizon GBDT for hourly PM2.5 + backtests + test prediction.

python src/train.py cv                      # rolling backtests, prints RMSE table
python src/train.py predict <fcst_tag> <out.csv> [origin_ts]
"""
import json
import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import TARGETS, load_all, load_test, to_grid  # noqa: E402
from features import build_samples, load_cams, load_weather  # noqa: E402
from neighbors import build_signals  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
SEED = 42
USE_NB = True
EXCLUDE_PREFIX = []  # feature-name prefixes to drop (ablations), e.g. ["fw"]
SEASON = ["doy", "month", "dvl", "dvl_c", "dvl0"]  # season proxies: GBDT recalls past years' levels (2026 bias)
CAMS_ABS = ["cams", "cams24", "cams72_o", "cams_mos", "cams_mos24"]  # CAMS level drifts between years -> use ratios
REWEIGHT = False  # target stations x3, 2026 rows x2 (ablation gbm_rel_nb_w)
EXCLUDE_EXACT = SEASON + CAMS_ABS  # default = relative design (see experiments.csv: gbm_rel*)


def add_derived(S):
    """Scale-free CAMS signals: expected change vs the last 72 h (robust to CAMS level/bias drift)."""
    if "cams_chg" not in S.columns:
        S["cams_chg"] = S["cams24"] / np.maximum(S["cams72_o"], 1)
        S["cams_chg_h"] = S["cams"] / np.maximum(S["cams72_o"], 1)
    return S
H = list(range(1, 85))
DROP = {"sid", "t0", "t", "y", "ts", "ts0", "loc"}
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, seed=SEED, deterministic=True, num_threads=int(os.environ.get("LGB_THREADS", 8)))


def loc_map(d):
    m = d.groupby("sid").location_id.first()
    return lambda s: str(int(m[s])) if int(m[s]) in TARGETS else "-1"


def train_origins(ts, sid, y):
    hrs = ts.hour.values
    hours = [23, 8, 15, 20] if not sid.startswith(("x24", "oth")) else [23]
    pos = np.where(np.isin(hrs, hours))[0]
    cnt = pd.Series(~np.isnan(y)).rolling(24, min_periods=1).sum().values
    return pos[cnt[pos] >= 8]


def build_train(fcst_tag=None, cache=True):
    import features as _F
    import neighbors as _N
    f = PROC / (f"train_samples_nb{int(USE_NB)}_up{int(_N.USE_UP)}_cp{int(_N.USE_CP)}"
                f"_mw{int(_F.WX_MULTI)}{'_hole' if os.environ.get('DIWALI_HOLE') == '1' else ''}.parquet")
    if cache and f.exists() and fcst_tag is None:
        return pd.read_parquet(f)
    d = load_all()
    d = d[~d.sid.str.startswith("k18")]
    grid = to_grid(d)
    wx, cams = load_weather(fcst_tag), load_cams(fcst_tag)
    sids = [s for s in grid.columns]
    nb = build_signals(d, grid.index) if USE_NB else None
    S = build_samples(grid, sids, train_origins, H, wx, cams, loc_map(d), nb=nb)
    S = S[S.ts >= "2024-01-08"]
    if cache and fcst_tag is None:
        S.to_parquet(f)
    return S


def feats(S):
    return [c for c in S.columns if c not in DROP and not c.startswith("gbm_") and c != "pred"
            and not any(c.startswith(p) for p in EXCLUDE_PREFIX) and c not in EXCLUDE_EXACT]


def sample_weight(tr):
    if not REWEIGHT:
        return np.ones(len(tr))
    return (1.0 + 2.0 * (tr["loc"] != "-1").values) * (1.0 + 1.0 * (tr.ts.dt.year.values >= 2026))


def fit(tr, mode, rounds=None, va=None):
    X = tr[feats(tr)]
    sw = sample_weight(tr)
    if mode == "raw":
        y, w = tr.y.clip(upper=600), sw
    else:  # ratio target, weights lvl^2 -> equivalent to raw-scale squared error
        lv = np.maximum(tr.lvl, 5)
        y, w = (tr.y.clip(upper=600) / lv).clip(upper=8), lv ** 2 * sw
    ds = lgb.Dataset(X, y, weight=w, categorical_feature=["loc_cat"], free_raw_data=False)
    if va is not None and rounds is None:
        Xv = va[feats(va)]
        lv = np.maximum(va.lvl, 5)
        yv, wv = (va.y, None) if mode == "raw" else (va.y / lv, lv ** 2)
        dv = lgb.Dataset(Xv, yv, weight=wv, reference=ds)
        m = lgb.train(PARAMS, ds, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(150, verbose=False)])
    else:
        m = lgb.train(PARAMS, ds, rounds or 800)
    return m


def predict(m, S, mode):
    p = m.predict(S[feats(S)], num_iteration=m.best_iteration or None)
    if mode != "raw":
        p = p * np.maximum(S.lvl.values, 5)
    return np.clip(p, 1, None)


def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    k = ~np.isnan(a) & ~np.isnan(b)
    return float(np.sqrt(np.mean((a[k] - b[k]) ** 2)))


FOLDS = [("recent26", "2026-09-15", "2026-12-31"), ("fall25", "2025-10-01", "2025-12-01"),
         ("fall24", "2024-10-10", "2024-12-10")]
BUCKETS = {"pub1-48": (1, 48), "prv16-63": (16, 63), "all1-84": (1, 84)}


def cv(out_dir=None, clean=True):
    """clean=True: boosting rounds chosen on an inner window (30 d before the fold), never on the fold."""
    out_dir = Path(out_dir) if out_dir else PROC
    out_dir.mkdir(parents=True, exist_ok=True)
    S = add_derived(build_train())
    print("samples", S.shape, "features", len(feats(S)))
    res, oof = [], []
    for name, a, b in FOLDS:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        va = S[(S.ts0 >= a) & (S.ts0 < b) & (S.loc != "-1") & (S.hod0 == 23)]
        tr = S[(S.ts < a) | (S.ts0 >= b + pd.Timedelta(days=4))]
        if name == "recent26":
            tr = S[S.ts < a]
        if clean:
            ai = a - pd.Timedelta(days=30)
            tri = tr[(tr.ts < ai) | (tr.ts0 >= b + pd.Timedelta(days=4))]
            vai = tr[(tr.ts0 >= ai) & (tr.ts0 < a) & (tr.loc != "-1") & (tr.hod0 == 23) & (tr.ts < a)]
        out = va[["sid", "loc", "ts0", "ts", "h", "y"]].copy()
        for k in ("m24", "S7", "S3", "ew48", "lagsh", "cams_mos24"):
            out["b_" + k] = va[k].values
        for mode in ("raw", "ratio"):
            if clean:
                mi = fit(tri, mode, va=vai)
                n = max(50, int(mi.best_iteration * 1.1))
                m = fit(tr, mode, rounds=n)
                res.append((name, mode, n))
            else:
                m = fit(tr, mode, va=va)
                res.append((name, mode, m.best_iteration))
            out["gbm_" + mode] = predict(m, va, mode)
        out["fold"] = name
        oof.append(out)
        print(name, "done", [(r[1], r[2]) for r in res if r[0] == name])
    O = pd.concat(oof)
    O.to_parquet(out_dir / "oof_gbm.parquet")
    cols = [c for c in O.columns if c.startswith(("b_", "gbm_"))]
    O["gbm_avg"] = O[["gbm_raw", "gbm_ratio"]].mean(1)
    cols.append("gbm_avg")
    rows = []
    for (fold), g in O.groupby("fold"):
        for bk, (lo, hi) in BUCKETS.items():
            gg = g[(g.h >= lo) & (g.h <= hi)]
            rows.append({"fold": fold, "bucket": bk, **{c: rmse(gg.y, gg[c]) for c in cols}})
    T = pd.DataFrame(rows).round(2)
    print(T.to_string())
    json.dump(res, open(out_dir / "cv_iters.json", "w"))
    return T


def predict_test(fcst_tag, out_csv, origin=None, modes=("raw", "ratio"), rounds=None):
    d = load_all()
    d = d[~d.sid.str.startswith("k18")]
    tgt = d[d.src == "train"]
    T0 = pd.Timestamp(origin) if origin else tgt.ts.max()
    te = load_test()
    grid = to_grid(d[d.ts <= T0])
    full = pd.date_range(grid.index.min(), max(te.ts.max(), T0 + pd.Timedelta(hours=max(H))), freq="h")
    grid = grid.reindex(full)
    wx, cams = load_weather(fcst_tag), load_cams(fcst_tag)
    lm = loc_map(d)
    nb = build_signals(d[d.ts <= T0], full, upto=T0) if USE_NB else None
    t0pos = int(np.searchsorted(full, T0))
    sids = [str(k) for k in TARGETS]
    # main forecast from T0
    Sf = build_samples(grid, sids, lambda ts, s, y: np.array([t0pos]), H, wx, cams, lm, need_target=False, nb=nb)
    # back-fill rows (<= T0): forecasts issued previous day 23:00 (never copies observations)
    pos23 = np.where((full.hour == 23) & (full < T0) & (full >= te.ts.min() - pd.Timedelta(days=2)))[0]
    Sb = build_samples(grid, sids, lambda ts, s, y: pos23, list(range(1, 25)), wx, cams, lm, need_target=False, nb=nb)
    S = add_derived(build_train(cache=True))
    S = S[S.ts <= T0]  # strictly causal training set
    Sf, Sb = add_derived(Sf), add_derived(Sb)
    iters = {}
    try:
        for f, m, it in json.load(open(PROC / "cv_iters.json")):
            iters.setdefault(m, []).append(it)
    except FileNotFoundError:
        pass
    pf, pb = np.zeros(len(Sf)), np.zeros(len(Sb))
    seeds = (42, 7, 2026)
    for mode in modes:
        n = rounds or int(np.median(iters.get(mode, [800])) * 1.1)
        for sd in seeds:  # seed-averaged (fixed seeds -> reproducible)
            PARAMS["seed"] = sd
            m = fit(S, mode, rounds=n)
            a_, b_ = predict(m, Sf, mode) / len(seeds), predict(m, Sb, mode) / len(seeds)
            pf += a_ / len(modes)
            pb += b_ / len(modes)
            Sf["gbm_" + mode] = Sf.get("gbm_" + mode, 0) + a_
            Sb["gbm_" + mode] = Sb.get("gbm_" + mode, 0) + b_
    PARAMS["seed"] = SEED
    Sf["pred"], Sb["pred"] = pf, pb
    keep = ["sid", "ts0", "ts", "h", "pred", "cams_mos24", "lvl"] + ["gbm_" + m for m in modes]
    P = pd.concat([Sf[keep].assign(src="T0"), Sb[keep].assign(src="bk")])
    sub = te[["ID", "location_id", "ts"]].copy()
    sub["sid"] = sub.location_id.astype(str)
    a = Sf[["sid", "ts", "pred"]]
    b = Sb[Sb.ts <= T0][["sid", "ts", "pred"]].drop_duplicates(["sid", "ts"])
    sub = sub.merge(a.rename(columns={"pred": "pf"}), on=["sid", "ts"], how="left")
    sub = sub.merge(b.rename(columns={"pred": "pb"}), on=["sid", "ts"], how="left")
    sub["Target"] = np.where(sub.ts > T0, sub.pf, sub.pb)
    # beyond horizon: same hour-of-day from the last forecast day
    if sub.Target.isna().any():
        last = Sf[Sf.h > max(H) - 24].copy()
        last["hod"] = last.ts.dt.hour
        fill = last.groupby(["sid", "hod"]).pred.mean()
        miss = sub.Target.isna()
        sub.loc[miss, "Target"] = [fill.get((s, t.hour), np.nan) for s, t in zip(sub.sid[miss], sub.ts[miss])]
    med = tgt.groupby("sid").y.median()
    sub["Target"] = sub.Target.fillna(sub.sid.map(med)).clip(lower=0)
    check_sub(sub[["ID", "Target"]])
    sub[["ID", "Target"]].to_csv(out_csv, index=False)
    P.to_parquet(PROC / (Path(out_csv).stem + "_detail.parquet"))
    print("origin", T0, "wrote", out_csv, sub.Target.describe().round(1).to_dict())
    return sub


def check_sub(sub):
    ss = pd.read_csv(ROOT / "data" / "raw" / "SampleSubmission.csv")
    assert len(sub) == len(ss) == 2592, len(sub)
    assert (sub.ID.values == ss.ID.values).all(), "ID order mismatch"
    assert np.isfinite(sub.Target).all() and (sub.Target >= 0).all()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cv":
        cv()
    elif cmd == "predict":
        predict_test(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
