"""Zero-shot time-series foundation models with known-future covariates.

* Chronos-2 (amazon/chronos-2): target + past/future covariates (dict API), mean forecast.
* TimesFM-3.0 (google/timesfm-3.0-pytorch): context + past_future covariates.

python src/tsfm.py cv                    -> data/processed/oof_tsfm.parquet  (same origins as GBDT cv)
"""
import os
import random
import sys
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (code-review reproducibility)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import DIWALI, TARGETS, fireworks_flag, load_all, to_grid  # noqa: E402
from features import load_cams, load_weather  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CTX = 1024
USE_FW = True
USE_NB = True
MASK_FW = False  # "_mfw": festival-smoke hours -> missing in the target context (models carry the spike forward)
NO_CAMS = False  # "_nocams": no CAMS covariate (fine-tune and inference), so pre-2022 history can be used
USE_K18 = False  # "_k18": fine-tune also on the 2018-19 MPCB history (2 extra pre-COVID Diwalis; src/k18.py)
FT_PIPE = None  # set to a fine-tuned Chronos2Pipeline to use it in chronos_forecast
HMAX = 84
COV = {"blh": "boundary_layer_height", "ws": "wind_speed_10m", "rh": "relative_humidity_2m",
       "t2": "temperature_2m", "pr": "precipitation", "wd": "wind_direction_10m"}
torch.manual_seed(0)
np.random.seed(0)


def smoke_mask(ts):
    """Festival-smoke hours: 20:00-08:00 IST (hour-end labels) of the nights Diwali-2 .. Diwali+2.
    The observed spike peaks at 02-04 h and lasts into the morning inversion (2024, 2025 grids)."""
    ts = pd.DatetimeIndex(ts)
    st = ts - pd.Timedelta(hours=1)
    night = (st - pd.Timedelta(hours=8)).normalize()  # 20:00..07:59 belongs to the evening's date
    is_n = (st.hour >= 20) | (st.hour < 8)
    off = np.min(np.abs(np.stack([(night - D).days for D in DIWALI])), axis=0)
    return np.asarray(is_n & (off <= 2))


class Exo:
    def __init__(self, index, fcst_tag=None, nb=None):
        wx, cams = load_weather(fcst_tag), load_cams(fcst_tag)
        self.nb = nb  # dict loc -> DataFrame[nb_med, nb_near] aligned to index (past-only use)
        self.wx = {L: wx[L].reindex(index) for L in wx}
        self.cams = cams.reindex(index)
        self.index = index
        self.fw = fireworks_flag(index)
        self.smoke = smoke_mask(index)

    def full(self, loc):
        """Full-length lead-0 covariates (for fine-tuning)."""
        wl = loc if (("wind_speed_10m", loc) in self.wx[0].columns) else "city"
        d = {k: self.wx[0][(v, wl)].values.astype(float) for k, v in COV.items()}
        wd = np.deg2rad(d.pop("wd"))
        d["u"] = -d["ws"] * np.sin(wd)
        d["v"] = -d["ws"] * np.cos(wd)
        cl = loc if loc in self.cams.columns else "city"
        if not NO_CAMS:
            d["cams"] = self.cams[cl].values.astype(float)
        for sv in ("ws_sd", "pr_sd"):
            if (sv, wl) in self.wx[0].columns:
                d[sv] = self.wx[0][(sv, wl)].values.astype(float)
        hrs = self.index.hour.values
        d["hs"] = np.sin(2 * np.pi * hrs / 24)
        d["hc"] = np.cos(2 * np.pi * hrs / 24)
        if USE_FW:
            d["fw"] = self.fw.astype(float)
        if self.nb is not None:
            src = self.nb.get(loc, self.nb["city"])
            for k in src.columns:
                d[k] = src[k].values.astype(float)
        return {k: pd.Series(v).interpolate(limit_direction="both").fillna(0).values.astype(np.float32)
                for k, v in d.items()}

    def covs(self, loc, t0, H):
        """Past covariates [t0-CTX+1 .. t0] from lead-0; future [t0+1 .. t0+H] lead-matched."""
        wl = loc if (("wind_speed_10m", loc) in self.wx[0].columns) else "city"
        past = slice(t0 - CTX + 1, t0 + 1)
        fut = np.arange(t0 + 1, t0 + H + 1)
        lead = np.minimum(3, np.maximum(1, np.ceil((fut - t0) / 24))).astype(int)
        out_p, out_f = {}, {}
        for k, v in COV.items():
            a0 = self.wx[0][(v, wl)].values.astype(float)
            out_p[k] = a0[past]
            f = np.empty(H)
            for L in (1, 2, 3):
                aL = self.wx[L][(v, wl)].values.astype(float)
                m = lead == L
                f[m] = aL[fut[m]]
            out_f[k] = f
        for sv in ("ws_sd", "pr_sd"):  # multi-NWP spread (forecast uncertainty), when available
            if (sv, wl) in self.wx[0].columns:
                out_p[sv] = self.wx[0][(sv, wl)].values.astype(float)[past]
                f = np.empty(H)
                for L in (1, 2, 3):
                    m = lead == L
                    f[m] = self.wx[L][(sv, wl)].values.astype(float)[fut[m]]
                out_f[sv] = f
        for d in (out_p, out_f):
            wd = np.deg2rad(d.pop("wd"))
            d["u"] = -d["ws"] * np.sin(wd)
            d["v"] = -d["ws"] * np.cos(wd)
        cl = loc if loc in self.cams.columns else "city"
        if not NO_CAMS:
            c = self.cams[cl].values.astype(float)
            out_p["cams"], out_f["cams"] = c[past], c[fut]
        hrs = self.index.hour.values
        for d, ix in ((out_p, np.arange(t0 - CTX + 1, t0 + 1)), (out_f, fut)):
            d["hs"] = np.sin(2 * np.pi * hrs[ix] / 24)
            d["hc"] = np.cos(2 * np.pi * hrs[ix] / 24)
            if USE_FW:
                d["fw"] = self.fw[ix].astype(float)
        out_po = {}
        if self.nb is not None:
            src = self.nb.get(loc, self.nb["city"])
            for k in src.columns:
                out_po[k] = src[k].values.astype(float)[past]
        for d in (out_p, out_f, out_po):
            for k in d:
                s = pd.Series(d[k]).interpolate(limit_direction="forward").bfill().fillna(0).values
                d[k] = s.astype(np.float32)
        return out_p, out_f, out_po


def chronos_forecast(items, H, use_cov=True):
    from chronos import Chronos2Pipeline
    pipe = FT_PIPE or Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    inputs = []
    for it in items:
        d = {"target": it["y"].astype(np.float32)}
        if use_cov:
            d["past_covariates"] = {**it["cp"], **it.get("po", {})}  # po = past-only (neighbours)
            d["future_covariates"] = {k: v for k, v in it["cf"].items()}
        inputs.append(d)
    q, mean = pipe.predict_quantiles(inputs, prediction_length=H, quantile_levels=[0.1, 0.5, 0.9],
                                     batch_size=64 if CTX <= 1024 else 16)
    return np.stack([m[0].float().cpu().numpy() for m in mean])


def timesfm_forecast(items, H, use_cov=True):
    from timesfm3 import TimesFM3Forecaster
    m = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cuda")
    ctxs, covs = [], []
    for it in items:
        y = pd.Series(it["y"]).interpolate(limit_direction="both").values.astype(np.float32)
        ctxs.append(y)
        if use_cov:
            covs.append(np.stack([np.r_[it["cp"][k], it["cf"][k]] for k in sorted(it["cp"])]).astype(np.float32))
    po = [np.stack([it["po"][k] for k in sorted(it["po"])]).astype(np.float32) for it in items]         if use_cov and items and items[0].get("po") else None
    outs = m.predict_batch(ctxs, H, past_future_covariates=covs if use_cov else None, past_only_covariates=po,
                           make_positive=True, return_quantiles=True)
    res = []
    for o in outs:
        q = None if o.quantiles is None else np.asarray(o.quantiles, dtype=float)
        if q is not None and q.ndim == 2:  # mean ~ average of quantiles (RMSE-optimal point)
            q = q if q.shape[0] == H or q.shape[0] >= H else q.T
            p = q[:H].mean(1) if q.shape[0] >= H else np.asarray(o.forecast, dtype=float)
        else:
            p = np.asarray(o.forecast, dtype=float)
        res.append(np.asarray(p, dtype=float).reshape(-1)[:H])
    return np.stack(res)


def timestral_forecast(items, H, use_cov=True, repo="mistralai/Timestral-209M", bs=None):
    from timestral import TimestralForecaster
    nvar = 1 + len(items[0]["cp"]) + len(items[0].get("po", {})) if items else 1
    bs = bs or max(4, (64 if nvar <= 20 else 16) * 1024 // max(CTX, 1024) // (2 if CTX > 1024 else 1))  # 8 GB GPU
    m = TimestralForecaster.from_pretrained(repo, device="cuda")
    res = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        ctx = np.stack([it["y"] for it in chunk]).astype(np.float32)[:, None, :]  # NaN = missing
        kw = {}
        if use_cov:
            kn = np.stack([np.stack([np.r_[it["cp"][k], it["cf"][k]] for k in sorted(it["cp"])]) for it in chunk])
            kw["known_covariates"] = kn.astype(np.float32)
            if chunk[0].get("po"):
                pc = np.stack([np.stack([it["po"][k] for k in sorted(it["po"])]) for it in chunk])
                kw["past_covariates"] = pc.astype(np.float32)
        f = m.forecast(ctx, prediction_length=H, **kw)
        q = f.quantiles
        q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
        res.append(q[:, 0, :H, :].mean(-1))  # mean of 21 quantiles ~ RMSE-optimal point
    return np.concatenate(res)


def _ffill_rows(a):
    """Forward/back-fill NaN along the last axis, then the row mean (t0 needs finite context)."""
    a = pd.DataFrame(a.reshape(-1, a.shape[-1])).T.ffill().bfill().T.values.reshape(a.shape)
    m = np.nanmean(np.where(np.isfinite(a), a, np.nan), axis=-1, keepdims=True)
    return np.where(np.isfinite(a), a, np.nan_to_num(m)).astype(np.float32)


T0_QL = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
T0_SYM = False  # variant: t0's symmetric averaging (published-benchmark setting), target-only context


def t0_forecast(items, H, use_cov=True, repo="theforecastingcompany/t0-gamma", bs=None):
    """t0-gamma (The Forecasting Company, Apache-2.0): target (+ past-only signals as extra variates) in the context,
    known weather/CAMS/fireworks covariates over context+horizon as future_covariates; mean of quantiles."""
    import torch
    from t0 import T0Forecaster
    m = T0Forecaster.from_pretrained(repo).eval().to("cuda")
    bs = bs or max(4, 64 * 1024 // max(CTX, 1024))
    res = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        y = np.stack([it["y"] for it in chunk]).astype(np.float32)
        kw = {}
        ctx = y
        if use_cov:
            kn = np.stack([np.stack([np.r_[it["cp"][k], it["cf"][k]] for k in sorted(it["cp"])]) for it in chunk])
            kw["future_covariates"] = torch.tensor(_ffill_rows(kn), device="cuda")
            if chunk[0].get("po") and not T0_SYM:
                pc = np.stack([np.stack([it["po"][k] for k in sorted(it["po"])]) for it in chunk])
                ctx = np.concatenate([y[:, None, :], pc.astype(np.float32)], 1)  # target first, then past-only variates
        with torch.no_grad():  # NaN gaps in the context are handled natively; covariates are gap-filled
            out = m.predict(torch.tensor(ctx, device="cuda"), horizon=H, quantile_levels=T0_QL,
                            clip_nonnegative=True, symmetric_averaging=T0_SYM, **kw)
        q = out.quantiles
        q = q.detach().float().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
        if q.ndim == 4:  # (B, V, H, Q) for multivariate context -> target variate
            q = q[:, 0]
        res.append(q[:, :H, :].mean(-1))
    return np.concatenate(res)


TIMEH2O_PY = os.environ.get("TIMEH2O_PY", "/dev/shm/pm25_work/venv312/bin/python")


def timeh2o_forecast(items, H, use_cov=True):
    """TimeH2O-v1 (H2O.ai, Oct 2026; fev-bench #1) through a worker in the Python 3.12 env (src/timeh2o_worker.py).
    Known-future weather/CAMS/fireworks covariates over context+horizon; v1 ignores past-only covariates."""
    import subprocess
    import tempfile
    y = np.stack([it["y"] for it in items]).astype(np.float32)
    kw = {"y": y, "H": np.array(H)}
    if use_cov:
        kw["fc"] = _ffill_rows(np.stack([np.stack([np.r_[it["cp"][k], it["cf"][k]] for k in sorted(it["cp"])])
                                         for it in items]))
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.npz"), os.path.join(td, "out.npy")
        np.savez(fi, **kw)
        subprocess.run([TIMEH2O_PY, str(ROOT / "src" / "timeh2o_worker.py"), fi, fo], check=True)
        return np.load(fo)[:, :H]


TABPFN_PY = os.environ.get("TABPFN_PY", "/dev/shm/pm25_work/venv_pfn/bin/python")
TABPFN_CKPT = None  # "_ts3": TabPFN-TS-3, the checkpoint fine-tuned for forecasting (default is TabPFN-3.5)
TS3_CKPT = "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"


def tabpfn_forecast(items, H, use_cov=True):
    """TabPFN-TS (Prior Labs; tabular in-context regression, TabPFN-3.5, LOCAL mode) through a worker in its own
    env (src/tabpfn_worker.py). Known-future weather/CAMS/fireworks covariates + calendar features from timestamps;
    TabPFN-TS ignores past-only covariates."""
    import subprocess
    import tempfile
    names = sorted(items[0]["cp"]) if use_cov else []
    y = np.stack([it["y"] for it in items]).astype(np.float32)
    kc = np.stack([np.stack([it["cp"][k] for k in names]) for it in items]).astype(np.float32) if names else \
        np.zeros((len(items), 0, y.shape[1]), np.float32)
    kf = np.stack([np.stack([it["cf"][k][:H] for k in names]) for it in items]).astype(np.float32) if names else \
        np.zeros((len(items), 0, H), np.float32)
    ts0 = np.array([pd.Timestamp(it["ts0"]).value for it in items], dtype=np.int64)
    home = os.environ.get("TABPFN_HOME", "/dev/shm/pm25_work/home")  # tabpfn caches its token under ~/.cache
    os.makedirs(home, exist_ok=True)
    env = {**os.environ, "HOME": home, **({"TABPFN_CKPT": TABPFN_CKPT} if TABPFN_CKPT else {}),
           "SKRUB_DATA_DIRECTORY": os.environ.get("SKRUB_DATA_DIRECTORY", "/dev/shm/pm25_work/skrub")}
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.npz"), os.path.join(td, "out.npy")
        np.savez(fi, y=y, kc=kc, kf=kf, ts0=ts0, names=np.array(names, dtype=object))
        subprocess.run([TABPFN_PY, str(ROOT / "src" / "tabpfn_worker.py"), fi, fo], check=True, env=env)
        return np.load(fo)[:, :H]


def build_items(grid, exo, sid_locs, origins):
    items = []
    for sid, loc in sid_locs:
        y = grid[sid].values.astype(float)
        for t0 in origins:
            if t0 - CTX + 1 < 0 or t0 + HMAX >= len(grid) or np.isnan(y[t0 - 23:t0 + 1]).sum() > 16:
                continue  # (t0 + HMAX past the grid: an origin after the newest reading, as with live data in cv)
            cp, cf, po = exo.covs(loc, t0, HMAX)
            yc = y[t0 - CTX + 1:t0 + 1].copy()
            if MASK_FW:
                m = exo.smoke[t0 - CTX + 1:t0 + 1]
                if m.any():
                    yc[m] = np.nan  # target: missing (Chronos-2 / Timestral take NaN context)
                    for k in po:    # neighbour context: bridged linearly over the smoke hours
                        s = po[k].astype(float).copy()
                        s[m] = np.nan
                        po[k] = pd.Series(s).interpolate(limit_direction="both").values.astype(np.float32)
            items.append({"sid": sid, "loc": loc, "t0": t0, "ts0": grid.index[t0], "y": yc, "cp": cp, "cf": cf,
                          "po": po})
    return items


def to_long(items, P, grid, name):
    rows = []
    for it, p in zip(items, P):
        ix = np.arange(it["t0"] + 1, it["t0"] + 1 + HMAX)
        ok = ix < len(grid)
        rows.append(pd.DataFrame({"sid": it["sid"], "ts0": grid.index[it["t0"]], "ts": grid.index[ix[ok]],
                                  "h": np.arange(1, HMAX + 1)[ok], name: p[ok]}))
    return pd.concat(rows, ignore_index=True)


FT_STEPS, FT_LR, FT_SEED = 800, 1e-5, 0  # set per experiment (experiments.set_flags)


def chronos_finetune(grid, exo, sid_locs, cutoff_pos, steps=None, lr=None, H=HMAX, seed=None):
    """Fine-tune Chronos-2 on all Mumbai series using only data strictly before cutoff_pos."""
    from chronos import Chronos2Pipeline
    from chronos.chronos2.preprocess import from_list_of_dicts
    data = []
    for sid, loc in sid_locs:
        y = grid[sid].values[:cutoff_pos].astype(np.float32)
        valid = np.where(~np.isnan(y))[0]
        if len(valid) < 24 * 21:
            continue
        a, b = valid[0], valid[-1] + 1
        cov = {k: v[a:b] for k, v in exo.full(loc).items()}
        data.append({"target": y[a:b], "past_covariates": cov})
    if USE_K18:  # 2018-19 MPCB stations (challenge AdditionalHistory) with ERA5 weather + k18 neighbours, same keys
        import k18
        extra = k18.ft_series(keys=set(data[0]["past_covariates"]))
        print(f"finetune: + {len(extra)} k18 series (2018-19, up to {k18.END})", flush=True)
        data += extra
    steps, lr, seed = steps or FT_STEPS, lr or FT_LR, FT_SEED if seed is None else seed
    names = [k for k in data[0]["past_covariates"] if not k.startswith(("nb_", "cp_"))]  # past-only context
    inputs = from_list_of_dicts(data, prediction_length=H, known_covariates_names=names)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    print(f"finetune on {len(data)} series, cutoff {grid.index[cutoff_pos]}", flush=True)
    return base.fit(inputs, prediction_length=H, num_steps=steps, learning_rate=lr, batch_size=32,
                    context_length=CTX, output_dir=str(PROC / f"chronos_ft_{os.getpid()}"), remove_printer_callback=True,
                    seed=seed, report_to="none")


def all_sid_locs(d, grid):
    locs = d.groupby("sid").location_id.first()
    return [(s, str(int(locs[s])) if int(locs[s]) in TARGETS else "-1") for s in grid.columns]


MODELS = {"chronos2": chronos_forecast, "timesfm3": timesfm_forecast, "timestral": timestral_forecast,
          "t0": t0_forecast, "timeh2o": timeh2o_forecast, "tabpfn": tabpfn_forecast}


def cv(models=("chronos2", "timesfm3"), out_path=None):
    from train import FOLDS
    d = load_all()
    d = d[~d.sid.str.startswith("k18")]
    grid = to_grid(d)
    grid = grid.reindex(pd.date_range(grid.index.min(), grid.index.max() + pd.Timedelta(hours=HMAX), freq="h"))
    from neighbors import build_signals
    exo = Exo(grid.index, nb=build_signals(d, grid.index) if USE_NB else None)
    locs = d.groupby("sid").location_id.first()
    tsids = [(s, str(int(locs[s]))) for s in grid.columns if int(locs[s]) in TARGETS]
    outs = []
    for name, a, b in FOLDS:
        org = np.where((grid.index >= a) & (grid.index < b) & (grid.index.hour == 23))[0]
        items = build_items(grid, exo, tsids, org)
        print(name, len(items), "items")
        res = None
        global FT_PIPE
        ft = chronos_finetune(grid, exo, all_sid_locs(d, grid), int(org.min()) + 1) if "chronos2ft" in models else None
        for model, fn in ((m, MODELS.get(m, chronos_forecast)) for m in models):
            FT_PIPE = ft if model == "chronos2ft" else None
            for cov in (True,):
                try:
                    P = fn(items, HMAX, use_cov=cov)
                except Exception as e:  # noqa
                    print(model, cov, "FAILED", repr(e)[:300])
                    continue
                col = model + ("_cov" if cov else "_uni")
                L = to_long(items, np.clip(P, 0, None), grid, col)
                res = L if res is None else res.merge(L, on=["sid", "ts0", "ts", "h"], how="outer")
                print(name, col, "ok")
        FT_PIPE = ft = None
        torch.cuda.empty_cache()
        res["fold"] = name
        outs.append(res)
    O = pd.concat(outs, ignore_index=True)
    if out_path is not None:
        O.to_parquet(out_path)
        return O
    f = PROC / "oof_tsfm.parquet"
    if f.exists() and set(models) != {"chronos2", "timesfm3"}:
        old = pd.read_parquet(f)
        old = old.drop(columns=[c for c in O.columns if c not in ("sid", "ts0", "ts", "h", "fold") and c in old])
        O = old.merge(O.drop(columns=["fold"]), on=["sid", "ts0", "ts", "h"], how="left")
    O.to_parquet(f)
    print(O.describe().T)


if __name__ == "__main__":
    if sys.argv[1] == "cv":
        cv(tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("chronos2", "timesfm3"))
