"""Origin-based direct multi-horizon samples.

A sample = (series sid, origin index t0 [last hour of info], horizon h) -> target y[t0+h].
All history features use only y[<= t0]. Weather/CAMS at target time use forecasts with
lead-day matched to h (Open-Meteo previous_dayN), mimicking what is known at t0.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data import TARGETS, diwali_offset_days, fireworks_flag

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
WV = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "surface_pressure",
      "cloud_cover", "wind_speed_10m", "wind_direction_10m", "boundary_layer_height", "shortwave_radiation"]
SHORT = {"temperature_2m": "t2", "relative_humidity_2m": "rh", "dew_point_2m": "td", "precipitation": "pr",
         "surface_pressure": "ps", "cloud_cover": "cc", "wind_speed_10m": "ws", "wind_direction_10m": "wd",
         "boundary_layer_height": "blh", "shortwave_radiation": "sw"}


# ----------------------------------------------------------------------------- exogenous
def load_weather(fcst_tag=None):
    """Return dict lead(0..3) -> DataFrame index=ts(hour-end), columns MultiIndex (var, loc) incl loc 'city'.
    If fcst_tag is given, rows after the history end are filled with the snapshot forecast (all leads)."""
    w = pd.read_parquet(EXT / "weather_prevruns.parquet")
    w["location_id"] = w["location_id"].astype(int)
    w["ts"] = w["time"] + pd.Timedelta(hours=1)
    out = {}
    fc = None
    if fcst_tag:
        fc = pd.read_parquet(EXT / f"weather_fcst_{fcst_tag}.parquet")
        fc["location_id"] = fc["location_id"].astype(int)
        fc["ts"] = fc["time"] + pd.Timedelta(hours=1)
    for L in range(4):
        d = w[["ts", "location_id"] + WV].copy()
        if L:
            for v in WV:  # lead-L forecast where archived, else fall back to day-0 (e.g. BLH)
                d[v] = w[f"{v}_previous_day{L}"].fillna(w[v]).values
        if fc is not None:
            # previous-runs best value (day0) wins for the past; forecast fills the future
            d = pd.concat([d, fc[["ts", "location_id"] + WV]]).drop_duplicates(["ts", "location_id"], keep="first")
        p = d.pivot_table(index="ts", columns="location_id", values=WV)
        city = p.T.groupby(level=0).mean().T
        city.columns = pd.MultiIndex.from_product([city.columns, ["city"]])
        p.columns = pd.MultiIndex.from_tuples([(a, str(b)) for a, b in p.columns])
        out[L] = pd.concat([p, city], axis=1).sort_index(axis=1)
    if WX_MULTI and (EXT / "weather_multinwp.parquet").exists():
        out = _apply_multi(out, fcst_tag)
    return out


WX_MULTI = False  # ablation E2: replace best-match by 4-NWP ensemble mean (+ spread vars)
NWP = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "ecmwf_aifs025_single"]
ENS_VARS = ["wind_speed_10m", "relative_humidity_2m", "precipitation", "temperature_2m"]
SPREAD = {"wind_speed_10m": "ws_sd", "relative_humidity_2m": "rh_sd", "precipitation": "pr_sd"}


def _ens_frame(M, L, has_leads=True):
    """Ensemble mean / spread per (ts, location) for lead L from a multi-NWP table."""
    suf = f"_previous_day{L}" if (L and has_leads) else ""
    out = M[["ts", "location_id"]].copy()
    for v in ENS_VARS:
        cols = [f"{v}{suf}_{m}" for m in NWP if f"{v}{suf}_{m}" in M]
        X = M[cols].astype(float).values if cols else np.full((len(M), 1), np.nan)
        with np.errstate(all="ignore"):
            out[v] = np.nanmean(X, 1)
            if v in SPREAD:
                out[SPREAD[v]] = np.nanstd(X, 1)
    # vector-mean wind direction across models
    us, vs = [], []
    for m in NWP:
        a, b = f"wind_speed_10m{suf}_{m}", f"wind_direction_10m{suf}_{m}"
        if a in M and b in M:
            r = np.deg2rad(M[b].astype(float).values)
            us.append(-M[a].astype(float).values * np.sin(r))
            vs.append(-M[a].astype(float).values * np.cos(r))
    if us:
        with np.errstate(all="ignore"):
            u, v = np.nanmean(us, 0), np.nanmean(vs, 0)
        out["wind_direction_10m"] = (np.rad2deg(np.arctan2(-u, -v)) + 360) % 360
    return out


def _apply_multi(out, fcst_tag):
    """Overwrite ensemble vars in load_weather output with the 4-NWP mean; add spread vars."""
    M = pd.read_parquet(EXT / "weather_multinwp.parquet")
    M["location_id"] = M["location_id"].astype(int)
    M["ts"] = M["time"] + pd.Timedelta(hours=1)
    F = None
    if fcst_tag and (EXT / f"weather_mnwp_fcst_{fcst_tag}.parquet").exists():
        F = pd.read_parquet(EXT / f"weather_mnwp_fcst_{fcst_tag}.parquet")
        F["location_id"] = F["location_id"].astype(int)
        F["ts"] = F["time"] + pd.Timedelta(hours=1)
    for L in out:
        E = _ens_frame(M, L)
        if F is not None:
            E = pd.concat([E, _ens_frame(F, L, has_leads=False)]).drop_duplicates(["ts", "location_id"], keep="first")
        vars_ = [c for c in E.columns if c not in ("ts", "location_id")]
        p = E.pivot_table(index="ts", columns="location_id", values=vars_)
        city = p.T.groupby(level=0).mean().T
        city.columns = pd.MultiIndex.from_product([city.columns, ["city"]])
        p.columns = pd.MultiIndex.from_tuples([(a, str(b)) for a, b in p.columns])
        p = pd.concat([p, city], axis=1)
        base = out[L].reindex(out[L].index.union(p.index))
        for col in p.columns:
            new = p[col].reindex(base.index)
            base[col] = new.where(new.notna(), base[col]) if col in base.columns else new
        out[L] = base.sort_index(axis=1)
    return out


def load_cams(fcst_tag=None):
    c = pd.read_parquet(EXT / "cams_hist.parquet")
    if fcst_tag:
        f = pd.read_parquet(EXT / f"cams_fcst_{fcst_tag}.parquet")
        c = pd.concat([c, f]).drop_duplicates(["time", "location_id"], keep="last")
    c["location_id"] = c["location_id"].astype(int)
    c["ts"] = c["time"] + pd.Timedelta(hours=1)
    p = c.pivot_table(index="ts", columns="location_id", values="pm2_5")
    p.columns = [str(x) for x in p.columns]
    p["city"] = p.mean(1)
    return p


# ----------------------------------------------------------------------------- helpers
def _roll(y, w, fn="mean"):
    s = pd.Series(y)
    r = s.rolling(w, min_periods=max(1, w // 4))
    return getattr(r, fn)().values


def series_arrays(y):
    """Precompute per-index history statistics (inclusive of index)."""
    s = pd.Series(y)
    idx = np.arange(len(y), dtype=float)
    lastidx = pd.Series(np.where(~np.isnan(y), idx, np.nan)).ffill().values
    A = {
        "last": s.ffill().values,
        "age": idx - lastidx,
        "m3": _roll(y, 3), "m6": _roll(y, 6), "m12": _roll(y, 12), "m24": _roll(y, 24),
        "m72": _roll(y, 72), "m168": _roll(y, 168), "m336": _roll(y, 336),
        "med24": _roll(y, 24, "median"), "sd24": _roll(y, 24, "std"),
        "mx24": _roll(y, 24, "max"), "mn24": _roll(y, 24, "min"),
        "q90_168": s.rolling(168, min_periods=40).quantile(0.9).values,
        "n24": s.rolling(24, min_periods=1).count().values,
        "ew12": s.ewm(halflife=12, ignore_na=True).mean().values,
        "ew48": s.ewm(halflife=48, ignore_na=True).mean().values,
    }
    # same-hour stacks: S_k[t] = nanmean(y[t], y[t-24], ..., y[t-24(k-1)])
    for k in (3, 7, 14):
        st = np.vstack([np.r_[np.full(24 * j, np.nan), y[: len(y) - 24 * j]] for j in range(k)])
        with np.errstate(all="ignore"):
            A[f"S{k}"] = np.nanmean(st, 0)
    return A


def build_samples(grid, sids, origins_fn, horizons, wx, cams, loc_of, need_target=True, nb=None):
    """grid: wide hourly DataFrame (ts x sid). origins_fn(ts_index, sid) -> array of origin positions.
    Returns DataFrame of samples."""
    ts = grid.index
    T = len(ts)
    hod = ts.hour.values
    # exogenous aligned to grid
    wxa = {L: wx[L].reindex(ts) for L in wx}
    camsa = cams.reindex(ts)
    fwg = fireworks_flag(ts)
    fw48 = pd.Series(fwg).rolling(48, min_periods=1).sum().values
    frames = []
    for sid in sids:
        y = grid[sid].values.astype(float)
        A = series_arrays(y)
        o = origins_fn(ts, sid, y)
        if len(o) == 0:
            continue
        loc = loc_of(sid)
        wl = loc if loc in set(wx[0].columns.get_level_values(1)) else "city"
        cl = loc if loc in camsa.columns else "city"
        cam = camsa[cl].values.astype(float)
        NBF = {}
        if nb is not None:  # neighbour signals: past-only, summarised at the origin
            src = nb.get(loc, nb["city"])
            for c in src.columns:
                a = pd.Series(src[c].values.astype(float))
                NBF[c + "_m3"] = a.rolling(3, min_periods=1).mean().values
                NBF[c + "_m24"] = a.rolling(24, min_periods=6).mean().values
                NBF[c + "_d24"] = NBF[c + "_m3"] - NBF[c + "_m24"]
        cam24 = pd.Series(cam).rolling(24, min_periods=6).mean().values
        cam72 = pd.Series(cam).rolling(72, min_periods=12).mean().values
        # obs/cams ratio over past 72h at origin
        rat = pd.Series(np.where(np.isnan(y), np.nan, y)).rolling(72, min_periods=12).mean().values / np.maximum(cam72, 1)
        # weather arrays
        W = {}
        for L in range(4):
            for v in WV:
                col = (v, wl)
                W[(L, v)] = wxa[L][col].values.astype(float) if col in wxa[L].columns else np.full(T, np.nan)
        for h in horizons:
            t = o + h
            ok = t < T
            oo, tt = o[ok], t[ok]
            if need_target:
                yt = y[tt]
                keep = ~np.isnan(yt)
                oo, tt, yt = oo[keep], tt[keep], yt[keep]
            else:
                yt = np.full(len(tt), np.nan)
            if len(oo) == 0:
                continue
            kd = int(np.ceil(h / 24))
            back = tt - 24 * kd  # most recent same-hour index <= t0
            f = {"sid": sid, "loc": loc, "t0": oo, "t": tt, "h": h, "y": yt}
            for k in ("last", "age", "m3", "m6", "m12", "m24", "m72", "m168", "m336", "med24", "sd24", "mx24",
                      "mn24", "q90_168", "n24", "ew12", "ew48"):
                f[k] = A[k][oo]
            for k in (3, 7, 14):
                f[f"S{k}"] = A[f"S{k}"][np.clip(back, 0, None)]
            f["lagsh"] = y[np.clip(back, 0, None)]
            for k, a in NBF.items():
                f[k] = a[oo]
            if NBF:
                f["r_self_nb"] = A["m24"][oo] / np.maximum(NBF["nb_near_m24"][oo], 1)
            f["fw"] = fwg[tt]
            f["fw48_o"] = fw48[oo]
            f["hod"] = hod[tt]
            f["hod0"] = hod[oo]
            L = min(3, max(1, kd))
            for v in WV:
                a = W[(L, v)]
                f[SHORT[v]] = a[tt]
                f[SHORT[v] + "_o"] = W[(0, v)][oo]
            # derived meteorology
            wd = np.deg2rad(f["wd"])
            f["u"] = -f["ws"] * np.sin(wd)
            f["v"] = -f["ws"] * np.cos(wd)
            f["onshore"] = f["u"]  # wind FROM the west (Arabian Sea) => eastward flow => u > 0
            f["vent"] = f["blh"] * f["ws"]
            f["vent_o"] = f["blh_o"] * f["ws_o"]
            # 24h forecast means ending at target
            for v, nm in (("boundary_layer_height", "blh24"), ("wind_speed_10m", "ws24"),
                          ("relative_humidity_2m", "rh24"), ("precipitation", "pr24")):
                a = pd.Series(W[(L, v)]).rolling(24, min_periods=6)
                a = (a.sum() if v == "precipitation" else a.mean()).values
                f[nm] = a[tt]
                b = pd.Series(W[(0, v)]).rolling(24, min_periods=6)
                b = (b.sum() if v == "precipitation" else b.mean()).values
                f[nm + "_o"] = b[oo]
            f["dblh24"] = f["blh24"] - f["blh24_o"]
            f["dws24"] = f["ws24"] - f["ws24_o"]
            for sv in ("ws_sd", "rh_sd", "pr_sd"):
                col = (sv, wl)
                if col in wxa[L].columns:
                    f[sv] = wxa[L][col].values.astype(float)[tt]
            f["cams"] = cam[tt]
            f["cams24"] = cam24[tt]
            f["cams72_o"] = cam72[oo]
            f["cams_rat"] = rat[oo]
            f["cams_mos"] = f["cams"] * np.clip(f["cams_rat"], 0.1, 5)
            f["cams_mos24"] = f["cams24"] * np.clip(f["cams_rat"], 0.1, 5)
            frames.append(pd.DataFrame(f))
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = ts[df.t.values]
    df["ts0"] = ts[df.t0.values]
    df["dow"] = df.ts.dt.dayofweek
    df["doy"] = df.ts.dt.dayofyear
    df["month"] = df.ts.dt.month
    df["dvl"] = diwali_offset_days(df.ts)
    df["dvl_c"] = df.dvl.clip(-10, 15)
    df["dvl0"] = diwali_offset_days(df.ts0).clip(-10, 15)
    df["is_target"] = df["loc"].astype(int).isin(list(TARGETS)).astype(int)
    df["loc_cat"] = df["loc"].astype(int)
    # relative features (scale-free)
    lvl = np.nanmean(np.vstack([df.m24.values, df.ew48.values]), 0)
    df["lvl"] = np.where(np.isnan(lvl), df["last"], lvl)
    for k in ("S3", "S7", "S14", "lagsh", "last", "m3", "m72", "m168", "cams_mos", "cams_mos24"):
        df[f"r_{k}"] = df[k] / np.maximum(df["lvl"], 1)
    return df
