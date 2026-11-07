"""City-wide neighbour signals for each target station (past-only use at the forecast origin).

nb_med  : median PM2.5 over all other stations at that hour
nb_near : inverse-distance weighted mean of the 4 nearest other stations (NaN-aware)
Sources : OpenAQ neighbours 2025-26 (openaq_neighbors.parquet), XKDR CPCB 2024 stations, the other targets.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data import RAW, TARGETS, clean_series

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
K_NEAR = 4
USE_UP = False  # Lagrangian upwind signal (ablation E3)
USE_CP = False  # co-pollutants / station met (ablation E1)


def _hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def neighbour_table(d):
    """d = load_all() output. Returns wide (ts x key) values and meta (key -> lat, lon, loc_id)."""
    st = pd.read_csv(RAW / "Stations.csv").set_index("location_id")
    xk = pd.read_csv(RAW / "AdditionalHistory" / "XKDR_All_Stations.csv").set_index("station_id")
    series, meta = [], {}
    # targets (Train + optional live)
    for loc in TARGETS:
        g = d[(d.sid == str(loc))][["ts", "y"]]
        series.append(g.assign(key=f"t{loc}"))
        meta[f"t{loc}"] = (st.loc[loc, "latitude"], st.loc[loc, "longitude"], loc)
    # XKDR 2024 CPCB stations
    for sid, g in d[d.sid.str.startswith("x24_")].groupby("sid"):
        code = sid[4:]
        if code not in xk.index:
            continue
        series.append(g[["ts", "y"]].assign(key=sid))
        meta[sid] = (xk.loc[code, "latitude"], xk.loc[code, "longitude"], int(g.location_id.iloc[0]))
    # OpenAQ neighbours 2025-26
    f = EXT / "openaq_neighbors.parquet"
    if f.exists():
        nb = pd.read_parquet(f)
        for loc, g in nb.groupby("nb_loc"):
            g = clean_series(g[["ts", "y"]].sort_values("ts").drop_duplicates("ts"))
            series.append(g.assign(key=f"n{loc}"))
            first = nb[nb.nb_loc == loc].iloc[0]
            meta[f"n{loc}"] = (first.lat, first.lon, -1)
    L = pd.concat(series, ignore_index=True)
    W = L.pivot_table(index="ts", columns="key", values="y", aggfunc="mean")
    return W, meta


def _wind_uv(index, upto=None):
    """Hourly flow vectors (u east, v north; km/h) per target: station observations where available, else NWP day-0."""
    from features import load_weather
    w0 = load_weather()[0]
    out = {}
    cp = EXT / "copoll.parquet"
    C = pd.read_parquet(cp) if cp.exists() else None
    for loc in [str(k) for k in TARGETS]:
        ws = w0[("wind_speed_10m", loc)].reindex(index).values.astype(float)
        wd = np.deg2rad(w0[("wind_direction_10m", loc)].reindex(index).values.astype(float))
        u, v = -ws * np.sin(wd), -ws * np.cos(wd)
        if C is not None:
            c = C[C.location_id == int(loc)].pivot_table(index="ts", columns="var", values="value")
            if upto is not None:
                c = c[c.index <= pd.Timestamp(upto)]
            c = c.reindex(index)
            if "obs_u" in c:  # station anemometer is m/s (verified via OpenAQ units) -> km/h
                ou, ov = c["obs_u"].values * 3.6, c["obs_v"].values * 3.6
                dead = np.hypot(ou, ov) < 0.05  # broken sensors report exact zeros (Malad W, Borivali E)
                ou, ov = np.where(dead, np.nan, ou), np.where(dead, np.nan, ov)
                u, v = np.where(np.isnan(ou), u, ou), np.where(np.isnan(ov), v, ov)
        out[loc] = (u, v)
    return out


def upwind_signal(V, keys, meta, loc, uv, ell=10.0, maxlag=12):
    """Lagrangian upwind neighbour signal (past-only):
    S_i(t) = sum_j w_ij(t) C_j(t - tau_ij(t)) / sum_j w_ij(t),
    w_ij = max(0, cos(flow, j->i)) * exp(-d_ij / ell),  tau_ij = d_ij / |flow|  (hours, clipped to maxlag)."""
    st = meta[f"t{loc}"]
    use = [i for i, k in enumerate(keys) if meta[k][2] != int(loc)]
    lat0, lon0 = st[0], st[1]
    dx = np.array([(lon0 - meta[keys[i]][1]) * 111.32 * np.cos(np.deg2rad(lat0)) for i in use])  # km east j->i
    dy = np.array([(lat0 - meta[keys[i]][0]) * 110.57 for i in use])                             # km north j->i
    dist = np.hypot(dx, dy)
    u, v = uv
    T = V.shape[0]
    spd = np.hypot(u, v)
    cosang = (u[:, None] * dx[None, :] + v[:, None] * dy[None, :]) / (np.maximum(spd, 1e-6)[:, None] * np.maximum(dist, 1e-6)[None, :])
    w = np.clip(cosang, 0, None) * np.exp(-dist / ell)[None, :]
    w[np.isnan(w)] = 0
    lag = np.clip(np.round(dist[None, :] / np.maximum(spd, 0.5)[:, None]), 0, maxlag).astype(int)
    rows = np.clip(np.arange(T)[:, None] - lag, 0, T - 1)
    vals = V[:, use][rows, np.arange(len(use))[None, :]]
    m = ~np.isnan(vals) & (w > 0)
    num = np.where(m, vals * w, 0).sum(1)
    den = np.where(m, w, 0).sum(1)
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)


def copoll_signals(d, index, upto=None):
    """Co-located pollutants / on-site met at each target (past-only covariates)."""
    cp = EXT / "copoll.parquet"
    out = {}
    if not cp.exists():
        return out
    C = pd.read_parquet(cp)
    if upto is not None:
        C = C[C.ts <= pd.Timestamp(upto)]
    for loc in [str(k) for k in TARGETS]:
        c = C[C.location_id == int(loc)].pivot_table(index="ts", columns="var", values="value").reindex(index)
        f = pd.DataFrame(index=index)
        for v in ("pm10", "no2", "co", "o3", "relativehumidity", "obs_u", "obs_v"):
            f["cp_" + v] = c[v].values if v in c else np.nan
        dead = np.hypot(f["cp_obs_u"], f["cp_obs_v"]) < 0.01  # zero-reporting anemometers -> missing
        f.loc[dead, ["cp_obs_u", "cp_obs_v"]] = np.nan
        f.loc[f["cp_relativehumidity"] <= 0, "cp_relativehumidity"] = np.nan
        y = d[d.sid == loc].set_index("ts").y.reindex(index)
        f["cp_ratio"] = (y / f["cp_pm10"].where(f["cp_pm10"] > 1)).clip(0, 1.5).values
        out[loc] = f
    return out


def build_signals(d, index, upto=None, up=None, cp=None, ell=10.0):
    """Returns dict: loc(str) or 'city' -> DataFrame(index, [nb_med, nb_near, (+nb_up), (+cp_*)]).
    Values after `upto` are masked. up/cp default to module flags USE_UP / USE_CP."""
    up = USE_UP if up is None else up
    cp = USE_CP if cp is None else cp
    W, meta = neighbour_table(d)
    if upto is not None:
        W = W[W.index <= pd.Timestamp(upto)]
    W = W.reindex(index)
    keys = list(W.columns)
    V = W.values.astype(float)
    out = {}
    for loc in [str(k) for k in TARGETS] + ["city"]:
        if loc == "city":
            use = np.arange(len(keys))
            lat0, lon0 = 19.07, 72.87
        else:
            use = np.array([i for i, k in enumerate(keys) if meta[k][2] != int(loc)])
            st = meta[f"t{loc}"]
            lat0, lon0 = st[0], st[1]
        sub = V[:, use]
        with np.errstate(all="ignore"):
            med = np.nanmedian(sub, 1)
        dist = np.array([_hav(lat0, lon0, meta[keys[i]][0], meta[keys[i]][1]) for i in use])
        # nearest K among those with data at each hour: weight = 1/(d+1) on the K nearest available
        order = np.argsort(dist)
        S = sub[:, order]
        w = 1.0 / (dist[order] + 1.0)
        avail = ~np.isnan(S)
        rank = np.cumsum(avail, 1)
        take = avail & (rank <= K_NEAR)
        num = (np.where(take, S, 0) * w).sum(1)
        den = (take * w).sum(1)
        near = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        out[loc] = pd.DataFrame({"nb_med": med, "nb_near": near}, index=index)
    if up:
        uvs = _wind_uv(index, upto)
        for loc in [str(k) for k in TARGETS]:
            out[loc]["nb_up"] = upwind_signal(V, keys, meta, loc, uvs[loc], ell=ell)
        out["city"]["nb_up"] = out["city"]["nb_med"]
    if cp:
        cps = copoll_signals(d, index, upto)
        cols = None
        for loc, f in cps.items():
            out[loc] = pd.concat([out[loc], f], axis=1)
            cols = list(f.columns)
        if cols:
            for c in cols:
                out["city"][c] = np.nan
    return out
