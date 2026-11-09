"""2018-19 MPCB history (challenge AdditionalHistory/History_Mumbai_2018_2020) as extra fine-tuning series for Chronos-2.

Adds two pre-COVID Diwalis (2018-11-07, 2019-10-27) on 10 Mumbai stations. Covariates match tsfm.Exo.full() with
NO_CAMS (CAMS has no 2018-19 history): ERA5 weather at the city point (Open-Meteo archive; the previous-runs forecast
archive starts in 2024), hour sin/cos, fireworks flag, and nb_med / nb_near from the other MPCB stations.
2020 (COVID) is excluded. K18_END (env) cuts the history earlier, e.g. for a leave-2019-out rehearsal.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import data as D
import tsfm as TS
from neighbors import K_NEAR, _hav

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
START = "2018-01-01"
END = os.environ.get("K18_END", "2019-12-31")
MPCB = {"MH005": (19.0627, 72.8461), "MH006": (19.2324, 72.8690), "MH007": (19.1003, 72.8746),
        "MH008": (18.9100, 72.8200), "MH009": (19.0863, 72.8888), "MH010": (19.1375, 72.9151),
        "MH011": (19.0470, 72.8746), "MH012": (19.3832, 72.8204), "MH013": (19.1086, 72.8362),
        "MH014": (18.9936, 72.8128)}  # approximate public coordinates (nearest-neighbour weights only)


def era5(start=START, end="2019-12-31"):
    """Hourly ERA5 at the city point, hour-ending index (as features.load_weather)."""
    f = EXT / f"era5_city_{start}_{end}.json"
    if not f.exists():
        url = ("https://archive-api.open-meteo.com/v1/archive?latitude=19.07&longitude=72.87"
               f"&start_date={start}&end_date={end}&hourly={','.join(sorted(set(TS.COV.values())))}"
               "&timezone=Asia%2FKolkata")
        for i in range(6):
            r = requests.get(url, timeout=300)
            if r.status_code == 200:
                js = r.json()
                js["_retrieved"] = pd.Timestamp.now().isoformat()
                f.write_text(json.dumps(js))
                break
            time.sleep(10 * (i + 1))
    h = json.loads(f.read_text())["hourly"]
    w = pd.DataFrame(h)
    w.index = pd.to_datetime(w.pop("time")) + pd.Timedelta(hours=1)
    return w


def _nb(grid, codes):
    V = grid[[f"k18_{c}" for c in codes]].values.astype(float)
    out = {}
    for i0, loc in enumerate(codes):
        use = np.array([i for i in range(len(codes)) if i != i0])
        sub = V[:, use]
        with np.errstate(all="ignore"):
            med = np.nanmedian(sub, 1)
        dist = np.array([_hav(*MPCB[loc], *MPCB[codes[i]]) for i in use])
        order = np.argsort(dist)
        S, w = sub[:, order], 1.0 / (dist[order] + 1.0)
        avail = ~np.isnan(S)
        take = avail & (np.cumsum(avail, 1) <= K_NEAR)
        num, den = (np.where(take, S, 0) * w).sum(1), (take * w).sum(1)
        out[loc] = {"nb_med": med, "nb_near": np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)}
    return out


def ft_series(keys=None, min_days=21):
    """List of {"target", "past_covariates"} dicts (tsfm.chronos_finetune format) for the k18 stations up to END."""
    d = D.load_all()
    k = d[d.sid.str.startswith("k18_") & (d.ts <= pd.Timestamp(END) + pd.Timedelta(days=1))]
    grid = D.to_grid(k)
    codes = [c for c in MPCB if f"k18_{c}" in grid.columns]
    wx = era5().reindex(grid.index)
    hrs = grid.index.hour.values
    base = {key: wx[v].values.astype(float) for key, v in TS.COV.items()}
    wd = np.deg2rad(base.pop("wd"))
    base["u"], base["v"] = -base["ws"] * np.sin(wd), -base["ws"] * np.cos(wd)
    base["hs"], base["hc"] = np.sin(2 * np.pi * hrs / 24), np.cos(2 * np.pi * hrs / 24)
    if TS.USE_FW:
        base["fw"] = D.fireworks_flag(grid.index).astype(float)
    nb = _nb(grid, codes) if TS.USE_NB else {}
    out = []
    for c in codes:
        y = grid[f"k18_{c}"].values.astype(np.float32)
        valid = np.where(~np.isnan(y))[0]
        if len(valid) < 24 * min_days:
            continue
        a, b = valid[0], valid[-1] + 1
        cov = {**base, **nb.get(c, {})}
        if keys is not None:
            missing = set(keys) - set(cov)
            if missing:
                raise ValueError(f"k18 series lack covariates {sorted(missing)} (use a _nocams config)")
            cov = {kk: cov[kk] for kk in keys}
        cov = {kk: pd.Series(v[a:b]).interpolate(limit_direction="both").fillna(0).values.astype(np.float32)
               for kk, v in cov.items()}
        out.append({"target": y[a:b], "past_covariates": cov})
    return out
