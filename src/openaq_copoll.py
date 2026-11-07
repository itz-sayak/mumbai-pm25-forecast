"""Co-located pollutants + on-site meteorology at the 4 target stations (OpenAQ v3, key in OPENAQ_API_KEY).

PAST-ONLY covariates (never read beyond the forecast origin). HARD CUTOFF as openaq_live (2026-11-11 00:00 IST).
* pm10, no2, co, o3, relativehumidity -> /sensors/{id}/hours (hourly means)
* wind_speed + wind_direction        -> /sensors/{id}/measurements (raw), combined into hourly vector u/v
  (hourly-averaged directions from /hours are wrong across the 0/360 wrap)
Per-sensor cache data/external/copoll/<sensor>.parquet ; output data/external/copoll.parquet (long:
[ts, location_id, var, value]).

python src/openaq_copoll.py [start=2024-12-01]
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openaq_live import CUTOFF_DEFAULT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
CP = EXT / "copoll"
CP.mkdir(parents=True, exist_ok=True)
TARGET_LOCS = [11611, 3409511, 11606, 3409510]
HOURLY = ["pm10", "no2", "co", "o3", "relativehumidity"]
WIND = ["wind_speed", "wind_direction"]


def _get(url, key, params):
    for attempt in range(6):
        try:
            r = requests.get(url, headers={"X-API-Key": key}, params=params, timeout=90)
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
            continue
        time.sleep(1.1)
        if r.status_code == 200:
            return r.json().get("results", [])
        if r.status_code in (408, 429, 500, 502, 503, 504):
            time.sleep(20 * (attempt + 1))
            continue
        print("  http", r.status_code, r.text[:120], flush=True)
        return None
    return None


def fetch(sensor, kind, start, end, key, window_days=30):
    """Paged fetch in 30-day windows (long windows time out with HTTP 408)."""
    url = f"https://api.openaq.org/v3/sensors/{sensor}/{kind}"
    out = []
    w0 = pd.Timestamp(start)
    while w0 < end:
        w1 = min(w0 + pd.Timedelta(days=window_days), end)
        page = 1
        while True:
            res = _get(url, key, {"datetime_from": w0.isoformat(), "datetime_to": w1.isoformat(),
                                  "limit": 1000, "page": page})
            if res is None:
                break
            for x in res:
                te = ((x.get("period") or {}).get("datetimeTo") or {}).get("local")
                if te is not None:
                    out.append({"t": pd.Timestamp(te[:19]), "value": x.get("value")})
            if len(res) < 1000:
                break
            page += 1
        w0 = w1
    return pd.DataFrame(out)


def sensor_map(key):
    rows = []
    for loc in TARGET_LOCS:
        r = _get(f"https://api.openaq.org/v3/locations/{loc}", key, {})
        for s in r[0]["sensors"]:
            p = s["parameter"]["name"]
            if p in HOURLY + WIND:
                rows.append({"location_id": loc, "var": p, "sensor": s["id"]})
    m = pd.DataFrame(rows).sort_values("sensor").groupby(["location_id", "var"]).tail(1)  # newest sensor id
    m.to_csv(EXT / "copoll_sensors.csv", index=False)
    return m


def cached(sensor, kind, start, cut, key):
    f = CP / f"{sensor}_{kind}.parquet"
    old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["t", "value"])
    st = pd.Timestamp(start) if old.empty else old.t.max() - pd.Timedelta(hours=6)
    new = fetch(sensor, kind, st, cut, key)
    df = pd.concat([old, new]).dropna(subset=["t"])
    df["t"] = pd.to_datetime(df["t"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.drop_duplicates("t", keep="last")
    df = df[df.t < cut]
    df.to_parquet(f)
    return df


def run(start="2024-12-01"):
    key = os.environ["OPENAQ_API_KEY"]
    cut = min(pd.Timestamp(CUTOFF_DEFAULT), pd.Timestamp.now().floor("h"))
    m = sensor_map(key)
    parts = []
    for loc in TARGET_LOCS:
        mm = m[m.location_id == loc].set_index("var").sensor
        for v in HOURLY:
            if v not in mm:
                continue
            d = cached(int(mm[v]), "hours", start, cut, key)
            d = d[pd.to_numeric(d.value, errors="coerce") >= 0]
            parts.append(pd.DataFrame({"ts": d.t, "location_id": loc, "var": v, "value": d.value.astype(float)}))
            print(loc, v, len(d), d.t.max(), flush=True)
        if "wind_speed" in mm and "wind_direction" in mm:
            ws = cached(int(mm["wind_speed"]), "measurements", start, cut, key)
            wd = cached(int(mm["wind_direction"]), "measurements", start, cut, key)
            w = ws.merge(wd, on="t", suffixes=("_s", "_d")).dropna()
            w = w[(w.value_s >= 0) & (w.value_d >= 0) & (w.value_d <= 360)]
            rad = np.deg2rad(w.value_d.astype(float))
            w = w.assign(u=-w.value_s * np.sin(rad), v=-w.value_s * np.cos(rad), ts=w.t.dt.ceil("h"))
            g = w.groupby("ts")[["u", "v"]].mean().reset_index()
            for c in ("u", "v"):
                parts.append(pd.DataFrame({"ts": g.ts, "location_id": loc, "var": "obs_" + c, "value": g[c]}))
            print(loc, "wind", len(g), g.ts.max(), flush=True)
    A = pd.concat(parts, ignore_index=True)
    A = A[A.ts < cut]
    A.to_parquet(EXT / "copoll.parquet")
    print("total", len(A), "cutoff", cut)


if __name__ == "__main__":
    run(*sys.argv[1:])
