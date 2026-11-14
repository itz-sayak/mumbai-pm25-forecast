"""Download the real-world fields behind the figures (cached in viz/cache/, not committed).

  * CAMS global (ECMWF Copernicus) PM2.5 + aerosol optical depth, 0.4 deg grid over Maharashtra   (Open-Meteo air-quality API)
  * ECMWF IFS 0.25 deg 10 m wind, 2 m temperature, humidity on the same grid                    (Open-Meteo forecast API)
  * GFS boundary-layer height on the same grid (IFS BLH is not served)
  * ECMWF IFS 10 m wind + GFS BLH on a 0.05 deg grid over Mumbai (sea/land-breeze detail)
  * hourly PM2.5 from every Mumbai-region CPCB/MPCB/IITM/BMC monitor                                 (OpenAQ v3, env OPENAQ_API_KEY)

python viz/fetch.py
"""
import json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"; CACHE.mkdir(exist_ok=True)
START, END = "2026-10-28", "2026-11-13"
MH = dict(lat=np.round(np.arange(15.6, 22.41, 0.4), 2), lon=np.round(np.arange(72.4, 81.21, 0.4), 2))
MUM = dict(lat=np.round(np.arange(18.85, 19.36, 0.05), 3), lon=np.round(np.arange(72.70, 73.16, 0.05), 3))


def grid_points(g):
    la, lo = np.meshgrid(g["lat"], g["lon"], indexing="ij")
    return la.ravel(), lo.ravel()


def om(base, lats, lons, params, chunk=60, pause=20):
    out = []
    for i in range(0, len(lats), chunk):
        p = dict(params, latitude=",".join(map(str, lats[i:i + chunk])), longitude=",".join(map(str, lons[i:i + chunk])),
                 start_date=START, end_date=END, timezone="Asia/Kolkata")
        for attempt in range(6):
            r = requests.get(base, params=p, timeout=180)
            if r.status_code == 200:
                break
            print("  retry", r.status_code, r.text[:150]); time.sleep(60 * (attempt + 1))
        j = r.json(); out += j if isinstance(j, list) else [j]
        print(f"  {base.split('/')[2]} {min(i + chunk, len(lats))}/{len(lats)}", flush=True); time.sleep(pause)
    return out


def to_cube(js, g, var):
    t = pd.to_datetime(js[0]["hourly"]["time"])
    a = np.array([[np.nan if v is None else v for v in j["hourly"][var]] for j in js], float)
    return t, a.T.reshape(len(t), len(g["lat"]), len(g["lon"]))


def save(name, g, js, vars_):
    out = {}
    for v in vars_:
        t, cube = to_cube(js, g, v); out[v] = cube
    np.savez_compressed(CACHE / f"{name}.npz", time=t.values.astype("datetime64[s]"), lat=g["lat"], lon=g["lon"], **out)
    print("saved", name, {v: (np.isfinite(c).mean().round(2), np.nanmax(c).round(1)) for v, c in out.items()})


def fields():
    AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"
    WX = "https://api.open-meteo.com/v1/forecast"
    jobs = [
        ("mh_cams", MH, AQ, dict(hourly="pm2_5,aerosol_optical_depth,dust"), ["pm2_5", "aerosol_optical_depth", "dust"]),
        ("mh_ifs", MH, WX, dict(hourly="wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m",
                                models="ecmwf_ifs025", wind_speed_unit="ms"),
         ["wind_speed_10m", "wind_direction_10m", "temperature_2m", "relative_humidity_2m"]),
        ("mh_blh", MH, WX, dict(hourly="boundary_layer_height", models="gfs_seamless"), ["boundary_layer_height"]),
        ("mum_ifs", MUM, WX, dict(hourly="wind_speed_10m,wind_direction_10m,temperature_2m", models="ecmwf_ifs025",
                                  wind_speed_unit="ms"), ["wind_speed_10m", "wind_direction_10m", "temperature_2m"]),
        ("mum_blh", MUM, WX, dict(hourly="boundary_layer_height", models="gfs_seamless"), ["boundary_layer_height"]),
    ]
    for name, g, base, params, vars_ in jobs:
        if (CACHE / f"{name}.npz").exists():
            print("cached", name); continue
        la, lo = grid_points(g)
        save(name, g, om(base, la, lo, params), vars_)


def monitors():
    f = CACHE / "monitors.csv"
    if f.exists():
        print("cached monitors"); return
    key = os.environ["OPENAQ_API_KEY"]
    s = pd.read_csv(ROOT / "monitors.csv")
    rows = []
    for _, m in s.iterrows():
        page = 1
        while True:
            r = requests.get(f"https://api.openaq.org/v3/sensors/{m.sensor}/hours", headers={"X-API-Key": key}, timeout=60,
                             params={"datetime_from": f"{START}T00:00:00+05:30", "datetime_to": f"{END}T23:59:00+05:30",
                                     "limit": 1000, "page": page})
            if r.status_code == 429:
                time.sleep(30); continue
            res = r.json().get("results", []) if r.status_code == 200 else []
            for x in res:
                te = ((x.get("period") or {}).get("datetimeTo") or {}).get("local")
                n = ((x.get("coverage") or {}).get("observedCount")) or 4
                if te and x.get("value") is not None and x["value"] >= 0 and n >= 2:
                    rows.append((pd.Timestamp(te[:19]), m["loc"], x["value"]))
            if len(res) < 1000:
                break
            page += 1
        time.sleep(1.2)
    df = pd.DataFrame(rows, columns=["ts", "loc", "pm25"]).drop_duplicates(["ts", "loc"])
    df.to_csv(f, index=False)
    print("monitors", df["loc"].nunique(), "stations", len(df), "rows", df.ts.min(), df.ts.max())


if __name__ == "__main__":
    monitors()
    fields()
