"""Hourly PM2.5 history for Mumbai-region neighbour stations from OpenAQ v3 (key in OPENAQ_API_KEY).

Used ONLY as past information at the forecast origin (never same-hour/future values).
HARD CUTOFF: nothing with hour-end >= min(now, 2026-11-11 00:00 IST) is stored.
Per-sensor cache: data/external/nb/<sensor>.parquet ; incremental refresh on rerun.
Output: data/external/openaq_neighbors.parquet [ts, nb_loc, y, n]

python src/openaq_neighbors.py [start=2025-02-19]
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openaq_live import CUTOFF_DEFAULT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
NB = EXT / "nb"
NB.mkdir(parents=True, exist_ok=True)
TARGET_LOCS = {11611, 3409511, 11606, 3409510}


def sensors():
    s = pd.read_csv(EXT / "openaq_mumbai_sensors.csv")
    s = s[~s["loc"].isin(TARGET_LOCS)]
    # prefer the newer sensor id per location (the series used by the current OpenAQ/CPCB feed)
    return s.sort_values("sensor").groupby("loc").tail(1)


def fetch_sensor(sensor, start, end, key):
    out, page = [], 1
    while True:
        for attempt in range(5):
            r = requests.get(f"https://api.openaq.org/v3/sensors/{sensor}/hours",
                             headers={"X-API-Key": key}, timeout=90,
                             params={"datetime_from": start.isoformat(), "datetime_to": end.isoformat(),
                                     "limit": 1000, "page": page})
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            break
        time.sleep(1.1)  # ~55 req/min, under the 60/min limit
        if r.status_code != 200:
            print("  http", r.status_code, r.text[:120], flush=True)
            break
        res = r.json().get("results", [])
        for x in res:
            te = ((x.get("period") or {}).get("datetimeTo") or {}).get("local")
            if te is None:
                continue
            out.append({"ts": pd.Timestamp(te[:19]), "y": x.get("value"),
                        "n": (x.get("coverage") or {}).get("observedCount")})
        if len(res) < 1000:
            break
        page += 1
    return pd.DataFrame(out)


def run(start="2025-02-19"):
    key = os.environ["OPENAQ_API_KEY"]
    cut = min(pd.Timestamp(CUTOFF_DEFAULT), pd.Timestamp.now().floor("h"))
    allp = []
    for _, s in sensors().iterrows():
        f = NB / f"{int(s.sensor)}.parquet"
        old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["ts", "y", "n"])
        st = pd.Timestamp(start) if old.empty else old.ts.max() - pd.Timedelta(hours=6)
        new = fetch_sensor(int(s.sensor), st, cut, key)
        df = pd.concat([old, new]).dropna(subset=["ts"]).drop_duplicates("ts", keep="last")
        df = df[(df.ts < cut) & (df.y >= 0) & (df.n.fillna(4) >= 3)]
        df.to_parquet(f)
        print(f"{s['name'][:40]:40s} sensor {int(s.sensor)}: {len(df)} h, last {df.ts.max()}", flush=True)
        allp.append(df.assign(nb_loc=int(s["loc"]), lat=s.lat, lon=s.lon))
    A = pd.concat(allp, ignore_index=True)
    A.to_parquet(EXT / "openaq_neighbors.parquet")
    print("total", len(A), "cutoff", cut)


if __name__ == "__main__":
    run(*sys.argv[1:])
