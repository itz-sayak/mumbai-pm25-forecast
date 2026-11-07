"""Fresh target-station PM2.5 from OpenAQ (same upstream source as Train.csv) to shorten forecast lead.

Sources:  (a) public S3 archive (no key, ~3-4 day lag): 15-min records -> hourly (>=3 readings, hour-ending)
          (b) OpenAQ v3 API /sensors/{id}/hours (needs free key in env OPENAQ_API_KEY), near real time
HARD CUTOFF: no observation with hour-end >= CUTOFF is ever kept (default 2026-11-11 00:00 IST, also capped at
the time of the fetch), so private-window (11-12 Nov) values can never enter the model, even on a later rerun.
Output: data/external/openaq_live.parquet  [ts, location_id, y, n, src, fetched_at]

python src/openaq_live.py [cutoff_iso]
"""
import gzip
import io
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
S3 = "https://openaq-data-archive.s3.amazonaws.com/records/csv.gz/locationid={loc}/year={y}/month={m:02d}/location-{loc}-{d}.csv.gz"
CUTOFF_DEFAULT = "2026-11-11 00:00"


def stations():
    s = pd.read_csv(ROOT / "data" / "raw" / "Stations.csv")
    return list(zip(s.location_id, s.selected_sensor))


def from_s3(loc, sensor, days):
    rows = []
    for day in days:
        u = S3.format(loc=loc, y=day.year, m=day.month, d=day.strftime("%Y%m%d"))
        r = requests.get(u, timeout=60)
        if r.status_code != 200:
            continue
        df = pd.read_csv(io.BytesIO(gzip.decompress(r.content)))
        df = df[(df.parameter == "pm25") & (df.sensors_id == sensor)]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows)
    t = pd.to_datetime(df.datetime.str[:19])  # reading time (IST, interval end)
    df = df.assign(ts=t.dt.ceil("h"), value=pd.to_numeric(df.value, errors="coerce"))
    df = df[df.value >= 0]
    g = df.groupby("ts").value.agg(["mean", "nunique", "size"]).reset_index()
    g = g[g["size"] >= 3]
    return pd.DataFrame({"ts": g.ts, "location_id": loc, "y": g["mean"], "n": g["size"], "src": "s3"})


def from_api(loc, sensor, start, end, key):
    u = f"https://api.openaq.org/v3/sensors/{sensor}/hours"
    out, page = [], 1
    while True:
        r = requests.get(u, headers={"X-API-Key": key}, timeout=60,
                         params={"datetime_from": start.isoformat(), "datetime_to": end.isoformat(),
                                 "limit": 1000, "page": page})
        if r.status_code != 200:
            print("api", loc, r.status_code, r.text[:200])
            break
        res = r.json().get("results", [])
        for x in res:
            p = x.get("period", {})
            te = (p.get("datetimeTo") or {}).get("local")
            cov = x.get("coverage", {}) or {}
            out.append({"ts": pd.Timestamp(te[:19]), "location_id": loc, "y": x.get("value"),
                        "n": cov.get("observedCount"), "src": "api"})
        if len(res) < 1000:
            break
        page += 1
    df = pd.DataFrame(out)
    if len(df):
        df = df[(df.y >= 0) & (df.n.fillna(4) >= 3)]
    return df


def fetch(cutoff=CUTOFF_DEFAULT, lookback_days=10):
    now = datetime.now()
    cut = min(pd.Timestamp(cutoff), pd.Timestamp(now).floor("h"))
    start = cut - pd.Timedelta(days=lookback_days)
    days = pd.date_range(start.normalize(), cut.normalize(), freq="D")
    key = os.environ.get("OPENAQ_API_KEY")
    parts = []
    for loc, sensor in stations():
        parts.append(from_s3(loc, sensor, days))
        if key:
            parts.append(from_api(loc, sensor, start - pd.Timedelta(hours=6), cut + pd.Timedelta(hours=6), key))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df = df[df.ts < cut]  # HARD CUTOFF
    df = df.sort_values("src").drop_duplicates(["ts", "location_id"], keep="first")  # prefer api over s3
    df["fetched_at"] = now.isoformat()
    df.to_parquet(EXT / "openaq_live.parquet")
    print("cutoff", cut, "rows", len(df), "api_key", bool(key))
    print(df.groupby("location_id").ts.agg(["min", "max", "size"]))
    return df


if __name__ == "__main__":
    fetch(sys.argv[1] if len(sys.argv) > 1 else CUTOFF_DEFAULT)
