"""Load & clean PM2.5 series into a single long table: [sid, location_id, ts, y, src, is_target].

ts = END of hourly interval, naive IST (matches Train.timestamp_ist and Test.target_end_ist).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TARGETS = {11611: "Malad West", 3409511: "Shivaji Nagar", 11606: "Borivali East", 3409510: "Byculla"}
XKDR_NAME = {"Borivali East, Mumbai - IITM": 11606, "Malad West, Mumbai - IITM": 11611,
             "Byculla, Mumbai - BMC": 3409510, "Shivaji Nagar, Mumbai - BMC": 3409511}
# Lakshmi Puja dates (main Diwali night), IST
DIWALI = pd.to_datetime(["2018-11-07", "2019-10-27", "2020-11-14", "2021-11-04", "2022-10-24",
                         "2023-11-12", "2024-11-01", "2025-10-21", "2026-11-08"])
YMAX = 1000.0


def _ts(s):
    return pd.to_datetime(s.astype(str).str[:19].str.replace("T", " "))


def clean_series(df):
    """df: one series sorted by ts with column y. Removes invalid readings (NaN-out)."""
    y = df["y"].astype(float).copy()
    y[(y <= 0) | (y > YMAX)] = np.nan
    # stuck sensor: >=6 identical consecutive values
    run = (y != y.shift()).cumsum()
    rl = y.groupby(run).transform("size")
    y[(rl >= 6) & y.notna()] = np.nan
    # isolated spikes: > 4x both neighbours' median and > 150
    med = y.rolling(7, center=True, min_periods=3).median()
    y[(y > 4 * med) & (y > 150)] = np.nan
    df = df.copy()
    df["y"] = y
    return df


def load_train():
    tr = pd.read_csv(RAW / "Train.csv")
    d = pd.DataFrame({"ts": _ts(tr.timestamp_ist), "y": tr.pm25, "location_id": tr.location_id,
                      "avail": _ts(tr.available_at.fillna(tr.timestamp_ist))})
    d["sid"] = d.location_id.astype(str)
    d["src"] = "train"
    live = ROOT / "data" / "external" / "openaq_live.parquet"
    if os.environ.get("USE_LIVE") == "1" and live.exists():
        L = pd.read_parquet(live)
        L = L.assign(location_id=L.location_id.astype(int))
        last = d.groupby("location_id").ts.max()
        new = (L.ts > L.location_id.map(last)).values  # hours newer than the official release
        gap = np.zeros(len(L), bool)
        if os.environ.get("LIVE_GAPS") == "1":  # also hours the release leaves out (the withheld public window),
            # observed long before the forecast origin; run_all --fill-gaps copies past rows from a Train-only run
            have = pd.MultiIndex.from_frame(d[["location_id", "ts"]])
            gap = ~new & ~pd.MultiIndex.from_frame(L[["location_id", "ts"]]).isin(have)
            if gap.any():
                print(f"[live] filling {int(gap.sum())} hours missing from Train "
                      f"({L.ts[gap].min()} .. {L.ts[gap].max()})")
        L = L[new | gap]
        if len(L):
            L = pd.DataFrame({"ts": L.ts, "y": L.y, "location_id": L.location_id, "avail": L.ts,
                              "sid": L.location_id.astype(str), "src": "train"})
            d = pd.concat([d, L], ignore_index=True)
            print(f"[live] appended {len(L)} OpenAQ hours, up to {L.ts.max()}")
    return d


def load_aux():
    out = []
    x = pd.read_csv(RAW / "AdditionalHistory" / "History_Mumbai_2024.csv.gz")
    x = x[x.source == "cpcb_caaqm"]
    d = pd.DataFrame({"ts": _ts(x.timestamp_ist) + pd.Timedelta(hours=1), "y": x.value,
                      "location_id": x.station_name.map(XKDR_NAME).fillna(-1).astype(int),
                      "sid": "x24_" + x.station_id})
    d["src"] = "x24"
    out.append(d)
    o = pd.read_csv(RAW / "AdditionalHistory" / "Other_Mumbai_Stations_2025_2026.csv.gz")
    d = pd.DataFrame({"ts": _ts(o.timestamp_ist), "y": o["mean"], "location_id": -1,
                      "sid": "oth_" + o.location_id.astype(str)})
    d["src"] = "oth"
    out.append(d)
    h = pd.read_csv(RAW / "AdditionalHistory" / "History_Mumbai_2018_2020.csv.gz")
    d = pd.DataFrame({"ts": _ts(h.Datetime) + pd.Timedelta(hours=1), "y": h["PM2.5"], "location_id": -1,
                      "sid": "k18_" + h.StationId})
    d["src"] = "k18"
    out.append(d)
    return pd.concat(out, ignore_index=True)


def load_all(include_aux=True):
    parts = [load_train()]
    if include_aux:
        parts.append(load_aux())
    d = pd.concat(parts, ignore_index=True)
    d = d.dropna(subset=["ts"]).sort_values(["sid", "ts"])
    d = d.drop_duplicates(["sid", "ts"], keep="last")
    d = pd.concat([clean_series(g) for _, g in d.groupby("sid", sort=True)], ignore_index=True)
    if os.environ.get("DIWALI_HOLE") == "1":  # research only (experiments.py "_hole" configs)
        d = d[~diwali_hole_mask(d)]
    d["is_target"] = d.src.eq("train")
    return d.reset_index(drop=True)


def diwali_hole_mask(d):
    """Target-station hours of Diwali eve + Diwali day in past years: the 2026 public window (7-8 Nov) is withheld
    from Train, so the final run's context has this 48 h hole; this reproduces it in the backtest years."""
    m = np.zeros(len(d), bool)
    tl = d.location_id.isin(list(TARGETS)).values
    for D in DIWALI[DIWALI.year < 2026]:
        m |= tl & (d.ts > D - pd.Timedelta(days=1)).values & (d.ts <= D + pd.Timedelta(days=1)).values
    return m


def to_grid(d):
    """Regular hourly grid per series (NaN where missing). Returns wide DataFrame index=ts, cols=sid."""
    w = d.pivot_table(index="ts", columns="sid", values="y", aggfunc="mean")
    full = pd.date_range(w.index.min().floor("h"), w.index.max().ceil("h"), freq="h")
    return w.reindex(full)


def load_test():
    te = pd.read_csv(RAW / "Test.csv")
    te["ts"] = _ts(te.target_end_ist)
    return te


def diwali_offset_days(ts):
    """Signed days from nearest Lakshmi Puja night (18:00 IST of the date)."""
    ts = pd.DatetimeIndex(ts)
    anchors = DIWALI + pd.Timedelta(hours=18)
    a = anchors.values.astype("datetime64[h]").astype(np.int64)
    t = ts.values.astype("datetime64[h]").astype(np.int64)
    diff = t[:, None] - a[None, :]
    idx = np.abs(diff).argmin(1)
    return diff[np.arange(len(t)), idx] / 24.0


def fireworks_flag(ts):
    """1 on festival nights (18:00-02:00 IST, hour-end) from Dhanteras-ish (LP-2) to Bhai Dooj (LP+2)."""
    ts = pd.DatetimeIndex(ts)
    st = ts - pd.Timedelta(hours=1)  # interval start
    night_date = (st - pd.Timedelta(hours=6)).normalize()  # 18:00..02:00 belongs to the evening's date
    hr = st.hour
    is_night = (hr >= 18) | (hr < 2)
    off = np.full(len(ts), 99.0)
    for D in DIWALI:
        off = np.where(np.abs((night_date - D).days) < np.abs(off), (night_date - D).days, off)
    flag = is_night & (np.abs(off) <= 2)
    # scenario nights (hedge high leg): e.g. FW_EXTRA_NIGHTS=2026-11-11 when Bhai Dooj may fall on Diwali+3
    extra = [e for e in os.environ.get("FW_EXTRA_NIGHTS", "").split(",") if e.strip()]
    if extra:
        flag |= is_night & night_date.isin(pd.to_datetime(extra))
    return flag.astype(np.float32)
