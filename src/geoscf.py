"""NASA GEOS-CF composition forecasts (free, OPeNDAP) as blend members: an independent chemistry-transport model.

* v1 archive (aqc_tavg_1hr_g1440x721_v1, 12z runs, 2019-01 .. 2026-01-02): 2019 rehearsal, fall 2024
* v2 (aqc_tavg_1hr_glo_L1440x721_slv, 09z runs, 2025-08 .. now, incl. the live run): fall 2025, 2026
PM2.5 at RH 35 %, 0.25 deg; bilinear to the station; hourly means centred on whole IST hours are averaged to
hour-ending labels. A run is used for an origin only if it is >= LAG_H old (publication delay; the 9 Nov 09z run was
not out 5.7 h after init, the 8 Nov 09z run was).
Members (OOF format, data/processed/exp/gcf/oof_tsfm.parquet):
  raw   : GEOS-CF as is (NNLS weight acts as the level correction)
  rel   : obs level of the last 24 h x GEOS-CF change vs the same 24 h  (relative design, like relative CAMS)
  rel05 : same with the change damped, (g / g24)^0.5

python src/geoscf.py cv          -> OOF for train.FOLDS (target series)
python src/geoscf.py live <T0>   -> members at origin T0 from the newest published run
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external" / "geoscf"
EXT.mkdir(parents=True, exist_ok=True)
EXP = ROOT / "data" / "processed" / "exp" / "gcf"
V1 = ("https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/fcast/aqc_tavg_1hr_g1440x721_v1/"
      "aqc_tavg_1hr_g1440x721_v1.{d}_12z", "pm25_rh35_gcc", 12)
V2 = ("https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/v2/fcst/aqc_tavg_1hr_glo_L1440x721_slv/"
      "aqc_tavg_1hr_glo_L1440x721_slv.{d}_09z", "pm25_rh35", 9)
V2_START = pd.Timestamp("2025-08-05")
LAT0, LON0 = 436, 1011  # grid cells 19.00/19.25 N x 72.75/73.00 E cover all Mumbai stations
LAG_H = 30
HMAX = 84


def _version(day):
    return V2 if pd.Timestamp(day) >= V2_START else V1


def run_init_ist(day):
    url, var, hz = _version(day)
    return pd.Timestamp(day) + pd.Timedelta(hours=hz, minutes=330)  # UTC init -> IST


def fetch_run(day):
    """120 h x 2 x 2 PM2.5 block of the run initialised on `day` (cached). Returns array (120, 2, 2) or None."""
    day = pd.Timestamp(day).strftime("%Y%m%d")
    f = EXT / f"run_{day}.npy"
    if f.exists():
        a = np.load(f)
        return a if a.size else None
    url, var, _ = _version(day)
    q = f"{url.format(d=day)}.ascii?{var}[0:119][0][{LAT0}:{LAT0 + 1}][{LON0}:{LON0 + 1}]"
    for _ in range(6):
        try:
            r = requests.get(q, timeout=300)
            if r.status_code == 200 and r.text.startswith(var):
                vals = []
                for line in r.text.splitlines()[1:]:
                    if line.startswith("["):
                        vals += [float(x) for x in line.split(",")[1:]]
                a = np.array(vals, dtype=np.float32).reshape(-1, 2, 2)
                np.save(f, a)
                return a
            if r.status_code == 404:
                np.save(f, np.zeros(0, np.float32))  # run does not exist: remember
                return None
            time.sleep(20)  # 5xx / gateway time-out under load: back off, retry
        except Exception:  # noqa
            pass
    return None  # transient failure: not cached, retried next time


def station_series(a, day, lat, lon):
    """Bilinear to (lat, lon); centred on whole IST hours -> hour-ending labels (mean of hours h-1 and h)."""
    wy, wx = (lat - 19.0) / 0.25, (lon - 72.75) / 0.25
    wy, wx = float(np.clip(wy, 0, 1)), float(np.clip(wx, 0, 1))
    v = (a[:, 0, 0] * (1 - wy) * (1 - wx) + a[:, 0, 1] * (1 - wy) * wx + a[:, 1, 0] * wy * (1 - wx)
         + a[:, 1, 1] * wy * wx)
    centre = run_init_ist(day) + pd.Timedelta(minutes=30) + pd.to_timedelta(np.arange(len(v)), "h")
    s = pd.Series(v, index=centre)
    lab = (s + s.shift(1)) / 2  # label h covers (h-1, h]: mean of the values centred at h-1 and h
    return lab.dropna()


def usable_run(t0, lag_h=LAG_H):
    """Newest run day whose init is at least lag_h hours before t0."""
    day = pd.Timestamp(t0).normalize()
    for k in range(5):
        dd = day - pd.Timedelta(days=k)
        if run_init_ist(dd) <= pd.Timestamp(t0) - pd.Timedelta(hours=lag_h):
            return dd
    return day - pd.Timedelta(days=5)


def members(t0, y_hist, lat, lon, run_day=None):
    """y_hist: hourly observed series (index ts) up to t0. Returns DataFrame ts, h, raw, rel, rel05 for (t0, t0+HMAX]."""
    run_day = run_day if run_day is not None else usable_run(t0)
    a = fetch_run(run_day)
    if a is None:
        return None
    g = station_series(a, run_day, lat, lon)
    tgt = pd.date_range(t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=HMAX), freq="h")
    past = pd.date_range(t0 - pd.Timedelta(hours=23), t0, freq="h")
    yo = y_hist.reindex(past)
    gp = g.reindex(past)
    ok = yo.notna() & gp.notna()
    gf = g.reindex(tgt)
    out = pd.DataFrame({"ts": tgt, "h": np.arange(1, HMAX + 1), "raw": gf.values})
    if ok.sum() >= 12 and gp[ok].mean() > 0:
        r = gf.values / gp[ok].mean()
        out["rel"] = yo[ok].mean() * r
        out["rel05"] = yo[ok].mean() * np.sign(r) * np.abs(r) ** 0.5
    else:
        out["rel"] = out["rel05"] = np.nan
    return out


def newest_run(t0, max_back=4):
    """Newest run initialised before t0 that is actually published (live use)."""
    for k in range(max_back + 1):
        day = pd.Timestamp(t0).normalize() - pd.Timedelta(days=k)
        if run_init_ist(day) <= pd.Timestamp(t0):
            f = EXT / f"run_{day.strftime('%Y%m%d')}.npy"
            if f.exists() and np.load(f).size == 0:
                f.unlink()  # a run missing earlier may be published by now
            if fetch_run(day) is not None:
                return day
    return None


def live_members(T0, d, origins):
    """Members for the target stations at the given origins (predict.py). d: target observations (sid, ts, y).
    The run is the newest published one before each origin."""
    C = _coords()
    rows = []
    for sid in sorted(d.sid.unique()):
        loc = int(sid)
        if loc not in C:
            continue
        y = d[d.sid == sid].drop_duplicates("ts").set_index("ts").y
        for t0 in origins:
            t0 = pd.Timestamp(t0)
            rd = newest_run(t0)
            if rd is None:
                continue
            m = members(t0, y[y.index <= t0], *C[loc], run_day=rd)
            if m is not None:
                m["sid"], m["ts0"] = sid, t0
                rows.append(m)
    if not rows:
        return None
    O = pd.concat(rows, ignore_index=True)
    print(f"GEOS-CF members: {len(O)} rows, run for T0 {newest_run(T0)}", flush=True)
    return O.rename(columns={c: f"gcf__{c}" for c in ("raw", "rel", "rel05")})[
        ["sid", "ts0", "ts", "h", "gcf__raw", "gcf__rel", "gcf__rel05"]]


def _coords():
    st = pd.read_csv(ROOT / "data" / "raw" / "Stations.csv").set_index("location_id")
    return {int(k): (float(v.latitude), float(v.longitude)) for k, v in st.iterrows()}


def cv(folds=None):
    from train import FOLDS
    folds = folds or FOLDS
    d = D.load_all()
    d = d[d.location_id.isin(list(D.TARGETS))]
    C = _coords()
    sids = d.groupby("sid").location_id.first()
    rows = []
    for name, a, b in folds:
        org = pd.date_range(pd.Timestamp(a) + pd.Timedelta(hours=23), pd.Timestamp(b), freq="24h")
        org = org[org <= d.ts.max()]
        days = sorted({usable_run(t) for t in org})
        with ThreadPoolExecutor(2) as ex:  # the OPeNDAP server times out (504) under heavier parallel load
            list(ex.map(fetch_run, days))
        print(name, len(org), "origins,", sum(fetch_run(x) is not None for x in days), "/", len(days), "runs", flush=True)
        for sid, loc in sids.items():
            y = d[d.sid == sid].set_index("ts").y
            y = y[~y.index.duplicated()]
            if y.index.max() < pd.Timestamp(a) or y.index.min() > pd.Timestamp(b):
                continue
            lat, lon = C[int(loc)]
            for t0 in org:
                m = members(t0, y[y.index <= t0], lat, lon)
                if m is None:
                    continue
                m["sid"], m["ts0"], m["fold"] = sid, t0, name
                m["y"] = y.reindex(m.ts).values
                rows.append(m)
    O = pd.concat(rows, ignore_index=True)
    EXP.mkdir(parents=True, exist_ok=True)
    O.to_parquet(EXP / "oof_tsfm.parquet")
    r = lambda g, c: float(np.sqrt(np.nanmean((g.y - g[c]) ** 2)))
    print(O.dropna(subset=["y"]).groupby("fold").apply(lambda g: pd.Series({c: r(g, c) for c in ("raw", "rel", "rel05")})).round(2))
    return O


if __name__ == "__main__":
    if sys.argv[1] == "cv":
        cv()
