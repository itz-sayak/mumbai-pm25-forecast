"""Fetch + cache exogenous covariates from Open-Meteo (free, no key; listed on ClimateVerse).

* Weather: Previous-Runs API -> as-issued forecasts at lead day 0..3 (training & backtests),
           Forecast API -> current forecast (inference).
* CAMS global PM2.5 etc: Air-Quality API (past = stitched short-lead forecasts; future = forecast).

Raw JSON is cached under data/external/ with the retrieval time, so code review can rerun
from the exact inputs used for a submission.
"""
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
EXT.mkdir(parents=True, exist_ok=True)

WVARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "surface_pressure",
         "cloud_cover", "wind_speed_10m", "wind_direction_10m", "boundary_layer_height", "shortwave_radiation"]
LEADS = [1, 2, 3]
AQVARS = ["pm2_5", "pm10", "dust", "carbon_monoxide", "aerosol_optical_depth"]
TZ = "Asia%2FKolkata"


def stations():
    s = pd.read_csv(ROOT / "data" / "raw" / "Stations.csv")
    return s[["location_id", "latitude", "longitude"]].values.tolist()


def _get(url, tries=8):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code == 200:
                return r.json()
            print("HTTP", r.status_code, r.text[:200], flush=True)
            if r.status_code == 429:  # minutely limit: wait it out
                time.sleep(65)
                continue
        except Exception as e:  # noqa
            print("err", e, flush=True)
        time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def _to_df(js, prefix=""):
    h = js["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_prev_runs(loc, lat, lon, start, end):
    hv = ",".join(WVARS + [f"{v}_previous_day{d}" for v in WVARS for d in LEADS])
    url = (f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly={hv}&start_date={start}&end_date={end}&timezone={TZ}")
    return _get(url)


def fetch_forecast(lat, lon, past_days=7, forecast_days=7):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly={','.join(WVARS)}"
           f"&past_days={past_days}&forecast_days={forecast_days}&timezone={TZ}")
    return _get(url)


def fetch_aq(lat, lon, start=None, end=None, past_days=None, forecast_days=None):
    base = (f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
            f"&hourly={','.join(AQVARS)}&timezone={TZ}&domains=cams_global")
    if start:
        base += f"&start_date={start}&end_date={end}"
    else:
        base += f"&past_days={past_days}&forecast_days={forecast_days}"
    return _get(base)


def yearly_chunks(start, end):
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    while s <= e:
        ce = min(date(s.year, 12, 31), e)
        yield s.isoformat(), ce.isoformat()
        s = ce + timedelta(days=1)


def build_history(start="2024-01-01", end=None):
    """Weather previous-runs + CAMS history -> parquet per kind."""
    end = end or (date.today() - timedelta(days=1)).isoformat()
    W, A = [], []
    for loc, lat, lon in stations():
        for s, e in yearly_chunks(start, end):
            fw = EXT / f"prevruns_{loc}_{s}_{e}.json"
            if not fw.exists() or e >= (date.today() - timedelta(days=10)).isoformat():
                js = fetch_prev_runs(loc, lat, lon, s, e)
                js["_retrieved"] = datetime.now().isoformat()
                fw.write_text(json.dumps(js))
            d = _to_df(json.loads(fw.read_text()))
            d["location_id"] = loc
            W.append(d)
            fa = EXT / f"cams_{loc}_{s}_{e}.json"
            if not fa.exists() or e >= (date.today() - timedelta(days=10)).isoformat():
                js = fetch_aq(lat, lon, start=s, end=e)
                js["_retrieved"] = datetime.now().isoformat()
                fa.write_text(json.dumps(js))
            d = _to_df(json.loads(fa.read_text()))
            d["location_id"] = loc
            A.append(d)
            print("hist", loc, s, e)
    pd.concat(W).to_parquet(EXT / "weather_prevruns.parquet")
    pd.concat(A).to_parquet(EXT / "cams_hist.parquet")


def snapshot_forecast(tag=None):
    """Current forecasts (issued now) — saved per snapshot tag for reproducibility."""
    tag = tag or datetime.now().strftime("%Y%m%dT%H%M")
    W, A = [], []
    for loc, lat, lon in stations():
        js = fetch_forecast(lat, lon)
        js["_retrieved"] = datetime.now().isoformat()
        (EXT / f"fcst_{tag}_{loc}.json").write_text(json.dumps(js))
        d = _to_df(js)
        d["location_id"] = loc
        W.append(d)
        js = fetch_aq(lat, lon, past_days=7, forecast_days=5)
        js["_retrieved"] = datetime.now().isoformat()
        (EXT / f"camsfc_{tag}_{loc}.json").write_text(json.dumps(js))
        d = _to_df(js)
        d["location_id"] = loc
        A.append(d)
    pd.concat(W).to_parquet(EXT / f"weather_fcst_{tag}.parquet")
    pd.concat(A).to_parquet(EXT / f"cams_fcst_{tag}.parquet")
    print("snapshot", tag)
    return tag





# ----------------------------------------------------------------------------- multi-NWP (ensemble mean/spread)
NWP_MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "ecmwf_aifs025_single"]
MVARS = ["boundary_layer_height", "wind_speed_10m", "wind_direction_10m", "relative_humidity_2m",
         "precipitation", "temperature_2m"]


def fetch_prev_runs_multi(lat, lon, start, end):
    hv = ",".join(MVARS + [f"{v}_previous_day{d}" for v in MVARS for d in LEADS])
    url = (f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly={hv}&start_date={start}&end_date={end}&timezone={TZ}&models={','.join(NWP_MODELS)}")
    return _get(url)


def build_multinwp_history(start="2024-01-01", end=None):
    end = end or (date.today() - timedelta(days=1)).isoformat()
    W = []
    for loc, lat, lon in stations():
        for s, e in yearly_chunks(start, end):
            f = EXT / f"mnwp_{int(loc)}_{s}_{e}.json"
            if not f.exists() or e >= (date.today() - timedelta(days=10)).isoformat():
                js = fetch_prev_runs_multi(lat, lon, s, e)
                js["_retrieved"] = datetime.now().isoformat()
                f.write_text(json.dumps(js))
                time.sleep(2)
            d = _to_df(json.loads(f.read_text()))
            d["location_id"] = int(loc)
            W.append(d)
            print("mnwp", int(loc), s, e, flush=True)
    pd.concat(W).to_parquet(EXT / "weather_multinwp.parquet")


def snapshot_multinwp(tag):
    W = []
    for loc, lat, lon in stations():
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly={','.join(MVARS)}"
               f"&past_days=7&forecast_days=7&timezone={TZ}&models={','.join(NWP_MODELS)}")
        js = _get(url)
        js["_retrieved"] = datetime.now().isoformat()
        (EXT / f"mnwpfc_{tag}_{int(loc)}.json").write_text(json.dumps(js))
        d = _to_df(js)
        d["location_id"] = int(loc)
        W.append(d)
    pd.concat(W).to_parquet(EXT / f"weather_mnwp_fcst_{tag}.parquet")
    print("mnwp snapshot", tag)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "hist"):
        build_history()  # multi-NWP (E2) was rejected by the ablations -> not part of the default pipeline
    if what in ("all", "fcst"):
        snapshot_forecast(sys.argv[2] if len(sys.argv) > 2 else None)
    if what == "mnwp":
        build_multinwp_history()
        if len(sys.argv) > 2:
            snapshot_multinwp(sys.argv[2])
