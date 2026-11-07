"""Experimental member: Microsoft Aurora 0.4 deg Air Pollution (open weights, HF microsoft/aurora).

Initial conditions = the two most recent CAMS global analyses (ADS dataset
cams-global-atmospheric-composition-forecasts, lead time 0, 00/12 UTC) at or before the analysis time; the model
rolls forward in 12-h steps; PM2.5 is sampled bilinearly at the 4 station coordinates and interpolated to hourly.
Needs an Atmosphere Data Store API key in env ADS_KEY (passed per command, never stored).

python src/aurora_ap.py fetch "2026-11-02 00"        # analysis time (UTC, 00 or 12)
python src/aurora_ap.py run   "2026-11-02 00" [steps] # -> data/external/aurora/aurora_<time>.parquet
"""
import os
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "aurora"
WORK = Path(os.environ.get("AURORA_WORK", "/dev/shm/pm25_work/aurora"))  # large NetCDFs live in RAM, not the share
SURF = {"2t": "t2m", "10u": "u10", "10v": "v10", "msl": "msl", "pm1": "pm1", "pm2p5": "pm2p5", "pm10": "pm10",
        "tcco": "tcco", "tc_no": "tc_no", "tcno2": "tcno2", "gtco3": "gtco3", "tcso2": "tcso2"}
ATMOS = ["t", "u", "v", "q", "z", "co", "no", "no2", "go3", "so2"]
VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure",
        "particulate_matter_1um", "particulate_matter_2.5um", "particulate_matter_10um",
        "total_column_carbon_monoxide", "total_column_nitrogen_monoxide", "total_column_nitrogen_dioxide",
        "total_column_ozone", "total_column_sulphur_dioxide", "u_component_of_wind", "v_component_of_wind",
        "temperature", "geopotential", "specific_humidity", "carbon_monoxide", "nitrogen_dioxide",
        "nitrogen_monoxide", "ozone", "sulphur_dioxide"]
LEVELS = ["50", "100", "150", "200", "250", "300", "400", "500", "600", "700", "850", "925", "1000"]


def _tag(t):
    return pd.Timestamp(t).strftime("%Y%m%dT%H")


def fetch(t_an):
    import cdsapi
    t_an = pd.Timestamp(t_an)
    t_prev = t_an - pd.Timedelta(hours=12)
    WORK.mkdir(parents=True, exist_ok=True)
    z = WORK / f"cams_{_tag(t_an)}.zip"
    if not z.exists():
        c = cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api", key=os.environ["ADS_KEY"])
        dates = sorted({t_prev.strftime("%Y-%m-%d"), t_an.strftime("%Y-%m-%d")})
        c.retrieve("cams-global-atmospheric-composition-forecasts",
                   {"type": "forecast", "leadtime_hour": "0", "variable": VARS, "pressure_level": LEVELS,
                    "date": dates[0] if len(dates) == 1 else f"{dates[0]}/{dates[-1]}",
                    "time": ["00:00", "12:00"], "format": "netcdf_zip"}, str(z))
    with zipfile.ZipFile(z) as zf:
        for member, name in (("data_sfc.nc", "sfc"), ("data_plev.nc", "plev")):
            p = WORK / f"cams_{_tag(t_an)}_{name}.nc"
            if not p.exists():
                p.write_bytes(zf.read(member))
    print("fetched", t_prev, t_an, "->", WORK)


def _two_times(ds, t_an):
    ds = ds.isel(forecast_period=0) if "forecast_period" in ds.dims else ds
    tdim = "forecast_reference_time" if "forecast_reference_time" in ds.dims else "valid_time"
    want = [np.datetime64(t_an - pd.Timedelta(hours=12)), np.datetime64(t_an)]
    return ds.sel({tdim: want}), tdim


def run(t_an, steps=8):
    import torch
    import xarray as xr
    from aurora import AuroraAirPollution, Batch, Metadata, rollout
    from huggingface_hub import hf_hub_download
    t_an = pd.Timestamp(t_an)
    sfc, tdim = _two_times(xr.open_dataset(WORK / f"cams_{_tag(t_an)}_sfc.nc", engine="netcdf4",
                                           decode_timedelta=True), t_an)
    atm, _ = _two_times(xr.open_dataset(WORK / f"cams_{_tag(t_an)}_plev.nc", engine="netcdf4",
                                        decode_timedelta=True), t_an)
    static = pickle.load(open(hf_hub_download("microsoft/aurora", "aurora-0.4-air-pollution-static.pickle"), "rb"))
    batch = Batch(
        surf_vars={k: torch.from_numpy(sfc[v].values[None]) for k, v in SURF.items()},
        static_vars={k: torch.from_numpy(v) for k, v in static.items()},
        atmos_vars={k: torch.from_numpy(atm[k].values[None]) for k in ATMOS},
        metadata=Metadata(lat=torch.from_numpy(atm.latitude.values), lon=torch.from_numpy(atm.longitude.values),
                          time=(t_an.to_pydatetime(),),
                          atmos_levels=tuple(int(x) for x in atm.pressure_level.values)))
    model = AuroraAirPollution()
    model.load_checkpoint("microsoft/aurora", "aurora-0.4-air-pollution.ckpt")
    model = model.eval().to("cuda")
    st = pd.read_csv(ROOT / "data" / "raw" / "Stations.csv")
    rows = []
    # the CAMS analysis itself at t_an (step 0) for reference
    a0 = xr.DataArray(sfc["pm2p5"].values[-1], dims=("lat", "lon"),
                      coords={"lat": sfc.latitude.values, "lon": sfc.longitude.values})
    for _, s in st.iterrows():
        rows.append((t_an, 0, s.location_id, float(a0.interp(lat=s.latitude, lon=s.longitude % 360)) * 1e9))
    with torch.inference_mode():
        for k, pred in enumerate(rollout(model, batch, steps=steps), start=1):
            f = pred.surf_vars["pm2p5"][0, 0].cpu().numpy()
            da = xr.DataArray(f, dims=("lat", "lon"), coords={"lat": pred.metadata.lat.cpu().numpy(),
                                                             "lon": pred.metadata.lon.cpu().numpy()})
            vt = t_an + pd.Timedelta(hours=12 * k)
            for _, s in st.iterrows():
                rows.append((vt, k, s.location_id, float(da.interp(lat=s.latitude, lon=s.longitude % 360)) * 1e9))
    D = pd.DataFrame(rows, columns=["valid_utc", "step", "location_id", "pm25_aurora"])
    D["ts_ist"] = (D.valid_utc + pd.Timedelta(hours=5, minutes=30))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"aurora_{_tag(t_an)}.parquet"
    D.to_parquet(p)
    print(D.pivot_table(index="ts_ist", columns="location_id", values="pm25_aurora").round(1).to_string())
    print("wrote", p)


def hourly(t_an):
    """12-hourly station values -> hourly (linear in time), IST hour labels, columns = location_id."""
    D = pd.read_parquet(OUT / f"aurora_{_tag(t_an)}.parquet")
    W = D.pivot_table(index="ts_ist", columns="location_id", values="pm25_aurora")
    full = pd.date_range(W.index.min().ceil("h"), W.index.max().floor("h"), freq="h")
    return W.reindex(W.index.union(full)).interpolate(method="time").reindex(full)


if __name__ == "__main__":
    cmd, t = sys.argv[1], sys.argv[2]
    if cmd == "fetch":
        fetch(t)
    elif cmd == "run":
        run(t, int(sys.argv[3]) if len(sys.argv) > 3 else 8)
