"""Meteorology figures for the 2026 Diwali smoke episode in Mumbai (inputs from viz/fetch.py, boundaries from datameet).

python viz/make_figures.py <boundaries_dir>      # dir with datameet Admin2.shp (states) and 2011_Dist.shp (districts)
Writes docs/img/*.png and docs/img/mumbai_diwali_night.gif
"""
import sys
from pathlib import Path

import geopandas as gpd
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import shapely

HERE = Path(__file__).resolve().parent
CACHE, OUT = HERE / "cache", HERE.parent / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
GEO = Path(sys.argv[1])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.edgecolor": "#888", "axes.linewidth": 0.6, "figure.dpi": 110, "savefig.dpi": 160})
INK, MUTED, SEA, LAND = "#222", "#666", "#dfe9f2", "#f4f2ee"
DIWALI = pd.Timestamp("2026-11-08")

# ---------- colour scales -------------------------------------------------------------------------------------------
PM_LEV = [0, 15, 30, 45, 60, 75, 90, 105, 120, 150, 180, 250, 350]
PM_CMAP = mcolors.LinearSegmentedColormap.from_list("pm", ["#fbf7e9", "#f6dc8a", "#f3a65a", "#e2663d", "#b8303f", "#7a1650", "#3b0f45"])
PM_NORM = mcolors.BoundaryNorm(PM_LEV, PM_CMAP.N, extend="max")
CPCB = [(30, "Good"), (60, "Satisf."), (90, "Moderate"), (120, "Poor"), (250, "Very poor")]
BLH_LEV = [0, 100, 200, 300, 400, 600, 800, 1000, 1300, 1600, 2000, 2500]
BLH_CMAP = plt.get_cmap("YlGnBu")
VC_LEV = [0, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000]
VC_CMAP = mcolors.ListedColormap(["#7f2704", "#b5450b", "#de6b1e", "#f4a259", "#fbd9a8", "#cfe3ef", "#8fbcdc", "#4a8fc1", "#1f5f9a"])


def pm_colorbar(fig, mappable, ax, **kw):
    cb = fig.colorbar(mappable, ax=ax, ticks=[0, 30, 60, 90, 120, 180, 250, 350], **kw)
    cb.set_label("PM2.5 (µg/m³)  ·  ticks = CPCB 24-h category limits", color=MUTED)
    cb.outline.set_linewidth(0.4)
    return cb


# ---------- data ----------------------------------------------------------------------------------------------------
def cube(name):
    z = np.load(CACHE / f"{name}.npz")
    return pd.to_datetime(z["time"]), z["lat"], z["lon"], {k: z[k] for k in z.files if k not in ("time", "lat", "lon")}


def uv(ws, wd):
    r = np.deg2rad(wd)
    return -ws * np.sin(r), -ws * np.cos(r)


states = gpd.read_file(GEO / "Admin2.shp")
dist = gpd.read_file(GEO / "2011_Dist.shp")
mh = states[states.ST_NM == "Maharashtra"]
mh_d = dist[dist.ST_NM == "Maharashtra"]
mum_land = dist[dist.DISTRICT.isin(["Mumbai", "Mumbai Suburban", "Thane", "Raigarh"]) & (dist.ST_NM == "Maharashtra")]
mon = pd.read_csv(HERE / "monitors.csv")
obs = pd.read_csv(CACHE / "monitors.csv", parse_dates=["ts"])
obs = obs.merge(mon[["loc", "short", "lat", "lon", "target"]], on="loc")

t_c, la_c, lo_c, C = cube("mh_cams")
t_w, la_w, lo_w, W = cube("mh_ifs")
t_b, la_b, lo_b, B = cube("mh_blh")
t_m, la_m, lo_m, M = cube("mum_ifs")
t_mb, la_mb, lo_mb, MB = cube("mum_blh")


def tmean(t, a, s, e):
    k = (t >= pd.Timestamp(s)) & (t <= pd.Timestamp(e))
    return np.nanmean(a[k], axis=0)


def mh_axes(ax, title):
    ax.set_facecolor(SEA)
    states.plot(ax=ax, facecolor="none", edgecolor="#555", linewidth=0.7, zorder=4)
    mh_d.plot(ax=ax, facecolor="none", edgecolor="#999", linewidth=0.25, zorder=4)
    mh.plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=5)
    for name, (x, y) in {"GUJARAT": (72.9, 21.9), "MADHYA PRADESH": (77.2, 22.15), "TELANGANA": (79.0, 18.2),
                         "KARNATAKA": (75.6, 15.9), "Arabian Sea": (72.9, 17.4)}.items():
        ax.text(x, y, name, fontsize=7.5, color=MUTED, style="italic" if "Sea" in name else "normal", zorder=6)
    for name, (x, y) in {"Mumbai": (72.88, 19.07), "Pune": (73.86, 18.52), "Nagpur": (79.09, 21.15), "Nashik": (73.79, 19.99),
                         "Aurangabad": (75.34, 19.88), "Solapur": (75.91, 17.66), "Kolhapur": (74.24, 16.70)}.items():
        ax.plot(x, y, "o", ms=3.5 if name != "Mumbai" else 6, mfc="white", mec=INK, mew=1, zorder=7)
        ax.text(x + 0.12, y + 0.08, name, fontsize=8 if name != "Mumbai" else 9, weight="bold" if name == "Mumbai" else "normal",
                color=INK, zorder=7, path_effects=[matplotlib.patheffects.withStroke(linewidth=2.2, foreground="white")])
    ax.set_xlim(72.4, 81.0); ax.set_ylim(15.6, 22.4); ax.set_aspect("equal")
    ax.set_title(title, loc="left"); ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}°E"); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}°N")


import matplotlib.patheffects  # noqa: E402

# ---------- Fig 1: CAMS PM2.5 + ECMWF winds over Maharashtra --------------------------------------------------------
periods = [("Ordinary night · 1–2 Nov 20:00–06:00", "2026-11-01 20:00", "2026-11-02 06:00"),
           ("Diwali night · 8–9 Nov 20:00–06:00", "2026-11-08 20:00", "2026-11-09 06:00"),
           ("Scored window · 11–12 Nov, all hours", "2026-11-11 01:00", "2026-11-13 00:00")]
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.75), constrained_layout=True)
for ax, (ttl, s, e) in zip(axs, periods):
    pm = tmean(t_c, C["pm2_5"], s, e)
    cf = ax.contourf(lo_c, la_c, pm, levels=PM_LEV, cmap=PM_CMAP, norm=PM_NORM, extend="max", alpha=0.92, zorder=2)
    ax.contour(lo_c, la_c, pm, levels=[60, 90, 120], colors="#3b0f45", linewidths=0.5, alpha=0.6, zorder=3)
    ws, wd = W["wind_speed_10m"], W["wind_direction_10m"]
    k = (t_w >= pd.Timestamp(s)) & (t_w <= pd.Timestamp(e))
    uu, vv = uv(ws[k], wd[k]); uu, vv = np.nanmean(uu, 0), np.nanmean(vv, 0)
    q = ax.quiver(lo_w[::2], la_w[::2], uu[::2, ::2], vv[::2, ::2], scale=38, width=0.0034, color="#1d2b3a", alpha=0.85, zorder=6)
    mh_axes(ax, ttl)
ax.quiverkey(q, 0.86, -0.09, 3, "3 m/s", labelpos="E", coordinates="axes", fontproperties={"size": 8})
pm_colorbar(fig, cf, axs, shrink=0.85, pad=0.01)
fig.suptitle("CAMS global PM2.5 (ECMWF/Copernicus, 0.4°) with ECMWF IFS 10 m wind — Maharashtra, Diwali 2026",
             x=0.01, ha="left", weight="bold", fontsize=12.5)
fig.savefig(OUT / "fig1_maharashtra_cams_pm25_wind.png"); plt.close(fig)

# ---------- Fig 2: boundary layer and ventilation ------------------------------------------------------------------
ws_b = np.stack([W["wind_speed_10m"][list(t_w).index(t)] if t in set(t_w) else np.full(W["wind_speed_10m"].shape[1:], np.nan) for t in t_b])
VC = B["boundary_layer_height"] * ws_b
fig, axs = plt.subplots(2, 2, figsize=(11.8, 9.6), constrained_layout=True)
for j, (lab, tt) in enumerate([("Diwali 15:00 IST", "2026-11-08 15:00"), ("Diwali night 04:00 IST", "2026-11-09 04:00")]):
    i = list(t_b).index(pd.Timestamp(tt))
    cf = axs[0, j].contourf(lo_b, la_b, B["boundary_layer_height"][i], levels=BLH_LEV, cmap=BLH_CMAP, extend="max", zorder=2)
    mh_axes(axs[0, j], f"Mixing height · {lab}")
    cv = axs[1, j].contourf(lo_b, la_b, VC[i], levels=VC_LEV, cmap=VC_CMAP, extend="max", zorder=2)
    axs[1, j].contour(lo_b, la_b, VC[i], levels=[6000], colors=INK, linewidths=1.4, zorder=3)
    mh_axes(axs[1, j], f"Ventilation coefficient · {lab}")
cb = fig.colorbar(cf, ax=axs[0, :], shrink=0.9, pad=0.01); cb.set_label("GFS boundary-layer height (m)", color=MUTED)
cb = fig.colorbar(cv, ax=axs[1, :], shrink=0.9, pad=0.01)
cb.set_label("BLH × 10 m wind (m²/s); black line = 6000 'poor ventilation' limit", color=MUTED)
fig.suptitle("The lid comes down at night: mixing height and ventilation over Maharashtra, Diwali 2026",
             x=0.01, ha="left", weight="bold", fontsize=12.5)
fig.savefig(OUT / "fig2_mixing_height_ventilation.png"); plt.close(fig)

# ---------- Mumbai observed field (IDW over land) -------------------------------------------------------------------
GX = np.arange(72.76, 73.14, 0.004); GY = np.arange(18.88, 19.33, 0.004)
gx, gy = np.meshgrid(GX, GY)
land = shapely.contains_xy(mum_land.union_all(), gx, gy)
piv = obs.pivot_table(index="ts", columns="loc", values="pm25")
locs = mon.set_index("loc").loc[piv.columns]
D2 = (gx[..., None] - locs.lon.values) ** 2 * np.cos(np.deg2rad(19)) ** 2 + (gy[..., None] - locs.lat.values) ** 2


def idw(row):
    ok = np.isfinite(row.values)
    if ok.sum() < 8:
        return None
    w = 1.0 / (D2[..., ok] + 0.012 ** 2) ** 1.0  # inverse squared distance, 1.3 km smoothing length
    z = (w * row.values[ok]).sum(-1) / w.sum(-1)
    return np.where(land, z, np.nan)


def mum_axes(ax):
    ax.set_facecolor(SEA)
    mum_land.plot(ax=ax, facecolor=LAND, edgecolor="none", zorder=1)
    mum_land.plot(ax=ax, facecolor="none", edgecolor="#777", linewidth=0.5, zorder=4)
    ax.set_xlim(GX[0], GX[-1]); ax.set_ylim(GY[0], GY[-1]); ax.set_aspect(1 / np.cos(np.deg2rad(19)))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for name, (x, y) in {"Arabian Sea": (72.775, 19.0), "Thane Creek": (72.965, 19.035)}.items():
        ax.text(x, y, name, fontsize=7.5, color="#5b7a99", style="italic", rotation=90 if "Sea" in name else 0, zorder=6)


def mum_wind(ax, ts):
    if ts not in set(t_m):
        return
    i = list(t_m).index(ts)
    u, v = uv(M["wind_speed_10m"][i], M["wind_direction_10m"][i])
    ax.quiver(lo_m[::2], la_m[::2], u[::2, ::2], v[::2, ::2], scale=30, width=0.004, color="#1d2b3a", alpha=0.8, zorder=7)


def mum_panel(ax, ts, title, label_targets=True):
    mum_axes(ax)
    z = idw(piv.loc[ts]) if ts in piv.index else None
    cf = None
    if z is not None:
        cf = ax.contourf(gx, gy, z, levels=PM_LEV, cmap=PM_CMAP, norm=PM_NORM, extend="max", zorder=2)
    r = piv.loc[ts] if ts in piv.index else pd.Series(dtype=float)
    for loc_, m in locs.iterrows():
        val = r.get(loc_, np.nan)
        if np.isfinite(val):
            ax.scatter(m.lon, m.lat, s=34 if m.target else 16, c=[val], cmap=PM_CMAP, norm=PM_NORM, edgecolors=INK,
                       linewidths=1.3 if m.target else 0.5, marker="s" if m.target else "o", zorder=8)
        if label_targets and m.target:
            ax.text(m.lon + 0.008, m.lat - 0.004, m.short, fontsize=7.5, weight="bold", color=INK, zorder=9,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground="white")])
    mum_wind(ax, ts)
    ax.set_title(title, loc="left")
    return cf


snaps = [("Diwali eve, 8 Nov 18:00", "2026-11-08 18:00"), ("Diwali night, 9 Nov 02:00", "2026-11-09 02:00"),
         ("Next afternoon, 9 Nov 15:00", "2026-11-09 15:00"), ("Scored day, 12 Nov 02:00", "2026-11-12 02:00")]
fig, axs = plt.subplots(1, 4, figsize=(16, 5.6), constrained_layout=True)
for ax, (ttl, ts) in zip(axs, snaps):
    cf = mum_panel(ax, pd.Timestamp(ts), ttl, label_targets=(ax is axs[0]))
pm_colorbar(fig, cf, axs, shrink=0.8, pad=0.01)
fig.suptitle(f"Mumbai PM2.5 from {piv.shape[1]} monitors (CPCB / MPCB / IITM / BMC via OpenAQ, inverse-distance field over land) "
             "+ ECMWF IFS 10 m wind;  squares = the 4 forecast stations", x=0.01, ha="left", weight="bold", fontsize=11.5)
fig.savefig(OUT / "fig3_mumbai_monitor_field.png"); plt.close(fig)

# ---------- GIF: Diwali night, hourly -------------------------------------------------------------------------------
frames = []
city = piv.mean(axis=1)
hours = pd.date_range("2026-11-08 12:00", "2026-11-10 12:00", freq="h")
tmp = CACHE / "frame.png"
for ts in hours:
    fig = plt.figure(figsize=(7.2, 8.0))
    ax = fig.add_axes([0.03, 0.25, 0.80, 0.68]); cax = fig.add_axes([0.85, 0.30, 0.025, 0.58])
    cf = mum_panel(ax, ts, "")
    ax.set_title(f"Mumbai PM2.5 · {ts:%a %d %b %H:%M} IST", loc="left", fontsize=12)
    if cf is not None:
        cb = fig.colorbar(cf, cax=cax, ticks=[0, 30, 60, 90, 120, 180, 250, 350]); cb.set_label("µg/m³", color=MUTED)
    bx = fig.add_axes([0.08, 0.06, 0.86, 0.14])
    win = city.loc["2026-11-08 12:00":"2026-11-10 12:00"]
    bx.plot(win.index, win.values, color="#7a1650", lw=1.6)
    bx.axvline(ts, color=INK, lw=1)
    bx.set_ylabel("city mean", color=MUTED, fontsize=8); bx.tick_params(labelsize=7.5, colors=MUTED)
    bx.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0, 12]))
    bx.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b %H:%M"))
    for sp in ("top", "right"): bx.spines[sp].set_visible(False)
    fig.text(0.03, 0.965, "monitors (dots), IDW field over land, ECMWF IFS 10 m wind", fontsize=8, color=MUTED)
    fig.savefig(tmp, dpi=90); plt.close(fig)
    frames.append(imageio.imread(tmp))
imageio.mimsave(OUT / "mumbai_diwali_night.gif", frames, duration=0.35, loop=0)

# ---------- Fig 4: hour x date heatmaps (PM2.5, mixing height) ------------------------------------------------------
d0, d1 = pd.Timestamp("2026-10-28"), pd.Timestamp("2026-11-12 23:00")
c = city.loc[d0:d1]
H = pd.DataFrame({"v": c.values, "d": c.index.normalize(), "h": c.index.hour}).pivot_table(index="h", columns="d", values="v")
iy, ix = np.argmin(np.abs(la_mb - 19.07)), np.argmin(np.abs(lo_mb - 72.87))
blh = pd.Series(MB["boundary_layer_height"][:, iy, ix], index=t_mb).loc[d0:d1]
HB = pd.DataFrame({"v": blh.values, "d": blh.index.normalize(), "h": blh.index.hour}).pivot_table(index="h", columns="d", values="v")
wsm = pd.Series(M["wind_speed_10m"][:, iy, ix], index=t_m).loc[d0:d1]
wdm = pd.Series(M["wind_direction_10m"][:, iy, ix], index=t_m).loc[d0:d1]
fig, axs = plt.subplots(2, 1, figsize=(13, 7.4), sharex=True, constrained_layout=True)
ext = [-0.5, H.shape[1] - 0.5, 23.5, -0.5]
im = axs[0].imshow(H.values, aspect="auto", cmap=PM_CMAP, norm=PM_NORM, extent=ext, interpolation="nearest")
pm_colorbar(fig, im, axs[0], pad=0.01)
axs[0].set_title("Mumbai city-mean PM2.5 (monitor average), hour of day × date", loc="left")
im2 = axs[1].imshow(HB.values, aspect="auto", cmap=BLH_CMAP, extent=ext, interpolation="nearest", vmin=0, vmax=1600)
cb = fig.colorbar(im2, ax=axs[1], pad=0.01); cb.set_label("GFS boundary-layer height (m)", color=MUTED)
W2 = pd.DataFrame({"u": uv(wsm.values, wdm.values)[0], "v": uv(wsm.values, wdm.values)[1], "d": wsm.index.normalize(), "h": wsm.index.hour})
for (dd, hh), g in W2[W2.h % 3 == 0].groupby(["d", "h"]):
    if dd in HB.columns:
        x = list(HB.columns).index(dd)
        axs[1].quiver(x, hh, g.u.iloc[0], -g.v.iloc[0], scale=60, width=0.0022, color=INK, alpha=0.8)  # y axis inverted
axs[1].set_title("Mixing height with ECMWF 10 m wind every 3 h (arrow = direction wind blows towards; north = up)", loc="left")
for ax in axs:
    ax.set_yticks([0, 3, 6, 9, 12, 15, 18, 21]); ax.set_ylabel("hour (IST)")
    dl = list(H.columns)
    ax.set_xticks(range(len(dl))); ax.set_xticklabels([f"{d:%d %b}" for d in dl], rotation=60, fontsize=7.5)
    for name, dd in [("Diwali", "2026-11-08"), ("scored", "2026-11-11"), ("scored", "2026-11-12")]:
        x = dl.index(pd.Timestamp(dd)); ax.axvline(x - 0.5, color=INK, lw=0.9, ls="--")
        if ax is axs[0]:
            ax.text(x - 0.4, -1.0, name, fontsize=8, weight="bold", va="bottom")
fig.savefig(OUT / "fig4_diurnal_heatmaps.png"); plt.close(fig)

# ---------- Fig 5: ventilation vs PM2.5 -----------------------------------------------------------------------------
vc = (blh * wsm.reindex(blh.index)).rename("vc")
df = pd.concat([c.rename("pm"), vc], axis=1).dropna()
fest = (df.index >= "2026-11-06 18:00") & (df.index <= "2026-11-10 08:00")
fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)
ax.scatter(df.vc[~fest], df.pm[~fest], s=14, color="#4a8fc1", alpha=0.7, edgecolors="white", linewidths=0.3, label="ordinary hours")
ax.scatter(df.vc[fest], df.pm[fest], s=22, color="#b8303f", alpha=0.85, edgecolors="white", linewidths=0.3, label="festival hours (6–10 Nov)")
ax.set_xscale("log"); ax.axvline(6000, color=INK, lw=0.9, ls="--"); ax.text(6300, ax.get_ylim()[1] * 0.93, "6000 m²/s\n'poor' limit", fontsize=8)
lx = np.log(df.vc[~fest]); b = np.polyfit(lx, df.pm[~fest], 1)
xs = np.geomspace(df.vc.min(), df.vc.max(), 50); ax.plot(xs, b[0] * np.log(xs) + b[1], color="#1f5f9a", lw=1.5)
r_ord = np.corrcoef(lx, df.pm[~fest])[0, 1]
ax.set_xlabel("ventilation coefficient, BLH × 10 m wind (m²/s, log scale)"); ax.set_ylabel("Mumbai city-mean PM2.5 (µg/m³)")
ax.set_title(f"Less ventilation, more smoke (ordinary hours r = {r_ord:.2f} with log VC)", loc="left")
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86))
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.savefig(OUT / "fig5_ventilation_vs_pm25.png"); plt.close(fig)
print("ventilation r", round(r_ord, 2), "| festival excess over fit", round(float((df.pm[fest] - (b[0] * np.log(df.vc[fest]) + b[1])).mean()), 1))

# ---------- Fig 6: forecast vs observed in the scored window ---------------------------------------------------------
pw = pd.read_csv(HERE / "private_window.csv", parse_dates=["ts"])
lab = {"y": ("observed", INK, 1.8, "-"), "hi": ("our forecast, high leg (counted)", "#7a1650", 1.6, "-"),
       "lo": ("our forecast, low leg", "#e2663d", 1.3, "-"), "naive7": ("7-day same-hour mean", "#4a8fc1", 1.3, "--")}
fig, axs = plt.subplots(2, 2, figsize=(13, 7.0), sharex=True, constrained_layout=True)
for ax, (st, g) in zip(axs.ravel(), pw.groupby("station")):
    for k, (nm, col, lw, ls) in lab.items():
        ax.plot(g.ts, g[k], color=col, lw=lw, ls=ls, label=nm)
    r = lambda c: np.sqrt(np.nanmean((g[c] - g.y) ** 2))
    ax.set_title(f"{st}   RMSE high {r('hi'):.0f} · low {r('lo'):.0f} · 7-day {r('naive7'):.0f}", loc="left")
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b\n%H:%M")); ax.tick_params(labelsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_ylabel("PM2.5 (µg/m³)")
axs[0, 0].legend(frameon=False, fontsize=8, ncol=2)
fig.suptitle("Scored window, 11–12 Nov 2026 (Diwali +3/+4): forecasts issued 10 Nov 20:00 vs observed (OpenAQ)",
             x=0.01, ha="left", weight="bold", fontsize=12)
fig.savefig(OUT / "fig6_scored_window.png"); plt.close(fig)
print("done", sorted(p.name for p in OUT.iterdir()))
