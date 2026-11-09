"""PIBM: physics-informed box model for hourly PM2.5 (a novel, decorrelated ensemble member).

Per station s, hourly mass balance of a well-mixed urban box (dt = 1 h):

    C_{t+1} = C_t + exp(e_s[hod, wk]) / H_t  -  kappa_t * (C_t - B_t)
    kappa_t = 1 - exp(-(u_t / L_s + k_s + lam_s * P_t))          (ventilation + decay + wet scavenging)
    B_t     = beta_s * NB0 * clip(cams_t / cams72_0, .3, 3) ** gamma_s   (regional background)

H = boundary-layer height [km], u = 10 m wind [km/h], P = precipitation [mm/h]; all from forecasts issued
at the origin (lead-matched day-1..3). NB0 = neighbour median (24 h mean) at the origin, C_0 = last obs.
Learnable per station: emission profile e (24 x weekday/weekend), log L, log k, lam, log beta, gamma, log Hmin.
Fit: backprop-through-time on the multi-step (1..HM h) MSE over training origins (Adam, full batch).

python src/pibm.py cv      -> data/processed/exp/pibm/oof_tsfm.parquet  (+ experiments.csv rows)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import TARGETS, load_all, to_grid  # noqa: E402
from features import load_cams, load_weather  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "processed" / "exp"
HM = 84
STN = {str(k): i for i, k in enumerate(TARGETS)}
torch.manual_seed(0)
np.random.seed(0)


def build_tensors(grid, sid_loc, origins_by_sid, wx, cams, nb):
    """Return dict of numpy arrays for all (sid, origin) samples."""
    ts = grid.index
    T = len(ts)
    hod = ts.hour.values
    wk = (ts.dayofweek.values >= 5).astype(int)
    rows = {k: [] for k in ("st", "c0", "nb0", "H", "U", "P", "CR", "hod", "wk", "Y", "sid", "t0")}
    wxa = {L: wx[L].reindex(ts) for L in wx}
    ca = cams.reindex(ts)
    for sid, loc in sid_loc:
        y = grid[sid].values.astype(float)
        lw = loc if ("wind_speed_10m", loc) in wxa[0].columns else "city"
        get = lambda L, v: wxa[L][(v, lw)].values.astype(float)  # noqa: E731
        Hs = {L: get(L, "boundary_layer_height") for L in (1, 2, 3)}
        Us = {L: get(L, "wind_speed_10m") for L in (1, 2, 3)}
        Ps = {L: get(L, "precipitation") for L in (1, 2, 3)}
        cam = ca[loc if loc in ca.columns else "city"].values.astype(float)
        cam72 = pd.Series(cam).rolling(72, min_periods=12).mean().values
        nbm = pd.Series(nb.get(loc, nb["city"])["nb_med"].values.astype(float)).rolling(24, min_periods=6).mean().values
        last = pd.Series(y).ffill().values
        for o in origins_by_sid[sid]:
            fut = np.arange(o + 1, o + HM + 1)
            if fut[-1] >= T or np.isnan(last[o]) or np.isnan(y[o - 2:o + 1]).all():
                continue
            lead = np.minimum(3, np.maximum(1, np.ceil((fut - o) / 24))).astype(int)
            pick = lambda D: np.array([D[L][t] for L, t in zip(lead, fut)])  # noqa: E731
            rows["st"].append(STN[loc]); rows["c0"].append(np.nanmean(y[o - 2:o + 1]))
            rows["nb0"].append(nbm[o] if not np.isnan(nbm[o]) else last[o])
            rows["H"].append(pick(Hs)); rows["U"].append(pick(Us)); rows["P"].append(pick(Ps))
            rows["CR"].append(np.clip(cam[fut] / max(cam72[o], 1), 0.3, 3))
            rows["hod"].append(hod[fut]); rows["wk"].append(wk[fut]); rows["Y"].append(y[fut])
            rows["sid"].append(sid); rows["t0"].append(o)
    out = {k: np.array(v) for k, v in rows.items()}
    # fill weather gaps (e.g. BLH before Sep 2024) with hour-of-day medians
    for k in ("H", "U", "P", "CR"):
        a = out[k].astype(float)
        if np.isnan(a).any():
            med = np.nanmedian(a, axis=0)
            a = np.where(np.isnan(a), med[None, :], a)
            a = np.where(np.isnan(a), np.nanmedian(a), a)
        out[k] = a
    out["H"] = np.maximum(out["H"], 50.0) / 1000.0  # km
    return out


class PIBM(torch.nn.Module):
    def __init__(self, ns=4):
        super().__init__()
        self.e = torch.nn.Parameter(torch.full((ns, 24, 2), float(np.log(2.0))))
        self.logL = torch.nn.Parameter(torch.full((ns,), float(np.log(60.0))))
        self.logk = torch.nn.Parameter(torch.full((ns,), float(np.log(0.03))))
        self.lam = torch.nn.Parameter(torch.full((ns,), 0.1))
        self.logbeta = torch.nn.Parameter(torch.full((ns,), float(np.log(0.8))))
        self.gamma = torch.nn.Parameter(torch.full((ns,), 0.5))
        self.logHmin = torch.nn.Parameter(torch.full((ns,), float(np.log(0.15))))

    def forward(self, d):
        st = d["st"]
        C = d["c0"]
        out = []
        Hmin = torch.exp(self.logHmin)[st][:, None]
        H = torch.maximum(d["H"], Hmin)
        kap = 1 - torch.exp(-(d["U"] / torch.exp(self.logL)[st][:, None] + torch.exp(self.logk)[st][:, None]
                              + torch.relu(self.lam)[st][:, None] * d["P"]))
        B = torch.exp(self.logbeta)[st][:, None] * d["nb0"][:, None] * d["CR"] ** self.gamma[st][:, None]
        E = torch.exp(self.e[st[:, None], d["hod"], d["wk"]])
        for t in range(d["H"].shape[1]):
            C = C + E[:, t] / H[:, t] - kap[:, t] * (C - B[:, t])
            C = torch.clamp(C, min=1.0)
            out.append(C)
        return torch.stack(out, 1)


def to_torch(a, dev):
    return {k: torch.as_tensor(v, device=dev, dtype=torch.long if k in ("st", "hod", "wk") else torch.float32)
            for k, v in a.items() if k not in ("sid", "t0", "Y")}


def fit(a, steps=600, lr=0.03, hfit=52, dev="cpu"):
    m = PIBM().to(dev)
    d = to_torch(a, dev)
    Y = torch.as_tensor(a["Y"][:, :hfit], dtype=torch.float32, device=dev)
    M = ~torch.isnan(Y)
    Y = torch.nan_to_num(Y)
    d = {k: (v[:, :hfit] if v.ndim == 2 else v) for k, v in d.items()}
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for i in range(steps):
        opt.zero_grad()
        p = m(d)
        loss = ((p - Y) ** 2 * M).sum() / M.sum()
        loss = loss + 1e-3 * ((m.e[:, 1:] - m.e[:, :-1]) ** 2).sum()  # smooth emission profile
        loss.backward()
        opt.step()
        if i % 200 == 0:
            print(f"   step {i} rmse {loss.item() ** .5:.2f}", flush=True)
    return m


def predict(m, a, dev="cpu"):
    with torch.no_grad():
        return m(to_torch(a, dev)).cpu().numpy()


def cv():
    import experiments as EX
    from neighbors import build_signals
    from train import FOLDS
    d = load_all()
    d = d[~d.sid.str.startswith("k18")]
    grid = to_grid(d)
    grid = grid.reindex(pd.date_range(grid.index.min(), grid.index.max() + pd.Timedelta(hours=HM), freq="h"))
    wx, cams = load_weather(), load_cams()
    nb = build_signals(d, grid.index)
    locs = d.groupby("sid").location_id.first()
    sid_loc = [(s, str(int(locs[s]))) for s in grid.columns if int(locs[s]) in TARGETS]
    ts = grid.index
    outs = []
    for name, a, b in FOLDS:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        tr_org = {s: np.where((ts.hour.isin([23, 8, 15, 20])) & ((ts + pd.Timedelta(hours=HM) < a) |
                                                                 ((ts >= b + pd.Timedelta(days=4)) & (name != "recent26"))))[0]
                  for s, _ in sid_loc}
        va_org = {s: np.where((ts >= a) & (ts < b) & (ts.hour == 23))[0] for s, _ in sid_loc}
        A = build_tensors(grid, sid_loc, tr_org, wx, cams, nb)
        V = build_tensors(grid, sid_loc, va_org, wx, cams, nb)
        print(name, "train", len(A["st"]), "val", len(V["st"]), flush=True)
        m = fit(A)
        P = predict(m, V)
        for i in range(len(V["st"])):
            fut = np.arange(V["t0"][i] + 1, V["t0"][i] + HM + 1)
            outs.append(pd.DataFrame({"sid": V["sid"][i], "ts0": ts[V["t0"][i]], "ts": ts[fut],
                                      "h": np.arange(1, HM + 1), "pibm": P[i], "fold": name}))
    O = pd.concat(outs, ignore_index=True)
    name = "pibm_hole" if os.environ.get("DIWALI_HOLE") == "1" else "pibm"  # research: Diwali-hole backtest
    (EXP / name).mkdir(parents=True, exist_ok=True)
    O.to_parquet(EXP / name / "oof_tsfm.parquet")
    EX.score(name, O, "pibm", {"model": "box-model"})
    print(O.describe())


if __name__ == "__main__":
    if sys.argv[1] == "cv":
        cv()


def forecast(T0, fcst_tag, origin_ts):
    """Production: fit on all target data <= T0, forecast from each origin in origin_ts (list of Timestamps)."""
    from neighbors import build_signals
    T0 = pd.Timestamp(T0)
    d = load_all()
    d = d[~d.sid.str.startswith("k18") & (d.ts <= T0)]
    grid = to_grid(d)
    grid = grid.reindex(pd.date_range(grid.index.min(), T0 + pd.Timedelta(hours=HM + 1), freq="h"))
    wx, cams = load_weather(fcst_tag), load_cams(fcst_tag)
    nb = build_signals(d, grid.index, upto=T0)
    locs = d.groupby("sid").location_id.first()
    sid_loc = [(s, str(int(locs[s]))) for s in grid.columns if int(locs[s]) in TARGETS]
    ts = grid.index
    tr_org = {s: np.where(ts.hour.isin([23, 8, 15, 20]) & (ts <= T0 - pd.Timedelta(hours=24)))[0] for s, _ in sid_loc}
    pos = set(np.searchsorted(ts, pd.DatetimeIndex(origin_ts)))
    pr_org = {s: np.array(sorted(pos)) for s, l in sid_loc if l == s}  # forecast only the 4 target series
    A = build_tensors(grid, sid_loc, tr_org, wx, cams, nb)
    m = fit(A)
    V = build_tensors(grid, [(s, l) for s, l in sid_loc if s in pr_org], pr_org, wx, cams, nb)
    P = predict(m, V)
    outs = []
    for i in range(len(V["st"])):
        fut = np.arange(V["t0"][i] + 1, V["t0"][i] + HM + 1)
        outs.append(pd.DataFrame({"sid": V["sid"][i], "ts0": ts[V["t0"][i]], "ts": ts[fut],
                                  "h": np.arange(1, HM + 1), "pibm": P[i]}))
    return pd.concat(outs, ignore_index=True)
