"""Two-pick hedge. The challenge scores the BEST of the 2 selected submissions on the private window, so the pair is
chosen as two multiplicative legs of the blend, a*p and b*p, minimising the expected best-of-two window RMSE

    (a, b) = argmin  E_w[ min( RMSE_w(y, a p), RMSE_w(y, b p) ) ]

over backtest windows w (one forecast origin, all stations, the private window's lead range). It is a two-point
quantisation of the uncertain window-level multiplier y/p under the competition's own loss. Optimising squared
error instead (2-means) chases spike windows (centroids ~1.0 / ~1.5) and does not help the recent fold, so the
objective is the window RMSE itself. Selection is leave-one-fold-out on clean OOF blends; legs are kept only if
every fold improves (else a = b = 1).

Festival regime: from Lakshmi Puja night to 4 days after it the blend over-forecasts in every backtest window
(2024 and 2025: best scale 0.80-0.85; the models carry the fireworks build-up forward while emissions return to
normal). Fix 1 (8 Nov, festival_tilt): for those origins the central forecast is 0.5 * blend + 0.5 * fine-tuned
Chronos-2, the least biased member there (it learned the fireworks effect from past Diwalis); this beats the
blend in every year x origin-day cell and every station, and LOFO 21.77 -> 21.54. Fix 2 (fallback): a festival
pair fit on post-Diwali windows, used only if a pair fit on one year beats the general pair on the other year,
shrunk 50/50 toward the general pair. With the tilt in place it is no longer accepted (the bias is gone).

python src/hedge.py [lo hi [origin]]   -> data/processed/hedge.json   (recipe from data/processed/blend_spec.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blend_select as BS  # noqa: E402
from data import DIWALI  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
A_GRID = np.round(np.arange(0.66, 1.0001, 0.01), 2)
B_GRID = np.round(np.arange(0.90, 1.3001, 0.01), 2)
FEST_DAYS = 4      # origins Diwali+0 .. +4 form the festival regime
FEST_SHRINK = 0.5  # festival pair = SHRINK * post-Diwali fit + (1 - SHRINK) * general pair
# festival-regime central forecast: the blend over-forecasts post-Diwali (x1.16-1.82 in every 2024/2025 origin day),
# the fine-tuned Chronos-2 (learned the fireworks effect from past Diwalis) is the least biased member (x0.82-1.35);
# their equal mix beats the blend in all 10 year x origin-day cells and all 9 stations (leads 3-51)
FEST_MIX = 0.5
FEST_MEMBER = "ft_fw_nb__chronos2ft_cov"
# festival member set (9 Nov): for origins Diwali+0..+FEST_DAYS the blend (before the tilt) uses only the zero-shot
# Chronos-2 / Timestral (ctx 1024, 2048) and the fine-tuned Chronos-2; GBDT, PIBM and the upwind variants carry the
# Diwali smoke forward. Post-Diwali LOFO leads 3-51 (tilt on): 17.92 -> 17.47, better in 8/10 year x origin-day cells
# incl. Diwali+2 (the final run's origin day) in both years (2024 18.22 -> 17.22, 2025 15.98 -> 14.83); non-festival
# origins keep the full recipe (better there in 2 of 3 folds). Pre-COVID rehearsal (2019, MPCB stations) in README.
# None disables.
FEST_SET = ["tsfm_fw_nb:chronos2_cov", "tsfm_fw_nb:timestral_cov", "tsfm_fw_nb_c2048:chronos2_cov",
            "tsfm_fw_nb_c2048:timestral_cov", "ft_fw_nb:chronos2ft_cov"]


def fest_cols():
    return [s.replace(":", "__") for s in FEST_SET] if FEST_SET else []


def is_fest(ts0):
    """True for rows whose origin is Diwali+0..+FEST_DAYS."""
    t = pd.Series(pd.DatetimeIndex(ts0))
    return t.map({u: diwali_day(u) <= FEST_DAYS for u in t.unique()}).values.astype(bool)


def festival_tilt(p, member, ts0):
    """Central forecast for origins Diwali+0..+FEST_DAYS: (1 - FEST_MIX) * blend + FEST_MIX * fine-tuned Chronos-2."""
    fest = is_fest(ts0)
    p, member = np.asarray(p, float), np.asarray(member, float)
    ok = fest & np.isfinite(member)
    return np.where(ok, (1 - FEST_MIX) * p + FEST_MIX * member, p)


def diwali_day(ts):
    """Days since the latest Lakshmi Puja (or 99 if none within FEST_DAYS)."""
    t = pd.Timestamp(ts).normalize()
    d = [(t - lp).days for lp in DIWALI if 0 <= (t - lp).days <= FEST_DAYS]
    return min(d) if d else 99


def _lofo_p(spec, lo, hi, blocks=True):
    from scipy.optimize import nnls
    M, cols = BS.load(spec)
    names = list(cols)
    B = M[(M.h >= lo) & (M.h <= hi)].copy()
    B["p"] = np.nan
    for f in B.fold.unique():
        tr, va = B[B.fold != f], B[B.fold == f]
        if blocks:
            p = BS.apply_blocks(va, names, BS.fit_blocks(tr, names))
        else:
            p = va[names].values @ nnls(tr[names].values, tr.y.values)[0]
        B.loc[va.index, "p"] = p
    B["p"] = B.p.clip(lower=0)
    return B


def lofo_blend(spec, lo, hi, blocks=True):
    """Clean OOF blend for each fold with weights fit on the other folds (same blender as predict.py)."""
    B = _lofo_p(spec, lo, hi, blocks)
    if FEST_SET and set(FEST_SET) < set(spec) and is_fest(B.ts0.values).any():  # festival member set
        K = ["sid", "ts0", "ts", "h"]
        C = _lofo_p(FEST_SET, lo, hi, blocks)[K + ["p"]].rename(columns={"p": "p_fest"})
        B = B.merge(C, on=K, how="left")
        m = is_fest(B.ts0.values) & B.p_fest.notna().values
        B.loc[m, "p"] = B.p_fest[m]
        B = B.drop(columns="p_fest")
    if FEST_MEMBER in B.columns:  # same festival-regime central forecast as predict.py writes
        B["p"] = festival_tilt(B.p.values, B[FEST_MEMBER].values, B.ts0.values)
    return B


def window_table(B, min_frac=0.5):
    """Per-window RMSE for every scale in A_GRID and B_GRID. Rows: windows (fold, ts0) with enough rows."""
    n = B.groupby(["fold", "ts0"]).size()
    keep = n[n >= min_frac * n.max()].index
    scales = np.r_[A_GRID, B_GRID]
    rows, meta = [], []
    for (f, t0), g in B.set_index(["fold", "ts0"]).loc[keep].groupby(level=[0, 1]):
        y, p = g.y.values, g.p.values
        rows.append(np.sqrt(((y[None, :] - scales[:, None] * p[None, :]) ** 2).mean(1)))
        meta.append((f, t0))
    W = pd.DataFrame(meta, columns=["fold", "ts0"])
    return W, np.array(rows), scales


def best_pair(R, mask):
    na = len(A_GRID)
    Ra, Rb = R[mask, :na], R[mask, na:]
    S = np.minimum(Ra[:, :, None], Rb[:, None, :]).mean(0)  # (a, b)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return i, j, float(S[i, j])


def score(R, mask, i, j):
    na = len(A_GRID)
    return float(np.minimum(R[mask, i], R[mask, na + j]).mean())


def pair_score(R, mask, a, b, scales):
    ia, ib = int(np.argmin(np.abs(scales - a))), int(np.argmin(np.abs(scales - b)))
    return float(np.minimum(R[mask, ia], R[mask, ib]).mean())


def main(spec=None, lo=None, hi=None, origin=None, write=True):
    J = json.load(open(PROC / "blend_spec.json"))
    spec = spec or [f'{v["exp"]}:{v["model"]}' for v in J["components"].values()]
    lo, hi = lo or J["lo"], hi or J["hi"]
    B = lofo_blend(spec, lo, hi, blocks=J.get("blocks") is not None)
    W, R, scales = window_table(B)
    one = int(np.argmin(np.abs(scales - 1.0)))
    folds = sorted(W.fold.unique())
    out = {"lo": lo, "hi": hi, "windows": int(len(W)), "lofo": {}}
    ok = True
    for f in folds:  # general pair: leave-one-fold-out
        tr, va = (W.fold != f).values, (W.fold == f).values
        i, j, _ = best_pair(R, tr)
        hs, ss = score(R, va, i, j), float(R[va, one].mean())
        out["lofo"][f] = {"a": float(A_GRID[i]), "b": float(B_GRID[j]), "hedged": round(hs, 3), "single": round(ss, 3)}
        ok &= hs < ss
    i, j, s_all = best_pair(R, np.ones(len(W), bool))
    a, b = (float(A_GRID[i]), float(B_GRID[j])) if ok else (1.0, 1.0)
    out["general"] = {"a": a, "b": b, "accepted": bool(ok), "in_sample_hedged": round(s_all, 3),
                      "in_sample_single": round(float(R[:, one].mean()), 3)}
    # festival regime: post-Diwali windows, leave-one-year-out check
    dd = np.array([diwali_day(t) for t in W.ts0])
    yr = W.ts0.dt.year.values
    m = dd <= FEST_DAYS
    fest = None
    if m.any() and len(np.unique(yr[m])) >= 2:
        fok, ly = True, {}
        for y in np.unique(yr[m]):
            tr, te = m & (yr != y), m & (yr == y)
            ii, jj, _ = best_pair(R, tr)
            hs, gs = score(R, te, ii, jj), pair_score(R, te, a, b, scales)
            ly[int(y)] = {"pair_from_other_year": [float(A_GRID[ii]), float(B_GRID[jj])], "hedged": round(hs, 3),
                          "general_pair": round(gs, 3), "single": round(float(R[te, one].mean()), 3)}
            fok &= hs < gs
        ii, jj, _ = best_pair(R, m)
        fa = round(FEST_SHRINK * float(A_GRID[ii]) + (1 - FEST_SHRINK) * a, 2)
        fb = round(FEST_SHRINK * float(B_GRID[jj]) + (1 - FEST_SHRINK) * b, 2)
        fest = {"windows": int(m.sum()), "leave_year_out": ly, "accepted": bool(fok),
                "subset_pair": [float(A_GRID[ii]), float(B_GRID[jj])], "shrunk_pair": [fa, fb],
                "post_diwali": {"single": round(float(R[m, one].mean()), 3),
                                "general_pair": round(pair_score(R, m, a, b, scales), 3),
                                "shrunk_pair": round(pair_score(R, m, fa, fb, scales), 3)},
                "non_festival_cost": {"general_pair": round(pair_score(R, ~m, a, b, scales), 3),
                                      "shrunk_pair": round(pair_score(R, ~m, fa, fb, scales), 3)}}
        out["festival"] = fest
    regime = "general"
    if origin is not None and diwali_day(origin) <= FEST_DAYS and fest and fest["accepted"]:
        a, b = fest["shrunk_pair"]
        regime = f"festival (Diwali+{diwali_day(origin)})"
    a, b = sorted([a, b])
    out.update({"origin": str(origin), "regime": regime, "a": a, "b": b})
    print(json.dumps(out, indent=1, default=str))
    if write:
        json.dump(out, open(PROC / "hedge.json", "w"), indent=1, default=str)
    return a, b


if __name__ == "__main__":
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else None
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else None
    origin = pd.Timestamp(sys.argv[3]) if len(sys.argv) > 3 else None
    main(lo=lo, hi=hi, origin=origin)
