"""End-to-end submission with the recipe selected by clean ablations (experiments.csv):
GBDT (relative CAMS, no season proxies, neighbours) raw+ratio, Timestral / Chronos-2 / TimesFM-3 zero-shot and
Chronos-2 fine-tuned, all with fireworks + neighbour covariates. NNLS weights fit on clean OOFs for the lead range
of the private window from this origin; origins 0-4 days after Diwali mix in the fine-tuned Chronos-2 50/50
(hedge.festival_tilt; the blend over-forecasts post-Diwali). Also writes the two hedge legs <out>_lo.csv / <out>_hi.csv (src/hedge.py):
the pair to select as the 2 finals.

python src/predict.py <fcst_tag> <out.csv> [origin_ts]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as TR  # noqa: E402
import tsfm as TS  # noqa: E402
from data import TARGETS, load_all, load_test, to_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RECIPE = ["gbm_rel_nb_cp:gbm_raw", "gbm_rel_nb_cp:gbm_ratio",
          "tsfm_fw_nb:chronos2_cov", "tsfm_fw_nb:timestral_cov",
          "ft_fw_nb:chronos2ft_cov", "pibm:pibm",
          "tsfm_fw_nb_c2048:chronos2_cov", "tsfm_fw_nb_c2048:timestral_cov",
          "tsfm_fw_nb_up:chronos2_cov", "tsfm_fw_nb_up:timestral_cov"]
# TimesFM-3 members dropped 8 Nov: zero NNLS weight at leads 3-51; LOFO 21.784 -> 21.772 without them (recent equal)
GBM_EXP = "gbm_rel_nb_cp"


def run(fcst_tag, out_csv, origin=None):
    out_csv = Path(out_csv)
    gbm_csv = out_csv.with_name(out_csv.stem + "_gbm.csv")
    import experiments as EX
    TR.USE_NB, TR.EXCLUDE_PREFIX, TR.EXCLUDE_EXACT, TR.REWEIGHT = True, [], TR.SEASON + TR.CAMS_ABS, False
    TS.USE_FW, TS.USE_NB = True, True
    EX.set_flags(GBM_EXP)  # data flags of the selected GBDT config (co-pollutants)
    it = json.load(open(PROC / "exp" / GBM_EXP / "cv_iters.json"))
    json.dump(it, open(PROC / "cv_iters.json", "w"))  # boosting rounds from the clean inner-window selection
    TR.predict_test(fcst_tag, gbm_csv, origin, modes=("raw", "ratio"))
    P = pd.read_parquet(PROC / (gbm_csv.stem + "_detail.parquet")).drop(columns=["pred"])
    # TSFM at the same origins
    d = load_all()
    d = d[d.src == "train"]
    T0 = pd.Timestamp(origin) if origin else d.ts.max()
    te = load_test()
    from neighbors import build_signals
    dall = load_all()
    dall = dall[~dall.sid.str.startswith("k18") & (dall.ts <= T0)]
    # blend weights re-fit (OOF, NNLS, shrunk 3h blocks) on the private window's lead range from this origin
    import blend_select as BS
    hp = ((te[te.scoring_block == "private"].ts - T0) / pd.Timedelta(hours=1)).astype(int)
    lo, hi = max(1, int(hp.min())), min(TS.HMAX, int(hp.max()))
    if lo > hi:  # private window beyond horizon (pre-final runs): weight for the next-48h public window
        lo, hi = 1, 48
    print("private lead range", lo, hi)
    BS.main(RECIPE, lo, hi)
    spec = json.load(open(PROC / "blend_spec.json"))
    # run every recipe member, not only those with non-zero global weight: the festival member set and the backfill
    # re-blend need them (10 Nov final, leads 5-52: global weights beat blocks, tsfm_fw_nb got weight 0 and was not
    # run, so the festival set was silently skipped). Block-weight runs are unchanged (same members, same order).
    comps = {f"{e}__{m}": {"exp": e, "model": m} for e, m in (s.split(":") for s in RECIPE)}
    P = P.rename(columns={"gbm_raw": f"{GBM_EXP}__gbm_raw", "gbm_ratio": f"{GBM_EXP}__gbm_ratio",
                          "cams_mos24": "b_cams_mos24"})
    L = None
    for col, cm in comps.items():
        exp, model = cm["exp"], cm["model"]
        if exp.startswith("gbm") or model == "pibm" or exp == "gcf":
            continue
        EX.set_flags(exp)  # per-member data flags (ctx length, upwind, co-pollutants)
        grid = to_grid(d[d.ts <= T0])
        grid = grid.reindex(pd.date_range(grid.index.min(), max(te.ts.max(), T0 + pd.Timedelta(hours=TS.HMAX)),
                                          freq="h"))
        exo = TS.Exo(grid.index, fcst_tag, nb=build_signals(dall, grid.index, upto=T0))
        org = sorted(set(np.searchsorted(grid.index, P.ts0.unique())))
        items = TS.build_items(grid, exo, [(str(k), str(k)) for k in TARGETS], org)
        TS.FT_PIPE = None
        if model == "chronos2ft_cov":  # fine-tune on every Mumbai series, data <= T0 only
            ga = to_grid(dall)
            ga = ga.reindex(pd.date_range(ga.index.min(), grid.index.max(), freq="h"))
            ea = TS.Exo(ga.index, fcst_tag, nb=build_signals(dall, ga.index, upto=T0))
            TS.FT_PIPE = TS.chronos_finetune(ga, ea, TS.all_sid_locs(dall, ga), int(np.searchsorted(ga.index, T0)) + 1)
        fn = TS.MODELS.get(model[:-4], TS.chronos_forecast)
        print("running", col, "ctx", TS.CTX, flush=True)
        Y = TS.to_long(items, np.clip(fn(items, TS.HMAX, use_cov=True), 0, None), grid, col)
        L = Y if L is None else L.merge(Y, on=["sid", "ts0", "ts", "h"], how="outer")
        TS.FT_PIPE = None
    EX.set_flags("")
    if any(cm["model"] == "pibm" for cm in comps.values()):  # physics-informed box model member
        import pibm as PB
        Y = PB.forecast(T0, fcst_tag, sorted(P.ts0.unique())).rename(columns={"pibm": "pibm__pibm"})
        L = Y if L is None else L.merge(Y, on=["sid", "ts0", "ts", "h"], how="left")
    import hedge as HG
    if any(cm["exp"] == "gcf" for cm in comps.values()) or any(s.startswith("gcf:") for s in (HG.FEST_SET or [])):
        import geoscf as GC  # NASA GEOS-CF members (newest published run before each origin)
        Y = GC.live_members(T0, d[d.ts <= T0][["sid", "ts", "y"]], sorted(P.ts0.unique()))
        if Y is not None:
            L = Y if L is None else L.merge(Y, on=["sid", "ts0", "ts", "h"], how="left")
    P = P.merge(L, on=["sid", "ts0", "ts", "h"], how="left")
    if spec.get("blocks"):
        cols = list(spec["names"])
        hb = (P.ts.dt.hour // 3 * 3).values
        wrow = np.vstack([np.asarray(spec["blocks"][str(b)]) for b in hb])
    else:
        cols = list(spec["weights"])
        wrow = np.tile(np.array([spec["weights"][c] for c in cols]), (len(P), 1))
    X = P[cols].values.astype(float)
    # missing component -> renormalise remaining weights per row
    M = ~np.isnan(X)
    num = np.nansum(X * wrow, 1)
    den = (M * wrow).sum(1)
    P["blend"] = np.where(den > 0, num / np.maximum(den, 1e-9) * wrow.sum(1), P[f"{GBM_EXP}__gbm_raw"])
    # rows before T0 come from day-ahead forecasts (issued 23:00 the previous day, leads 1-24): blend them with
    # weights fit for that lead range, not the private window's (5-6 Nov public rows: 10.0 -> 8.5)
    bk = (P.src == "bk").values
    if bk.any():
        Mb, cb = BS.load(RECIPE)
        nb = [c for c in cb if c in P.columns]
        if len(nb) == len(cb):
            Wb = BS.fit_blocks(Mb[(Mb.h >= 1) & (Mb.h <= 24)], nb)
            idx = P.index[bk & P[nb].notna().all(1).values]
            P.loc[idx, "blend"] = np.clip(BS.apply_blocks(P.loc[idx], nb, Wb), 0, None)
            print(f"backfill rows re-blended with lead 1-24 weights: {len(idx)}")
    import hedge as HG
    fc = HG.fest_cols()  # festival regime (origin Diwali+0..4): blend of the festival member set only
    if fc and all(c in P.columns for c in fc):
        fm = HG.is_fest(P.ts0.values) & P[fc].notna().all(1).values
        if fm.any():
            Mf, _ = BS.load(HG.FEST_SET)
            for rows, (a, b) in ((fm & ~bk, (lo, hi)), (fm & bk, (1, 24))):
                if rows.any():
                    Wf = BS.fit_blocks(Mf[(Mf.h >= a) & (Mf.h <= b)], fc)
                    P.loc[rows, "blend"] = np.clip(BS.apply_blocks(P.loc[rows], fc, Wf), 0, None)
            t = P[(P.src == "T0") & fm]
            print(f"festival member set ({len(fc)} models): {int(fm.sum())} rows; T0 blend mean {t.blend.mean():.1f}")
    if HG.FEST_MEMBER in P.columns:  # festival regime (origin Diwali+0..4): mix in the fine-tuned Chronos-2
        P["blend_nnls"] = P.blend
        P["blend"] = HG.festival_tilt(P.blend.values, P[HG.FEST_MEMBER].values, P.ts0.values)
        t = P[P.src == "T0"]
        print(f"festival tilt at T0 (Diwali+{HG.diwali_day(T0)}): blend mean {t.blend_nnls.mean():.1f} -> {t.blend.mean():.1f}")
    P.to_parquet(PROC / (out_csv.stem + "_detail.parquet"))
    # assemble submission (same rules as GBDT: future rows from T0, past rows from previous-day 23h)
    sub = te[["ID", "location_id", "ts"]].copy()
    sub["sid"] = sub.location_id.astype(str)
    a = P[P.src == "T0"][["sid", "ts", "blend"]].rename(columns={"blend": "pf"})
    b = P[(P.src == "bk") & (P.ts <= T0)].sort_values("h")[["sid", "ts", "blend"]].drop_duplicates(["sid", "ts"])
    sub = sub.merge(a, on=["sid", "ts"], how="left").merge(b.rename(columns={"blend": "pb"}), on=["sid", "ts"], how="left")
    sub["Target"] = np.where(sub.ts > T0, sub.pf, sub.pb)
    if sub.Target.isna().any():
        last = P[(P.src == "T0") & (P.h > TS.HMAX - 24)].copy()
        fill = last.groupby(["sid", last.ts.dt.hour]).blend.mean()
        miss = sub.Target.isna()
        sub.loc[miss, "Target"] = [fill.get((s, t.hour), np.nan) for s, t in zip(sub.sid[miss], sub.ts[miss])]
    g = pd.read_csv(gbm_csv)
    sub["Target"] = sub.Target.fillna(pd.Series(g.Target.values, index=sub.index)).clip(lower=0)
    TR.check_sub(sub[["ID", "Target"]])
    sub[["ID", "Target"]].to_csv(out_csv, index=False)
    print("origin", T0, "wrote", out_csv)
    # two-pick hedge: the challenge scores the best of the 2 selected submissions -> write the lo / hi legs (src/hedge.py)
    ha, hb = HG.main(RECIPE, lo, hi, origin=T0)
    # the hedge scales only the private-window rows; every other row keeps the central forecast
    priv = sub.ID.map(te.set_index("ID").scoring_block).eq("private").values
    for leg, k in (("lo", ha), ("hi", hb)):
        s_leg = sub[["ID"]].assign(Target=np.clip(np.where(priv, sub.Target * k, sub.Target), 0, None))
        TR.check_sub(s_leg)
        s_leg.to_csv(out_csv.with_name(f"{out_csv.stem}_{leg}.csv"), index=False)
        print(f"hedge leg {leg} x{k:.2f} ->", out_csv.with_name(f"{out_csv.stem}_{leg}.csv"))
    s = sub.merge(te[["ID", "target_end_ist"]], on="ID")
    s["d"] = s.target_end_ist.str[:10]
    print(s[(s.ts > T0 - pd.Timedelta(days=2)) & (s.ts < T0 + pd.Timedelta(days=4))]
          .pivot_table(index="d", columns="location_id", values="Target").round(1))


if __name__ == "__main__":
    run(sys.argv[2 - 1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
