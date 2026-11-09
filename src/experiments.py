"""Clean ablation experiments on a frozen data snapshot.

Every experiment uses the same folds (train.FOLDS), the same origins (23:00 daily) and writes its OOF
predictions to data/processed/exp/<name>/. Results (RMSE per fold x lead bucket) go to experiments.csv.

python src/experiments.py gbm            # GBDT ablations (CPU)
python src/experiments.py tsfm           # foundation-model ablations (GPU)
python src/experiments.py report         # print the table
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "processed" / "exp"
EXP.mkdir(parents=True, exist_ok=True)
TABLE = ROOT / "experiments.csv"
BUCKETS = {"h1-48": (1, 48), "h4-52": (4, 52), "h16-63": (16, 63)}

SEASON = ["doy", "month", "dvl"]
REL = ["doy", "month", "dvl", "dvl_c", "dvl0", "cams", "cams24", "cams72_o", "cams_mos", "cams_mos24"]
GBM_CONFIGS = [  # name, use_nb, exclude prefixes, exclude exact
    ("gbm_base", False, ["fw"], []),
    ("gbm_base_ns", False, ["fw"], SEASON),
    ("gbm_rel", False, [], REL),
    ("gbm_rel_nb", True, [], REL),
    ("gbm_rel_nb_w", True, [], REL),  # + sample reweighting (targets x3, 2026 x2)
    ("gbm_rel_nb_mw", True, [], REL),  # E2 multi-NWP ensemble mean + spread
    ("gbm_rel_nb_up", True, [], REL),  # E3 Lagrangian upwind neighbour signal
    ("gbm_rel_nb_cp", True, [], REL),  # E1 co-pollutants + station met
    ("gbm_rel_nb_cp_hole", True, [], REL),  # research: Diwali eve+day target hours removed (data.diwali_hole_mask)
]
TSFM_CONFIGS = [  # name, models, use_fw, use_nb
    ("tsfm_base", ("chronos2", "timestral", "timesfm3"), False, False),
    ("tsfm_fw", ("chronos2", "timestral", "timesfm3"), True, False),
    ("tsfm_fw_nb", ("chronos2", "timestral", "timesfm3"), True, True),
    ("ft_fw", ("chronos2ft",), True, False),
    ("ft_fw_nb", ("chronos2ft",), True, True),
    ("tsfm_fw_nb_mw", ("chronos2", "timestral", "timesfm3"), True, True),  # E2
    ("tsfm_fw_nb_up", ("chronos2", "timestral", "timesfm3"), True, True),  # E3
    ("tsfm_fw_nb_cp", ("chronos2", "timestral", "timesfm3"), True, True),  # E1
    ("ft_fw_nb_mw", ("chronos2ft",), True, True),
    ("tsfm_fw_nb_c2048", ("chronos2", "timestral"), True, True),  # diversity: context length
    ("tsfm_fw_nb_c512", ("chronos2", "timestral"), True, True),
    ("tsfm_fw_nb_cp_x", ("timestral",), True, True),  # E1 Timestral rerun (OOM at batch 64)
    # Chronos-2 fine-tuning variants (GPU server)
    ("ft_fw_nb_up", ("chronos2ft",), True, True),
    ("ft_fw_nb_s2000", ("chronos2ft",), True, True),
    ("ft_fw_nb_lr3e-5", ("chronos2ft",), True, True),
    ("ft_fw_nb_seed1", ("chronos2ft",), True, True),
    ("ft_fw_nb_seed2", ("chronos2ft",), True, True),
    ("tsfm_fw_nb_t0", ("t0",), True, True),  # t0-gamma (open weights, known-future covariates)
    ("tsfm_fw_nb_t0sym", ("t0",), True, True),  # t0-gamma, symmetric averaging, target-only context
    ("tsfm_fw_th2o", ("timeh2o",), True, False),  # TimeH2O-v1 (Python 3.12 worker; known-future covariates)
    ("tsfm_fw_th2o_c4096", ("timeh2o",), True, False),  # ... with a 4096-h context (model supports 16k)
    # festival-smoke hours masked in the target context (tsfm.smoke_mask); only post-Diwali origins change
    ("tsfm_fw_nb_mfw", ("chronos2", "timestral"), True, True),
    ("tsfm_fw_nb_mfw_c2048", ("chronos2", "timestral"), True, True),
    ("tsfm_fw_nb_up_mfw", ("chronos2", "timestral"), True, True),
    ("ft_fw_nb_mfw", ("chronos2ft",), True, True),
    # TabPFN-TS (tabular in-context regression on known covariates + calendar; src/tabpfn_worker.py)
    ("tsfm_fw_tabpfn", ("tabpfn",), True, False),
    ("tsfm_fw_tabpfn_ts3", ("tabpfn",), True, False),  # TabPFN-TS-3: checkpoint fine-tuned for forecasting
    # research: the final run's context lacks the withheld 2026 public window (Diwali eve + Diwali day, 7-8 Nov);
    # "_hole" removes the same target hours in 2024/2025 to measure the effect on post-Diwali forecasts
    ("tsfm_fw_nb_hole", ("chronos2", "timestral"), True, True),
    ("ft_fw_nb_hole", ("chronos2ft",), True, True),
    ("tsfm_fw_nb_hole_c2048", ("chronos2", "timestral"), True, True),
    ("tsfm_fw_nb_up_hole", ("chronos2", "timestral"), True, True),
    # 9 Nov frontier (Track A): fine-tune also on the 2018-19 MPCB history (2 extra pre-COVID Diwalis, src/k18.py);
    # CAMS has no 2018-19 history, so both drop it ("_nocams" = control isolating that effect)
    ("ft_fw_nb_nocams", ("chronos2ft",), True, True),
    ("ft_fw_nb_nocams_k18", ("chronos2ft",), True, True),
]


def set_flags(name):
    """Experiment-name suffixes switch the new data flags (E1 _cp, E2 _mw, E3 _up)."""
    import features as F
    import neighbors as N
    F.WX_MULTI = "_mw" in name
    N.USE_UP = "_up" in name
    N.USE_CP = "_cp" in name
    import re
    import tsfm as TS
    m = re.search(r"_c(\d+)$", name)
    TS.CTX = int(m.group(1)) if m else 1024
    m = re.search(r"_s(\d+)", name)
    TS.FT_STEPS = int(m.group(1)) if m else 800
    m = re.search(r"_lr(\d+e-\d+)", name)
    TS.FT_LR = float(m.group(1)) if m else 1e-5
    m = re.search(r"_seed(\d+)", name)
    TS.FT_SEED = int(m.group(1)) if m else 0
    TS.T0_SYM = "_t0sym" in name
    TS.MASK_FW = "_mfw" in name
    TS.NO_CAMS = "_nocams" in name
    TS.USE_K18 = "_k18" in name
    TS.TABPFN_CKPT = TS.TS3_CKPT if "_ts3" in name else None
    import os
    os.environ["DIWALI_HOLE"] = "1" if "_hole" in name else "0"  # read by data.load_all at call time


def snapshot_id():
    h = hashlib.md5()
    for f in ["data/raw/Train.csv", "data/external/weather_prevruns.parquet", "data/external/cams_hist.parquet",
              "data/external/openaq_neighbors.parquet"]:
        p = ROOT / f
        h.update(p.read_bytes() if p.exists() else b"-")
    return h.hexdigest()[:10]


def score(name, O, kind, flags):
    if "y" not in O.columns:  # ground truth straight from the (cleaned) data
        from data import load_all, to_grid
        d = load_all()
        G = to_grid(d[~d.sid.str.startswith("k18")]).stack().rename("y").reset_index()
        G.columns = ["ts", "sid", "y"]
        O = O.merge(G, on=["ts", "sid"], how="inner")
    cols = [c for c in O.columns if c.startswith(("gbm_", "b_", "chronos2", "timestral", "timesfm3", "pibm", "tabpfn", "t0_", "timeh2o"))]
    snap = snapshot_id()
    rows = []
    for c in cols:
        for fold, g in O.groupby("fold"):
            for bk, (lo, hi) in BUCKETS.items():
                gg = g[(g.h >= lo) & (g.h <= hi)].dropna(subset=[c, "y"])
                rows.append({"time": datetime.now().isoformat(timespec="seconds"), "snapshot": snap, "exp": name,
                             "kind": kind, "flags": json.dumps(flags), "model": c, "fold": fold, "bucket": bk,
                             "n": len(gg), "rmse": float(np.sqrt(np.mean((gg.y - gg[c]) ** 2))) if len(gg) else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(TABLE, mode="a", header=not TABLE.exists(), index=False)
    return R


def run_gbm(only=None):
    import train as TR
    for name, nb, excl, exact in GBM_CONFIGS:
        if only and name not in only:
            continue
        TR.USE_NB, TR.EXCLUDE_PREFIX, TR.EXCLUDE_EXACT = nb, excl, exact
        TR.REWEIGHT = name.endswith("_w")
        set_flags(name)
        d = EXP / name
        print(f"=== {name}", flush=True)
        TR.cv(out_dir=d, clean=True)
        score(name, pd.read_parquet(d / "oof_gbm.parquet"), "gbm", {"nb": nb, "exclude": excl + exact})


def run_tsfm(only=None):
    import tsfm as TS
    for name, models, fw, nb in TSFM_CONFIGS:
        if only and name not in only:
            continue
        TS.USE_FW, TS.USE_NB = fw, nb
        set_flags(name)
        d = EXP / name
        d.mkdir(exist_ok=True)
        print(f"=== {name}", flush=True)
        O = TS.cv(models, out_path=d / "oof_tsfm.parquet")
        score(name, O, "tsfm", {"fw": fw, "nb": nb, "models": list(models)})


def report(snapshot=None):
    T = pd.read_csv(TABLE)
    snap = snapshot or T.snapshot.iloc[-1]
    T = T[T.snapshot == snap].drop_duplicates(["exp", "model", "fold", "bucket"], keep="last")
    P = T.pivot_table(index=["exp", "model"], columns=["bucket", "fold"], values="rmse")
    for bk in BUCKETS:
        sub = P[bk].copy()
        sub["mean"] = sub.mean(1)
        print(f"\n--- {bk}  (snapshot {snap})")
        print(sub.round(2).sort_values("mean").to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    {"gbm": lambda: run_gbm(only), "tsfm": lambda: run_tsfm(only), "report": lambda: report()}[cmd]()
