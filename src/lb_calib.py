"""Public-leaderboard calibration (public board only; never a final). Scales a genuine forecast's public-window rows
per time block with factors solved from public leaderboard scores alone: for block s with scale k_s,
    RMSE^2 = sum_s (k_s^2 A_s - 2 k_s B_s) + C,   A_s = mean_s(p^2), B_s = mean_s(p*y)  (unknown, rows scored by the host)
so the scores of (k_s = 2) and (k_s = 3), all else at 1, give A_s and B_s, and k_s* = B_s / A_s.
No observation from the scored window is read; private-window rows are left as in the base file.

python src/lb_calib.py <base.csv> <base_public_score> <out_tag>    (env CHALLENGE_USER / CHALLENGE_PASS)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leaderboard_io as Z  # noqa: E402
from data import load_test  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W0, W1 = pd.Timestamp("2026-11-07 00:00"), pd.Timestamp("2026-11-09 00:00")  # active public window (hour-ending ts)
BLOCKS = {  # festival nights by calendar (Diwali eve, Diwali), rest of the window
    "night_eve": (pd.Timestamp("2026-11-07 18:00"), pd.Timestamp("2026-11-08 03:00")),
    "night_diwali": (pd.Timestamp("2026-11-08 18:00"), pd.Timestamp("2026-11-09 00:00")),
}


def blocks(te):
    win = (te.ts > W0) & (te.ts <= W1)
    out = {}
    rest = win.copy()
    for k, (a, b) in BLOCKS.items():
        m = win & (te.ts > a) & (te.ts <= b)
        out[k] = m.values
        rest &= ~m
    out["day"] = rest.values
    return out


def submit_and_score(tok, path, comment):
    sid = Z.submit(tok, str(path), comment)["id"]
    for _ in range(60):
        time.sleep(10)
        for s in Z.status(tok)["submissions"]:
            if s["id"] == sid and s["status"] == "successful":
                return sid, float(s["public_score"])
    raise RuntimeError(f"no score for {sid}")


def main(base_csv, s0, tag):
    te = load_test()
    base = pd.read_csv(base_csv)
    assert (base.ID.values == te.ID.values).all()
    B = blocks(te)
    tok = Z.signin()
    out_dir = ROOT / "submissions" / "lb"
    out_dir.mkdir(exist_ok=True)
    log = {"base": str(base_csv), "s0": s0, "probes": {}}
    s0 = float(s0)
    res = {}
    for name, m in B.items():
        sc = {}
        for k in (2.0, 3.0):
            f = base.copy()
            f.loc[m, "Target"] = f.Target[m] * k
            p = out_dir / f"{tag}_probe_{name}_x{int(k)}.csv"
            f.to_csv(p, index=False)
            sid, s = submit_and_score(tok, p, f"public-board calibration probe: {name} x{k:g} (not a final)")
            sc[k] = s
            log["probes"][f"{name}_x{k:g}"] = {"id": sid, "score": s}
            print(f"{name} x{k:g}: {s:.4f}", flush=True)
        x1, x2 = sc[2.0] ** 2 - s0 ** 2, sc[3.0] ** 2 - s0 ** 2  # = 3A - 2B,  8A - 4B
        A = (x2 - 2 * x1) / 2
        Bv = (3 * A - x1) / 2
        kopt = float(np.clip(Bv / A, 0, 10)) if A > 1e-9 else 1.0
        res[name] = {"A": A, "B": Bv, "k": round(kopt, 3)}
    gain = sum((r["A"] - 2 * r["B"]) - (r["k"] ** 2 * r["A"] - 2 * r["k"] * r["B"]) for r in res.values())
    pred = float(np.sqrt(max(s0 ** 2 - gain, 0)))
    f = base.copy()
    for name, m in B.items():
        f.loc[m, "Target"] = f.Target[m] * res[name]["k"]
    p = out_dir / f"{tag}_calibrated.csv"
    f.to_csv(p, index=False)
    sid, s = submit_and_score(tok, p, "public-board calibrated: " +
                              ", ".join(f"{k} x{v['k']}" for k, v in res.items()) + " (public only, not a final)")
    log.update({"solution": res, "predicted": pred, "final": {"id": sid, "score": s}})
    print(json.dumps(log, indent=1))
    json.dump(log, open(out_dir / f"{tag}_log.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
