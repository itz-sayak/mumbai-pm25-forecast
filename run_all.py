"""One-command reproducible pipeline (final recipe chosen by clean ablations, see experiments.csv).
Recipe (src/predict.py RECIPE): GBDT (relative CAMS, neighbours, co-pollutants) raw+ratio; Chronos-2, Timestral
(fireworks + neighbours; + 2048-h context variants; + Lagrangian upwind variants); Chronos-2 fine-tuned;
physics-informed box model; blended by NNLS with shrunk 3-h blocks on the private window's lead range; for origins
0-4 days after Diwali the central forecast is 0.5 blend + 0.5 fine-tuned Chronos-2 (hedge.festival_tilt).

    python run_all.py --tag v9nov --download                # after the 9 Nov Train update
    python run_all.py --tag final_am --download --live      # 10 Nov ~10:00 (refresh experiments; fallback pair)
    python run_all.py --tag final --live --skip-exp         # 10 Nov ~21:30 (freshest hours, ~22 min; the 2 finals)
    python run_all.py --tag final --live --skip-exp --fill-gaps --past-from final_tr   # + withheld-window context
    python run_all.py --tag rerun --skip-exp --fcst-tag <tag>   # reproduce from cached inputs
Outputs submissions/<tag>_PICK_lo.csv and <tag>_PICK_hi.csv = the two submissions to select (src/hedge.py).
(--fw-hedge, an extra fireworks night in the high leg, is kept as an option but not used: the foundation models
barely respond to an isolated future flag, and past Bhai Dooj nights were over-forecast too.)

Steps: (1) challenge data download (env CHALLENGE_USER/CHALLENGE_PASS)  (2) Open-Meteo weather/CAMS history + forecast snapshot
(3) OpenAQ neighbour history (past-only covariates) and, with --live, fresh target-station hours newer than
Train.csv — both with a HARD cutoff of 2026-11-11 00:00 IST  (4) re-run the selected experiments on the new
snapshot (clean OOFs for blend weights)  (5) predict -> submissions/<tag>.csv
Seeds fixed: LightGBM seeds (42, 7, 2026) deterministic; torch/numpy seed 0.
"""
import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
SNAP = ROOT / "data" / "snapshots"


def sh(*args, env=None):
    print(">>", " ".join(args), flush=True)
    subprocess.run([PY, "-u", *args], cwd=ROOT, check=True, env={**os.environ, **(env or {})})


def past_from(tag, ref):
    """Rows whose target hour is at or before the origin of the Train-only run <ref> (this includes the withheld public
    window) are copied from <ref>: with --fill-gaps the gap-filled observations only reach the future (private) rows."""
    import pandas as pd
    sys.path.insert(0, str(ROOT / "src"))
    from data import load_test
    te = load_test().set_index("ID")
    P = pd.read_parquet(ROOT / "data" / "processed" / f"{ref}_detail.parquet", columns=["src", "ts0"])
    t0 = P[P.src == "T0"].ts0.iloc[0]
    R = pd.read_csv(ROOT / "submissions" / f"{ref}.csv").set_index("ID").Target
    ids = te.index[(te.ts <= t0).values]
    assert not te.loc[ids].scoring_block.eq("private").any()
    for suf in ("", "_lo", "_hi", "_PICK_lo", "_PICK_hi"):
        f = ROOT / "submissions" / f"{tag}{suf}.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        m = s.ID.isin(ids).values
        s.loc[m, "Target"] = s.ID[m].map(R).values
        assert s.Target.notna().all()
        s.to_csv(f, index=False)
        print(f"past rows (<= {t0}) of {f.name} copied from {ref}.csv: {int(m.sum())}")


def recipe_exps():
    """Experiment dirs of predict.RECIPE (parsed, no heavy imports)."""
    tree = ast.parse((ROOT / "src" / "predict.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "RECIPE":
            return sorted({s.split(":")[0] for s in ast.literal_eval(node.value)})
    raise RuntimeError("RECIPE not found")


def snapshot_files():
    """Every input predict.py reads: raw challenge data, external covariates (forecast snapshots, OpenAQ), and the
    clean OOFs + boosting rounds of the recipe members (blend weights)."""
    files = [p for p in (ROOT / "data" / "raw").rglob("*") if p.is_file()]
    files += [p for p in (ROOT / "data" / "external").rglob("*") if p.is_file() and "aurora" not in p.parts]
    for e in recipe_exps() + ["gbm_base"]:  # gbm_base: ground truth of the festival member set (blend_select.load)
        files += [p for p in (ROOT / "data" / "processed" / "exp" / e).glob("*") if p.is_file()]
    return files


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_snapshot(tag, args):
    """Freeze the exact inputs of this run (code-review reproducibility): data/snapshots/<tag>/ + manifest.json."""
    dst = SNAP / tag
    if dst.exists():
        shutil.rmtree(dst)
    man = {"tag": tag, "args": args, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {}}
    for p in snapshot_files():
        rel = p.relative_to(ROOT).as_posix()
        q = dst / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        man["files"][rel] = md5(p)
    json.dump(man, open(dst / "manifest.json", "w"), indent=1)
    print(f"snapshot: {len(man['files'])} files -> {dst}")


def replay(tag):
    """Re-run predict.py on the frozen inputs of <tag> and compare with the original submissions. The current
    data files are backed up first and restored afterwards, so a replay never changes the working state."""
    src = SNAP / tag
    man = json.load(open(src / "manifest.json"))
    bak = SNAP / f"_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    for rel in man["files"]:
        cur = ROOT / rel
        if cur.exists():
            (bak / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cur, bak / rel)
    newer = [p for p in snapshot_files() if p.relative_to(ROOT).as_posix() not in man["files"]]
    for p in newer:  # inputs that did not exist at snapshot time (moved aside during the replay)
        (bak / "_extra" / p.relative_to(ROOT)).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(bak / "_extra" / p.relative_to(ROOT)))
    try:
        for rel in man["files"]:
            (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, ROOT / rel)
        a = man["args"]
        env = {"USE_LIVE": "1" if a.get("live") else "0", "LIVE_GAPS": "1" if a.get("fill_gaps") else "0"}
        out = f"submissions/{tag}_replay.csv"
        sh("src/predict.py", a.get("fcst_tag") or tag, out, *([a["origin"]] if a.get("origin") else []), env=env)
        if a.get("past_from"):
            past_from(f"{tag}_replay", a["past_from"])
    finally:
        for rel in man["files"]:
            if (bak / rel).exists():
                shutil.copy2(bak / rel, ROOT / rel)
        for p in (bak / "_extra").rglob("*") if (bak / "_extra").exists() else []:
            if p.is_file():
                shutil.move(str(p), str(ROOT / p.relative_to(bak / "_extra")))
    import pandas as pd
    for suf in ("", "_lo", "_hi"):
        o, r = ROOT / "submissions" / f"{tag}{suf}.csv", ROOT / "submissions" / f"{tag}_replay{suf}.csv"
        if o.exists() and r.exists():
            d = (pd.read_csv(o).set_index("ID").Target - pd.read_csv(r).set_index("ID").Target).abs()
            print(f"replay {tag}{suf}: max abs diff {d.max():.6f}, mean {d.mean():.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--live", action="store_true", help="append fresh OpenAQ target hours newer than Train.csv")
    ap.add_argument("--skip-exp", action="store_true", help="reuse existing experiment OOFs for blend weights")
    ap.add_argument("--fcst-tag", default=None, help="reuse a cached forecast snapshot instead of fetching")
    ap.add_argument("--origin", default=None)
    ap.add_argument("--fw-hedge", nargs="?", const="2026-11-11", default=None,
                    help="high hedge leg assumes an extra fireworks night on this date (Bhai Dooj may be Diwali+3)")
    ap.add_argument("--replay", action="store_true", help="re-run predict on data/snapshots/<tag> and diff the CSVs")
    ap.add_argument("--snapshot-only", action="store_true", help="freeze the current inputs as data/snapshots/<tag>")
    ap.add_argument("--fill-gaps", action="store_true",
                    help="with --live: also use OpenAQ hours missing from Train (the withheld public window) as context")
    ap.add_argument("--past-from", default=None,
                    help="Train-only run whose rows at or before its origin replace this run's (use with --fill-gaps)")
    a = ap.parse_args()
    if a.replay:
        return replay(a.tag)
    if a.snapshot_only:
        return save_snapshot(a.tag, {"live": a.live, "fcst_tag": a.fcst_tag or a.tag, "origin": a.origin})
    env = {"USE_LIVE": "1" if a.live else "0", "LIVE_GAPS": "1" if a.fill_gaps else "0"}
    if a.fill_gaps and not (a.live and a.past_from):
        raise SystemExit("--fill-gaps needs --live and --past-from <Train-only tag>")
    if a.download:
        sh("src/leaderboard_io.py", "download")
    if a.fcst_tag is None:
        sh("src/external.py", "hist")
        tag = a.tag
        sh("src/external.py", "fcst", tag)
    else:
        tag = a.fcst_tag
    if os.environ.get("OPENAQ_API_KEY"):
        sh("src/openaq_neighbors.py")   # neighbour stations (past-only context)
        sh("src/openaq_copoll.py")      # co-located pollutants + station wind (past-only context)
        if a.live:
            sh("src/openaq_live.py")
    if not a.skip_exp:  # re-run exactly the selected members on the new snapshot (clean OOFs -> blend weights)
        for f in (ROOT / "data" / "processed").glob("train_samples_nb*.parquet"):
            f.unlink()
        sh("src/experiments.py", "gbm", "gbm_rel_nb_cp", env=env)
        sh("src/experiments.py", "tsfm", "tsfm_fw_nb,ft_fw_nb,tsfm_fw_nb_c2048,tsfm_fw_nb_up", env=env)
        sh("src/pibm.py", "cv", env=env)
    args = ["src/predict.py", tag, f"submissions/{a.tag}.csv"] + ([a.origin] if a.origin else [])
    sh(*args, env=env)  # -> <tag>.csv (central) + <tag>_lo.csv / <tag>_hi.csv (two-pick hedge legs)
    lo_pick, hi_pick = ROOT / "submissions" / f"{a.tag}_lo.csv", ROOT / "submissions" / f"{a.tag}_hi.csv"
    if a.fw_hedge:  # high leg = scenario with an extra fireworks night (Bhai Dooj on Diwali+3), low leg = calendar as is
        args = ["src/predict.py", tag, f"submissions/{a.tag}_fw.csv"] + ([a.origin] if a.origin else [])
        sh(*args, env={**env, "FW_EXTRA_NIGHTS": a.fw_hedge})
        hi_pick = ROOT / "submissions" / f"{a.tag}_fw_hi.csv"
    for name, src in (("PICK_lo", lo_pick), ("PICK_hi", hi_pick)):
        dst = ROOT / "submissions" / f"{a.tag}_{name}.csv"
        shutil.copyfile(src, dst)
        print("final pair:", dst, "<-", src.name)
    if a.past_from:
        past_from(a.tag, a.past_from)
    save_snapshot(a.tag, {"live": a.live, "fcst_tag": tag, "origin": a.origin, "fill_gaps": a.fill_gaps,
                          "past_from": a.past_from})  # frozen inputs for code review


if __name__ == "__main__":
    main()
