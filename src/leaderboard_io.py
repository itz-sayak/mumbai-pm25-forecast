"""Minimal client for the challenge platform API (data download, submit, status, leaderboard, final selection).

Platform endpoint and credentials are read ONLY from the environment: CHALLENGE_API (REST base URL), CHALLENGE_SITE
(web origin), CHALLENGE_SLUG, CHALLENGE_USER, CHALLENGE_PASS.
Usage:
    python src/leaderboard_io.py download            # -> data/raw/
    python src/leaderboard_io.py submit <csv> "<comment>"
    python src/leaderboard_io.py status              # limits, rank, recent submissions
    python src/leaderboard_io.py leaderboard [n]     # public leaderboard (participations endpoint)
    python src/leaderboard_io.py choose <id> [<id>]  # select the final 1-2 submissions (unselects others)
"""
import json
import os
import sys
from pathlib import Path

import requests

API = os.environ.get("CHALLENGE_API", "")
CHALLENGE = os.environ.get("CHALLENGE_SLUG", "")
SITE = os.environ.get("CHALLENGE_SITE", "")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) mumbai-aq-solution"}
ROOT = Path(__file__).resolve().parents[1]


def _data(resp):
    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError(f"Non-JSON response {resp.status_code}: {resp.text[:300]}")
    data = payload.get("data", payload)
    if isinstance(data, dict) and "errors" in data:
        raise RuntimeError(f"API error: {data['errors']}")
    return data


def signin():
    user, pwd = os.environ.get("CHALLENGE_USER"), os.environ.get("CHALLENGE_PASS")
    if not user or not pwd or not API or not CHALLENGE:
        raise SystemExit("Set CHALLENGE_API, CHALLENGE_SLUG, CHALLENGE_USER and CHALLENGE_PASS.")
    d = _data(requests.post(f"{API}/auth/signin", data={"username": user, "password": pwd}, headers=UA, timeout=60))
    return d["auth_token"]


def _h(tok, current_url=None):
    h = {**UA, "auth-token": tok, "auth_token": tok}
    if current_url:
        h["current-url"] = current_url
    return h


def join(tok):
    r = requests.post(f"{API}/competitions/{CHALLENGE}/participations", headers=_h(tok), timeout=60)
    try:
        return r.json()
    except ValueError:
        return {"status": r.status_code}


def competition(tok):
    return _data(requests.get(f"{API}/competitions/{CHALLENGE}", headers=_h(tok, f"{SITE}/competitions"), timeout=60))


def download(tok, dest=ROOT / "data" / "raw"):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    comp = competition(tok)
    files = comp.get("datafiles", [])
    out = []
    for item in files:
        fn = item["filename"]
        r = requests.get(f"{API}/competitions/{CHALLENGE}/files/{fn}", headers=_h(tok),
                         data={"auth_token": tok}, stream=True, timeout=300)
        r.raise_for_status()
        p = dest / fn
        with open(p, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        out.append((fn, p.stat().st_size))
        print(f"downloaded {fn} ({p.stat().st_size:,} B)")
    return out


def submit(tok, filepath, comment=""):
    filepath = Path(filepath)
    with open(filepath, "rb") as fh:
        r = requests.post(f"{API}/competitions/{CHALLENGE}/submissions",
                          files={"file": (filepath.name, fh, "text/plain")},
                          data={"comment": comment}, params={"auth_token": tok},
                          headers=_h(tok), timeout=300)
    return _data(r)


def status(tok):
    lim = _data(requests.get(f"{API}/competitions/{CHALLENGE}/submissions/limits", headers=_h(tok), timeout=60))
    subs = _data(requests.get(f"{API}/competitions/{CHALLENGE}/submissions", headers=_h(tok),
                              params={"per_page": 50}, timeout=60))
    try:
        mine = _data(requests.get(f"{API}/competitions/{CHALLENGE}/participations/my_participation",
                                  headers=_h(tok), timeout=60))
    except Exception as e:  # noqa
        mine = str(e)
    return {"limits": lim, "my_participation": mine, "submissions": subs}


def leaderboard(tok, n=50):
    rows = _data(requests.get(f"{API}/competitions/{CHALLENGE}/participations", headers=_h(tok),
                              params={"per_page": n}, timeout=60))
    rows = rows if isinstance(rows, list) else rows.get("data", [])
    return [(r.get("public_rank"), r.get("best_public_score"),
             (r.get("user") or {}).get("username") if isinstance(r.get("user"), dict) else (r.get("team") or {}).get("title"),
             r.get("submission_count")) for r in rows]


def choose(tok, ids):
    """Select exactly these submissions as the finals (same PATCH the website uses); unselect any others."""
    subs = _data(requests.get(f"{API}/competitions/{CHALLENGE}/submissions", headers=_h(tok),
                              params={"per_page": 100}, timeout=60))
    subs = subs if isinstance(subs, list) else subs.get("data", [])
    known = {s["id"] for s in subs}
    missing = [i for i in ids if i not in known]
    if missing or not 1 <= len(ids) <= 2:
        raise SystemExit(f"choose 1-2 of your submission ids; unknown: {missing}")
    for s in sorted(subs, key=lambda s: s["id"] in ids):  # unselect first, then select
        want = s["id"] in ids
        if bool(s.get("chosen")) != want:
            r = requests.patch(f"{API}/competitions/{CHALLENGE}/submissions/{s['id']}", json={"chosen": want},
                               headers=_h(tok), timeout=60)
            _data(r)
    subs = _data(requests.get(f"{API}/competitions/{CHALLENGE}/submissions", headers=_h(tok),
                              params={"per_page": 100}, timeout=60))
    subs = subs if isinstance(subs, list) else subs.get("data", [])
    return [(s["id"], s["filename"], s.get("public_score")) for s in subs if s.get("chosen")]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    tok = signin()
    if cmd == "download":
        print(json.dumps(join(tok), default=str)[:300])
        download(tok)
    elif cmd == "submit":
        print(json.dumps(submit(tok, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""), default=str, indent=1)[:2000])
    elif cmd == "status":
        print(json.dumps(status(tok), default=str, indent=1)[:6000])
    elif cmd == "leaderboard":
        for row in leaderboard(tok, int(sys.argv[2]) if len(sys.argv) > 2 else 50):
            print(*row)
    elif cmd == "choose":  # python src/leaderboard_io.py choose <submission_id> [<submission_id>]
        print("chosen now:", choose(tok, sys.argv[2:]))
    elif cmd == "comp":
        c = competition(tok)
        print(json.dumps({k: c.get(k) for k in ["is_participant", "datafiles", "end_time"]}, default=str, indent=1)[:4000])
