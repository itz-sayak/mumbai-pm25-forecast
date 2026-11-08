"""Paper (CVPR-style) pipeline figure, drawn at final print size for a full-width figure* (6.9 in = 497 pt):
every font size is a true point size. All thumbnails are real data (figdata.json): covariate timeline around the
Diwali+2 origin of 2025, NNLS block weights, the 2024 festival-tilt example, per-window optimal scales, and the
2025 best-of-two example.

python figures/make_pipeline_fig_paper.py  -> figures/pipeline_paper.svg
"""
import html
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "figdata.json"))
W, H = 497, 206

INK, INK2, MUTED, FAINT, SURF, CARD = "#1a1a1a", "#4a4946", "#8a8984", "#d6d4cf", "#ffffff", "#f6f6f3"
SANS = "Helvetica, Arial, sans-serif"
SERIF = "'Times New Roman', Times, serif"
C = {"fm": "#2a6fc4", "fm_t": "#e6effb", "gbm": "#d9612f", "gbm_t": "#fcebe2", "phys": "#2f8a3a", "phys_t": "#e7f3e8",
     "ctx": "#16956a", "ctx_t": "#e3f4ec", "blend": "#5a48c8", "blend_t": "#ecebfa", "dec": "#c24a76",
     "dec_t": "#fae9f0", "nwp": "#16956a", "cams": "#c98500", "fw": "#d23f3f", "hz": "#f1eff9"}
STAR = "#d99a00"
RAMP = ["#dbe8fa", "#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
out, defs = [], []


def esc(s):
    return html.escape(str(s), quote=False)


def add(s):
    out.append(s)


def text(x, y, s, size=6.0, weight=400, fill=INK2, anchor="start", family=SANS, italic=False, ls=None):
    st = ' font-style="italic"' if italic else ""
    lsp = f' letter-spacing="{ls}"' if ls is not None else ""
    add(f'<text x="{x:.2f}" y="{y:.2f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}"{st}{lsp}>{esc(s)}</text>')


def _parse(s, off=0.0, fs=1.0, up=False):
    parts, buf, i = [], "", 0
    while i < len(s):
        c = s[i]
        if c in "_^":
            if buf:
                parts.append((buf, off, fs, up)); buf = ""
            i += 1
            if s[i] == "{":
                depth, j = 1, i + 1
                while depth:
                    depth += {"{": 1, "}": -1}.get(s[j], 0); j += 1
                grp, i = s[i + 1:j - 1], j
            else:
                grp, i = s[i], i + 1
            parts += _parse(grp, off + (0.27 if c == "_" else -0.40) * fs, fs * 0.70, up)
        elif c == "`":
            if buf:
                parts.append((buf, off, fs, up)); buf = ""
            j = s.index("`", i + 1)
            parts += _parse(s[i + 1:j], off, fs, True)
            i = j + 1
        else:
            buf += c; i += 1
    if buf:
        parts.append((buf, off, fs, up))
    return parts


def math(x, y, s, size=7.0, fill=INK, anchor="start"):
    spans, cur = [], 0.0
    for t, off, fs, up in _parse(s):
        dy = (off - cur) * size
        cur = off
        spans.append(f'<tspan dy="{dy:.2f}" font-size="{fs * size:.2f}" font-style="{"normal" if up else "italic"}">'
                     f'{esc(t)}</tspan>')
    add(f'<text x="{x:.2f}" y="{y:.2f}" fill="{fill}" text-anchor="{anchor}" font-family="{SERIF}">{"".join(spans)}</text>')


def rect(x, y, w, h, fill="none", stroke="none", rx=2.5, sw=0.6, dash=None, op=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' fill-opacity="{op}"' if op is not None else ""
    add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}"{o} stroke="{stroke}" '
        f'stroke-width="{sw}"{da}/>')


def line(x1, y1, x2, y2, stroke=INK2, sw=0.6, dash=None, marker=None, cap="round"):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#{marker})"' if marker else ""
    add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{sw}"{da}{mk} '
        f'stroke-linecap="{cap}"/>')


def path(d, stroke=INK2, sw=0.6, fill="none", dash=None, marker=None, op=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#{marker})"' if marker else ""
    o = f' fill-opacity="{op}"' if op is not None else ""
    add(f'<path d="{d}" fill="{fill}"{o} stroke="{stroke}" stroke-width="{sw}"{da}{mk} stroke-linejoin="round" '
        f'stroke-linecap="round"/>')


def polyline(xs, ys, stroke, sw=0.7, dash=None):
    """Polyline with gaps at NaN."""
    seg = []
    for a, b in zip(xs, ys):
        if b is None or (isinstance(b, float) and np.isnan(b)):
            if len(seg) > 1:
                path("M" + " L".join(f"{p:.2f},{q:.2f}" for p, q in seg), stroke, sw, dash=dash)
            seg = []
        else:
            seg.append((a, b))
    if len(seg) > 1:
        path("M" + " L".join(f"{p:.2f},{q:.2f}" for p, q in seg), stroke, sw, dash=dash)


def marker(name, color):
    defs.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5.5" markerHeight="5.5" '
                f'markerUnits="strokeWidth" orient="auto"><path d="M0,0.8 L10,5 L0,9.2 z" fill="{color}"/></marker>')


def star(cx, cy, r=2.6, fill=STAR):
    pts = []
    for k in range(10):
        a = -np.pi / 2 + k * np.pi / 5
        rr = r if k % 2 == 0 else r * 0.45
        pts.append(f"{cx + rr * np.cos(a):.2f},{cy + rr * np.sin(a):.2f}")
    add(f'<polygon points="{" ".join(pts)}" fill="{fill}"/>')


def snowflake(cx, cy, r=2.6, col="#3f8fd8"):
    for k in range(3):
        a = k * np.pi / 3
        dx, dy = r * np.cos(a), r * np.sin(a)
        line(cx - dx, cy - dy, cx + dx, cy + dy, col, 0.55)
        for sgn in (1, -1):
            ex, ey = cx + sgn * dx * 0.62, cy + sgn * dy * 0.62
            for b in (a + 2.4, a - 2.4):
                line(ex, ey, ex + sgn * 0.9 * np.cos(b), ey + sgn * 0.9 * np.sin(b), col, 0.45)


def flame(cx, cy, s=1.0):
    path(f"M{cx:.2f},{cy - 3.6 * s:.2f} C{cx + 2.9 * s:.2f},{cy - 0.6 * s:.2f} {cx + 2.8 * s:.2f},{cy + 3.0 * s:.2f} "
         f"{cx:.2f},{cy + 3.2 * s:.2f} C{cx - 2.8 * s:.2f},{cy + 3.0 * s:.2f} {cx - 2.6 * s:.2f},{cy + 0.2 * s:.2f} "
         f"{cx - 0.9 * s:.2f},{cy - 1.2 * s:.2f} C{cx - 0.6 * s:.2f},{cy + 0.2 * s:.2f} {cx + 0.2 * s:.2f},{cy - 1.5 * s:.2f} "
         f"{cx:.2f},{cy - 3.6 * s:.2f} z", "none", 0, fill="#f07a2a")
    path(f"M{cx:.2f},{cy - 0.4 * s:.2f} C{cx + 1.3 * s:.2f},{cy + 1.0 * s:.2f} {cx + 1.2 * s:.2f},{cy + 2.9 * s:.2f} "
         f"{cx:.2f},{cy + 2.9 * s:.2f} C{cx - 1.2 * s:.2f},{cy + 2.9 * s:.2f} {cx - 1.2 * s:.2f},{cy + 1.2 * s:.2f} "
         f"{cx:.2f},{cy - 0.4 * s:.2f} z", "none", 0, fill="#ffc94a")


def encoder(x, y, w, h, col, tint):
    """Classic encoder trapezoid with layer strokes."""
    path(f"M{x:.2f},{y:.2f} L{x + w:.2f},{y + h * 0.22:.2f} L{x + w:.2f},{y + h * 0.78:.2f} L{x:.2f},{y + h:.2f} z",
         col, 0.6, fill=tint)
    for k in (1, 2, 3):
        xx = x + w * k / 4
        top = y + h * 0.22 * k / 4
        line(xx, top + 1.0, xx, y + h - h * 0.22 * k / 4 - 1.0, col, 0.35)


def card(x, y, w, h, title, sub=None):
    rect(x, y, w, h, fill=CARD, stroke="#e3e1dc", rx=4, sw=0.5)
    text(x + 5, y + 10, title, 7.0, 700, INK)
    if sub:
        text(x + w - 5, y + 10, sub, 5.6, 400, MUTED, "end")


for nm, col in (("ar", INK2), ("arb", C["fm"]), ("arv", C["blend"]), ("ard", C["dec"]), ("arg", "#9b9a95")):
    marker(nm, col)

TOP, BOT = 3, H - 3
PANELS = [(2, 130), (138, 70), (214, 104), (324, 88), (418, 77)]

# =============================================================== (a) covariate timeline
x, w = PANELS[0]
card(x, TOP, w, BOT - TOP, "(a) Covariate timeline")
TL = D["timeline"]
n = len(TL["y"]); i0 = TL["t0_index"]
lx0, lx1 = x + 44, x + w - 5
X = lambda i: lx0 + i / (n - 1) * (lx1 - lx0)
xT0 = X(i0)
ry0, ry1 = TOP + 24, BOT - 15
rect(xT0, ry0 - 6, lx1 - xT0, ry1 - ry0 + 6, fill=C["hz"], rx=1.5)
text((xT0 + lx1) / 2 + 1.5, ry0 - 1.5, "horizon", 5.4, 400, C["blend"], "middle")
text((lx0 + xT0) / 2, ry0 - 1.5, "history", 5.4, 400, MUTED, "middle")
rows = [("PM2.5", "target", TL["y"], INK, True), ("Neighbours", "OpenAQ", TL["nb"], "#7d7c77", True),
        ("Weather", "NWP as issued", TL["blh"], C["nwp"], False), ("CAMS", "PM2.5 forecast", TL["cams"], C["cams"], False),
        ("Fireworks", "calendar", TL["fw"], C["fw"], False)]
rh = (ry1 - ry0) / len(rows)
for k, (lab, sub, ser, col, past_only) in enumerate(rows):
    yt = ry0 + k * rh
    text(x + 5, yt + rh / 2 + 0.2, lab, 5.9, 700, INK)
    text(x + 5, yt + rh / 2 + 6.6, sub, 5.2, 400, MUTED)
    if k:
        line(lx0, yt, lx1, yt, "#e8e6e1", 0.4)
    arr = np.array([np.nan if v is None else v for v in ser], float)
    last = i0 + 1 if past_only else n
    pad = 3.2
    if lab == "Fireworks":
        for i in np.where(arr[:last] > 0)[0]:
            line(X(i), yt + rh - pad, X(i), yt + rh * 0.55, col, 0.75, cap="butt")
        continue
    a = arr[:last]
    lo, hi = np.nanmin(a), np.nanmax(a)
    Y = lambda v: yt + rh - pad - (v - lo) / (hi - lo + 1e-9) * (rh - 2 * pad)
    polyline([X(i) for i in range(last)], [None if np.isnan(v) else Y(v) for v in a], col, 0.75 if k else 0.85)
    if past_only:
        if k == 0:
            math((xT0 + lx1) / 2, yt + rh / 2 + 2.5, "ŷ = ?", 7, C["blend"], "middle")
        else:
            text((xT0 + lx1) / 2, yt + rh / 2 + 2, "not used", 5.2, 400, "#b3b1ab", "middle", italic=True)
line(xT0, ry0 - 6, xT0, ry1, INK, 0.6, dash="1.6 1.2")
math(xT0, ry1 + 7.5, "T_0", 6.6, INK, "middle")
text(lx0, ry1 + 7.5, "−96 h", 5.2, 400, MUTED)
text(lx1, ry1 + 7.5, "+60 h", 5.2, 400, MUTED, "end")

# =============================================================== (b) context
x, w = PANELS[1]
card(x, TOP, w, BOT - TOP, "(b) Context")
# upwind map
mx, my, mw, mh = x + 4, TOP + 16, w - 8, 86
rect(mx, my, mw, mh, fill=SURF, stroke="#e3e1dc", rx=3, sw=0.4)
ti = (mx + mw * 0.66, my + mh * 0.52)
# upwind sector (wind from the west-south-west)
ang0 = np.deg2rad(180 + 12)
r_s = 34
pts = [ti] + [(ti[0] + r_s * np.cos(a), ti[1] + r_s * np.sin(a)) for a in np.linspace(ang0 - 0.55, ang0 + 0.55, 12)]
path("M" + " L".join(f"{p:.2f},{q:.2f}" for p, q in pts) + " z", "none", 0, fill=C["ctx"], op=0.13)
for k, yy in enumerate((my + 14, my + 40, my + 66)):
    path(f"M{mx + 3:.2f},{yy + 4:.2f} C{mx + 18:.2f},{yy - 3:.2f} {mx + 34:.2f},{yy + 5:.2f} {mx + 52:.2f},{yy - 1:.2f}",
         "#b9b7b1", 0.5, marker="arg")
nbs = [((mx + 13, my + 30), 1.4), ((mx + 17, my + 58), 1.0), ((mx + 30, my + 44), 1.6), ((mx + 52, my + 16), 0.0),
       ((mx + 56, my + 76), 0.0)]
for (px, py), wgt in nbs:
    if wgt:
        line(px, py, ti[0], ti[1], C["ctx"], wgt)
    else:
        line(px, py, ti[0], ti[1], "#c9c7c1", 0.45, dash="1 1.3")
for (px, py), wgt in nbs:
    add(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.1" fill="{SURF}" stroke="{C["ctx"] if wgt else "#b9b7b1"}" '
        f'stroke-width="0.7"/>')
add(f'<circle cx="{ti[0]:.2f}" cy="{ti[1]:.2f}" r="4.2" fill="{C["ctx"]}" fill-opacity="0.18"/>')
add(f'<circle cx="{ti[0]:.2f}" cy="{ti[1]:.2f}" r="2.6" fill="{C["ctx"]}"/>')
math(ti[0] + 3.5, ti[1] + 8.5, "i", 6.4, INK)
math(mx + 4, my + mh - 4, "u", 6.4, MUTED)
star(mx + mw - 5.5, my + 5.5, 2.4)
yb = my + mh + 9
text(x + 5, yb, "Upwind signal", 6.0, 700, INK)
math(x + 5, yb + 9.5, "S_i(t) = Σ_j ŵ_{ij} C_j(t − τ_{ij})", 6.3)
text(x + 5, yb + 17.5, "wind-aligned, time-lagged", 5.2, 400, MUTED)
# relative CAMS chip
yc = yb + 24
rect(x + 4, yc, w - 8, 25, fill=SURF, stroke="#e3e1dc", rx=3, sw=0.4)
star(x + 9.5, yc + 6.6, 2.3)
text(x + 14, yc + 8.6, "Relative CAMS", 6.0, 700, INK)
math(x + 9, yc + 19.5, "r_t = c_t / c̄_{t−72:t}", 6.8)
yd = yc + 30
rect(x + 4, yd, w - 8, BOT - 5 - yd, fill=SURF, stroke="#e3e1dc", rx=3, sw=0.4)
text(x + 9, yd + 8.6, "Cleaning", 6.0, 700, INK)
text(x + 9, yd + 15.6, "stuck sensors, spikes;", 5.2, 400, MUTED)
text(x + 9, yd + 22.0, "NWP lead-matched", 5.2, 400, MUTED)

# =============================================================== (c) forecasters
x, w = PANELS[2]
card(x, TOP, w, BOT - TOP, "(c) Forecasters", "10 members")
tw, th_ = (w - 12) / 2, 52
tiles = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2)]
pos = {k: (x + 4 + c * (tw + 4), TOP + 16 + r * (th_ + 4)) for k, (c, r) in enumerate(tiles)}


def tile(k, col_t, col_s):
    tx, ty = pos[k]
    rect(tx, ty, tw, th_, fill=SURF, stroke=col_s, rx=3, sw=0.5)
    return tx, ty


TILE_C = {}
for k, (name, mult) in enumerate((("Chronos-2", "×3"), ("Timestral", "×3"), ("Chronos-2 FT", "×1"))):
    tx, ty = tile(k, C["fm_t"], C["fm"])
    encoder(tx + 10, ty + 8, 27, 24, C["fm"], C["fm_t"])
    if k < 2:
        snowflake(tx + tw - 7, ty + 7.5)
    else:
        flame(tx + tw - 7, ty + 8)
    text(tx + tw / 2, ty + 41, name, 6.0, 700, INK, "middle")
    text(tx + tw / 2, ty + 48, "ctx 1k / 2k / +S" if k < 2 else "all Mumbai series", 5.0, 400, MUTED, "middle")
    text(tx + 4, ty + 8, mult, 5.6, 700, C["fm"])
    TILE_C[k] = (tx + tw, ty + th_ / 2)
# LightGBM tile: tree glyph
tx, ty = tile(3, C["gbm_t"], C["gbm"])
cxr, cyr = tx + 23, ty + 9
nodes = [(cxr, cyr), (cxr - 9, cyr + 10), (cxr + 9, cyr + 10), (cxr - 14, cyr + 20), (cxr - 4, cyr + 20),
         (cxr + 4, cyr + 20), (cxr + 14, cyr + 20)]
for a_, b_ in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)):
    line(*nodes[a_], *nodes[b_], C["gbm"], 0.55)
for k_, (px, py) in enumerate(nodes):
    if k_ < 3:
        add(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.0" fill="{C["gbm_t"]}" stroke="{C["gbm"]}" stroke-width="0.6"/>')
    else:
        rect(px - 2.2, py - 1.6, 4.4, 3.2, fill=C["gbm"], rx=0.6)
flame(tx + tw - 7, ty + 8)
text(tx + tw / 2, ty + 41, "LightGBM", 6.0, 700, INK, "middle")
text(tx + tw / 2, ty + 48, "raw + ratio target", 5.0, 400, MUTED, "middle")
text(tx + 4, ty + 8, "×2", 5.6, 700, C["gbm"])
TILE_C[3] = (tx + tw, ty + th_ / 2)
# PIBM tile: box model glyph
tx, ty = tile(4, C["phys_t"], C["phys"])
bx, by, bw_, bh_ = tx + 9, ty + 7, 26, 24
rect(bx, by + 3, bw_, bh_ - 3, fill=C["phys_t"], stroke=C["phys"], rx=1, sw=0.5, dash="1.8 1")
line(bx - 2, by + bh_, bx + bw_ + 2, by + bh_, INK2, 0.6)
line(bx + 9, by + bh_ - 0.5, bx + 9, by + 10, C["phys"], 0.8)
path(f"M{bx + 7.4},{by + 11.4} L{bx + 9},{by + 8} L{bx + 10.6},{by + 11.4} z", "none", 0, fill=C["phys"])
line(bx + bw_ - 6, by + 11, bx + bw_ + 4, by + 11, C["phys"], 0.8)
path(f"M{bx + bw_ + 3},{by + 9.4} L{bx + bw_ + 6.2},{by + 11} L{bx + bw_ + 3},{by + 12.6} z", "none", 0, fill=C["phys"])
line(bx + 2.5, by + 5, bx + 2.5, by + bh_ - 1.5, MUTED, 0.35)
math(bx + 4, by + 20, "H", 5.4, INK2)
math(bx + 14, by + 9.6, "C", 5.6, INK)
star(tx + tw - 7, ty + 7.5, 2.4)
text(tx + tw / 2, ty + 41, "PIBM", 6.0, 700, INK, "middle")
text(tx + tw / 2, ty + 48, "physics box model", 5.0, 400, MUTED, "middle")
text(tx + 4, ty + 8, "×1", 5.6, 700, C["phys"])
TILE_C[4] = (tx + tw, ty + th_ / 2)
# legend tile
tx, ty = pos[5]
rect(tx, ty, tw, th_, fill="none", stroke="none")
for k_, (lab, fn) in enumerate((("frozen", lambda a, b: snowflake(a, b)), ("trained", lambda a, b: flame(a, b, 0.9)),
                                ("ours", lambda a, b: star(a, b, 2.4)))):
    fn(tx + 7, ty + 12 + k_ * 12)
    text(tx + 13, ty + 14.2 + k_ * 12, lab, 5.6, 400, INK2)

# =============================================================== (d) blending
x, w = PANELS[3]
card(x, TOP, w, BOT - TOP, "(d) Regime-aware blend")
text(x + 5, TOP + 21, "Hour-block NNLS", 6.0, 700, INK)
fams = ["gbm", "gbm", "fm", "fm", "fm", "phys", "fm", "fm", "fm", "fm"]
blocks = D["blocks"]
hx, hy, cwid, chei, g = x + 11, TOP + 26, 8.2, 4.1, 0.7
for r_, fm in enumerate(fams):
    yy = hy + r_ * (chei + g)
    add(f'<circle cx="{x + 7:.2f}" cy="{yy + chei / 2:.2f}" r="1.25" fill="{C[fm]}"/>')
    for c_, b in enumerate(sorted(blocks, key=int)):
        v = blocks[b][r_]
        fill = "#ecebe7" if v < 0.005 else RAMP[min(len(RAMP) - 1, int(v / 0.52 * len(RAMP)))]
        rect(hx + c_ * (cwid + g), yy, cwid, chei, fill=fill, rx=0.7)
hy_end = hy + 10 * (chei + g)
text(hx, hy_end + 5.4, "00 h", 5.0, 400, MUTED)
text(hx + 8 * (cwid + g) - g, hy_end + 5.4, "21 h", 5.0, 400, MUTED, "end")
NNLS_MID = (x, hy + 25)
# festival tilt
ty0 = hy_end + 12
rect(x + 4, ty0, w - 8, BOT - 5 - ty0, fill=SURF, stroke="#e3e1dc", rx=3, sw=0.4)
star(x + 9.5, ty0 + 6.6, 2.3)
text(x + 14, ty0 + 8.6, "Festival tilt", 6.0, 700, INK)
math(x + 8, ty0 + 19, "p̂ = ½ p_{`NNLS`} + ½ p_{`FT`}", 6.8)
text(x + 8, ty0 + 26.5, "origin within 4 days of Diwali", 5.2, 400, MUTED)
ex = D["examples"]["2024-11-03 23:00"]
hh = np.array(ex["h"], float)
px0, py0, pw, ph = x + 8, ty0 + 38, w - 16, BOT - 5 - ty0 - 46
ymin, ymax = 35, 125
Xp = lambda h_: px0 + (h_ - 3) / 48 * pw
Yp = lambda v: py0 + ph - (v - ymin) / (ymax - ymin) * ph
line(px0, py0 + ph, px0 + pw, py0 + ph, FAINT, 0.4)
for k, col, sw, da in (("nnls", C["blend"], 0.6, "1.8 1.2"), ("y", INK, 0.75, None), ("tilt", C["blend"], 1.05, None)):
    path("M" + " L".join(f"{Xp(a):.2f},{Yp(b):.2f}" for a, b in zip(hh, ex[k])), col, sw, dash=da)
lx = x + 8
for lab, col, sw, da in (("obs", INK, 0.75, None), ("NNLS", C["blend"], 0.6, "1.8 1.2"), ("tilt", C["blend"], 1.05, None)):
    line(lx, ty0 + 33, lx + 6, ty0 + 33, col, sw, dash=da)
    text(lx + 7.5, ty0 + 34.8, lab, 5.2, 700 if lab == "tilt" else 400, INK2)
    lx += 7.5 + 3.0 * len(lab) + 6
TILT_IN = (x + 4, ty0 + 18)
TILT_OUT = (x + w - 4, ty0 + 18)

# =============================================================== (e) best-of-two
x, w = PANELS[4]
card(x, TOP, w, BOT - TOP, "(e) Best-of-two")
star(x + 9.5, TOP + 19, 2.3)
text(x + 14, TOP + 21, "Two-leg hedge", 6.0, 700, INK)
math(x + 6, TOP + 31.5, "`min`_{a,b} 𝔼_w `min`(ε_w^a, ε_w^b)", 6.6)
sc = np.array(D["scales"])
edges = np.arange(0.70, 1.401, 0.05)
cnt, _ = np.histogram(np.clip(sc, 0.70, 1.399), edges)
hx0, hy0, hw, hh_ = x + 7, TOP + 44, w - 14, 34
Xs = lambda s: hx0 + (s - 0.70) / 0.70 * hw
for k, cn in enumerate(cnt):
    if cn:
        bw2 = hw / len(cnt) - 0.8
        bh2 = cn / cnt.max() * hh_
        rect(Xs(edges[k]) + 0.4, hy0 + hh_ - bh2, bw2, bh2, fill="#cfcdc7", rx=0.6)
line(hx0, hy0 + hh_, hx0 + hw, hy0 + hh_, MUTED, 0.4)
for tv in (0.8, 1.0, 1.2):
    text(Xs(tv), hy0 + hh_ + 5.6, f"{tv:.1f}", 5.0, 400, MUTED, "middle")
for v, lab in ((0.94, "a"), (1.09, "b")):
    line(Xs(v), hy0 - 2, Xs(v), hy0 + hh_, C["dec"], 0.85)
    math(Xs(v), hy0 - 3.5, lab, 6.4, INK, "middle")
math(x + w - 6, hy0 + hh_ + 12.5, "window scale s*_w", 5.6, MUTED, "end")
HEDGE_IN = (x, TOP + 30)
# legs example (2025 Diwali+2, rain wash-out): low leg wins
L = D["legs"]["2025-10-23 23:00"]
hh2 = np.array(L["h"], float)
p = np.array(L["p"]); yv = np.array(L["y"])
ly0 = hy0 + hh_ + 20
rect(x + 4, ly0, w - 8, BOT - 5 - ly0, fill=SURF, stroke="#e3e1dc", rx=3, sw=0.4)
qx0, qy0, qw, qh = x + 8, ly0 + 7, w - 16, BOT - 5 - ly0 - 32
lo_, hi_ = 0, max(np.max(1.09 * p), np.max(yv)) * 1.04
Xq = lambda h_: qx0 + (h_ - 3) / 48 * qw
Yq = lambda v: qy0 + qh - (v - lo_) / (hi_ - lo_) * qh
band = "M" + " L".join(f"{Xq(a):.2f},{Yq(1.09 * b):.2f}" for a, b in zip(hh2, p)) + " L" + \
       " L".join(f"{Xq(a):.2f},{Yq(0.94 * b):.2f}" for a, b in zip(hh2[::-1], p[::-1])) + " z"
path(band, "none", 0, fill=C["dec"], op=0.10)
path("M" + " L".join(f"{Xq(a):.2f},{Yq(1.09 * b):.2f}" for a, b in zip(hh2, p)), C["dec"], 0.6, dash="1.8 1.2")
path("M" + " L".join(f"{Xq(a):.2f},{Yq(0.94 * b):.2f}" for a, b in zip(hh2, p)), C["dec"], 0.95)
path("M" + " L".join(f"{Xq(a):.2f},{Yq(b):.2f}" for a, b in zip(hh2, yv)), INK, 0.7)
line(qx0, qy0 + qh, qx0 + qw, qy0 + qh, FAINT, 0.4)
yl = qy0 + qh + 7.5
line(x + 8, yl - 1.8, x + 14, yl - 1.8, C["dec"], 0.95)
math(x + 16, yl, "final 1 = a p̂", 6.0, INK)
line(x + 8, yl + 6.4, x + 14, yl + 6.4, C["dec"], 0.6, dash="1.8 1.2")
math(x + 16, yl + 8.2, "final 2 = b p̂", 6.0, INK)
text(x + 8, yl + 16.0, "score = better of the two", 5.2, 400, MUTED)

# =============================================================== connectors
def badge_arrow(xm, ym):
    add(f'<circle cx="{xm:.2f}" cy="{ym:.2f}" r="4.4" fill="{SURF}" stroke="#cfcdc7" stroke-width="0.5"/>')
    path(f"M{xm - 1.6:.2f},{ym - 2.2:.2f} L{xm + 1.4:.2f},{ym:.2f} L{xm - 1.6:.2f},{ym + 2.2:.2f}", INK2, 0.8)


for (xa, wa), (xb, _) in zip(PANELS[:-1], PANELS[1:]):
    badge_arrow((xa + wa + xb) / 2, BOT - 42)
# FT member -> festival tilt (blue)
fx, fy = TILE_C[2]
tx_, ty_ = TILT_IN
path(f"M{fx + 0.5:.2f},{fy:.2f} C{fx + 10:.2f},{fy:.2f} {tx_ - 9:.2f},{ty_:.2f} {tx_ - 0.8:.2f},{ty_:.2f}", C["fm"], 0.7,
     marker="arb")
# tilt -> hedge (violet)
ox, oy = TILT_OUT
hx_, hy_ = HEDGE_IN
path(f"M{ox + 0.5:.2f},{oy:.2f} C{ox + 7:.2f},{oy:.2f} {hx_ - 7:.2f},{hy_:.2f} {hx_ + 4:.2f},{hy_:.2f}", C["blend"], 0.8,
     marker="arv")
math((ox + hx_) / 2 - 3.2, (oy + hy_) / 2 + 1, "p̂", 6.6, C["blend"], "middle")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" viewBox="0 0 {W} {H}">'
       f'<defs>{"".join(defs)}</defs><rect width="{W}" height="{H}" fill="{SURF}"/>' + "\n".join(out) + "</svg>")
(HERE / "pipeline_paper.svg").write_text(svg, encoding="utf-8")
print("wrote", HERE / "pipeline_paper.svg")
