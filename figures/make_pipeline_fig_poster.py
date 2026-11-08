"""Pipeline figure (SVG) for the Mumbai PM2.5 forecaster. All mini-panels use real backtest numbers (figdata.json)."""
import html
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "figdata.json"))
W, H = 1600, 960

INK, INK2, MUTED, RULE, SURF = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d4", "#ffffff"
SANS = "Arial, Helvetica, sans-serif"
SERIF = "'Cambria Math', 'Times New Roman', serif"
FAM = {  # stroke (identity), tint fill
    "data": ("#8a8984", "#f4f4f2"),
    "ctx": ("#1baf7a", "#e8f7f0"),
    "fm": ("#2a78d6", "#eaf2fc"),
    "gbm": ("#eb6834", "#fdefe8"),
    "phys": ("#008300", "#e8f4e8"),
    "blend": ("#6250d6", "#efedfb"),
    "dec": ("#d55181", "#fcecf2"),
}
NOVEL = ("#eda100", "#fff4d6", "#6b4800")
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281", "#0d366b"]

out = []
defs = []


def esc(s):
    return html.escape(str(s), quote=False)


def add(s):
    out.append(s)


def text(x, y, s, size=13, weight=400, fill=INK, anchor="start", family=SANS, italic=False, extra=""):
    st = ' font-style="italic"' if italic else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}"{st} {extra}>{esc(s)}</text>')


def _parse(s, off=0.0, fs=1.0, up=False):
    """Mini math markup: _x / _{..} subscript, ^x / ^{..} superscript, `..` upright."""
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
            parts += _parse(grp, off + (0.28 if c == "_" else -0.40) * fs, fs * 0.70, up)
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


def math(x, y, s, size=15, fill=INK, anchor="start"):
    spans, cur = [], 0.0
    for t, off, fs, up in _parse(s):
        dy = (off - cur) * size
        cur = off
        st = "normal" if up else "italic"
        fam = SANS if up and fs < 1 else SERIF
        spans.append(f'<tspan dy="{dy:.2f}" font-size="{fs * size:.2f}" font-style="{st}" font-family="{fam}">{esc(t)}</tspan>')
    add(f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" text-anchor="{anchor}" font-family="{SERIF}">{"".join(spans)}</text>')


def rect(x, y, w, h, fill="none", stroke="none", rx=8, sw=1.5, dash=None, extra=""):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{sw}"{da} {extra}/>')


def line(x1, y1, x2, y2, stroke=INK2, sw=1.5, dash=None, marker=None, extra=""):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#{marker})"' if marker else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{da}{mk} '
        f'stroke-linecap="round" {extra}/>')


def marker(name, color, size=7):
    defs.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="{size}" markerHeight="{size}" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')


def star(cx, cy, r, fill, stroke):
    pts = []
    for k in range(10):
        a = -np.pi / 2 + k * np.pi / 5
        rr = r if k % 2 == 0 else r * 0.45
        pts.append(f"{cx + rr * np.cos(a):.1f},{cy + rr * np.sin(a):.1f}")
    add(f'<polygon points="{" ".join(pts)}" fill="{fill}" stroke="{stroke}" stroke-width="1" stroke-linejoin="round"/>')


def badge(xr, y):
    """'ours' novelty badge, right edge at xr, top at y."""
    w = 56
    rect(xr - w, y, w, 20, fill=NOVEL[1], stroke=NOVEL[0], rx=10, sw=1.2)
    star(xr - w + 12, y + 10, 6.5, NOVEL[0], NOVEL[0])
    text(xr - w + 22, y + 14.5, "novel", 11.5, 700, NOVEL[2])


def module(x, y, w, h, fam, title, novel=False, title_size=15):
    s, f = FAM[fam]
    rect(x, y, w, h, fill=f, stroke=s, rx=9, sw=1.4)
    add(f'<path d="M{x + 9},{y} h0 a0 0 0 0 0 0 0" fill="none"/>')
    rect(x, y + 9, 4.5, h - 18, fill=s, stroke="none", rx=2)  # identity bar
    text(x + 14, y + 21, title, title_size, 700, INK)
    if novel:
        badge(x + w - 8, y + 7)


def chip(x, y, w, h, label, fam, size=12, dash=None, weight=400):
    s, f = FAM[fam]
    rect(x, y, w, h, fill=SURF, stroke=s, rx=h / 2, sw=1.2, dash=dash)
    text(x + w / 2, y + h / 2 + size * 0.36, label, size, weight, INK2, "middle")


def lines_of(x, y, items, size=12.5, gap=16.5, fill=INK2):
    for k, s in enumerate(items):
        text(x, y + k * gap, s, size, 400, fill)


for nm, col in (("ar", INK2), ("arb", FAM["fm"][0]), ("arv", FAM["blend"][0]), ("ard", FAM["dec"][0]),
                ("arg", "#b9b8b3")):
    marker(nm, col)

# ------------------------------------------------------------------ header
text(20, 34, "Regime-aware foundation-model ensemble with a best-of-two hedge", 23, 700)
text(20, 57, "Hourly PM2.5 at 4 Mumbai stations · forecast origin T0, horizon 84 h · private window = 11–12 Nov 2026 "
             "(Diwali +3/+4), leads ≈ 3–51 h from the final run", 13.5, 400, INK2)
# legend (right)
lx = 1150
badge(lx + 56, 20)
text(lx + 64, 34.5, "our component", 12.5, 400, INK2)
line(lx + 170, 30, lx + 200, 30, INK2, 1.6, marker="ar")
text(lx + 206, 34.5, "known-future covariates", 12.5, 400, INK2)
line(lx + 170, 50, lx + 200, 50, INK2, 1.6, dash="4 3", marker="ar")
text(lx + 206, 54.5, "past-only context (≤ T0)", 12.5, 400, INK2)

COLS = [(20, 222, "1", "Data"), (272, 262, "2", "Context & covariates"), (566, 388, "3", "Forecast members (10)"),
        (986, 292, "4", "Regime-aware blending"), (1310, 270, "5", "Best-of-two decision")]
Y0 = 84
for x, w, n, t in COLS:
    add(f'<circle cx="{x + 11}" cy="{Y0 + 10}" r="11" fill="{INK}"/>')
    text(x + 11, Y0 + 14.5, n, 13, 700, SURF, "middle")
    text(x + 29, Y0 + 15, t, 16, 700, INK)
    line(x, Y0 + 30, x + w, Y0 + 30, RULE, 1.2)

TOP = 126

# ------------------------------------------------------------------ column 1: data
x, w = COLS[0][0], COLS[0][1]
cards = [
    ("Station PM2.5", ["CPCB, via OpenAQ", "46 Mumbai series (4 targets)", "fresh hours up to T0"], "target", None),
    ("NWP forecasts, as issued", ["Open-Meteo Previous-Runs", "lead-matched, day 1–3", "BLH, wind u/v, RH, T, rain"],
     "known future", None),
    ("CAMS PM2.5 forecast", ["global atmospheric-", "composition model"], "known future", None),
    ("OpenAQ neighbourhood", ["neighbour PM2.5 stations", "PM10, NO₂, CO, O₃", "station anemometer"], "past only", "4 3"),
    ("Festival calendar", ["Diwali fireworks nights", "Dhanteras … Bhai Dooj"], "known future", None),
]
hs = [116, 116, 98, 116, 98]
y = TOP
DATA_Y = []
for (t, body, tag, dash), h in zip(cards, hs):
    module(x, y, w, h, "data", t, title_size=14.5)
    lines_of(x + 14, y + 41, body, 12.5, 16.5)
    tw = 7.0 * len(tag) + 18
    chip(x + w - tw - 8, y + h - 25, tw, 18, tag, "data", 11, dash=dash)
    DATA_Y.append(y + h / 2)
    y += h + 12
COL_BOTTOM = y - 12

# ------------------------------------------------------------------ column 2: context
x, w = COLS[1][0], COLS[1][1]
y = TOP
module(x, y, w, 72, "ctx", "Cleaning & alignment", title_size=14.5)
lines_of(x + 14, y + 41, ["drop ≤0 / >1000, stuck ≥ 6 h, spikes;", "hour-end IST grid, lead matching"], 12.5)
B_CLEAN = y + 36
y += 84
module(x, y, w, 96, "ctx", "Relative CAMS", novel=True, title_size=14.5)
math(x + 14, y + 46, "r_t = `CAMS`_t / ⟨`CAMS`⟩_{T0−72h … T0}", 15.5)
lines_of(x + 14, y + 68, ["removes year-to-year CAMS drift", "(GBDT recent fold 19.3 → 13.2)"], 12.5)
B_REL = y + 48
y += 108
uh = 214
module(x, y, w, uh, "ctx", "Lagrangian upwind signal", novel=True, title_size=14.5)
# glyph: wind from the left, neighbours j, target i
gx, gy = x + 16, y + 34
rect(gx, gy, 118, 92, fill=SURF, stroke=RULE, rx=6, sw=1)
for k in range(3):
    line(gx + 8, gy + 18 + k * 26, gx + 30, gy + 18 + k * 26, "#b9b8b3", 1.4, marker="arg")
text(gx + 19, gy + 88, "u", 12, 400, MUTED, "middle", SERIF, True)
ti = (gx + 96, gy + 50)
nbs = [((gx + 46, gy + 30), 3.2, "j₁"), ((gx + 54, gy + 78), 1.4, "j₂"), ((gx + 108, gy + 18), 0.0, "j₃")]
for (px, py), wgt, lab in nbs:
    if wgt > 0:
        line(px, py, ti[0] - 7, ti[1] - (ti[1] - py) * 7 / max(1, np.hypot(ti[0] - px, ti[1] - py)), FAM["ctx"][0], wgt,
             marker=None)
    else:
        line(px, py, ti[0], ti[1], "#b9b8b3", 1.0, dash="2 3")
    add(f'<circle cx="{px}" cy="{py}" r="5" fill="{SURF}" stroke="{FAM["ctx"][0]}" stroke-width="1.6"/>')
    text(px - 9 if lab != "j₃" else px + 8, py - 7, lab, 11.5, 400, INK2, "middle", SERIF, True)
add(f'<circle cx="{ti[0]}" cy="{ti[1]}" r="6.5" fill="{FAM["ctx"][0]}"/>')
text(ti[0] + 1, ti[1] + 22, "i", 12, 400, INK2, "middle", SERIF, True)
lines_of(gx + 128, gy + 14, ["smoke reaches", "station i from", "upwind neighbour", "j after travel", "time τ"], 12, 15.5)
math(x + 14, y + 150, "S_i(t) = Σ_j w_{ij} C_j(t − τ_{ij}) / Σ_j w_{ij}", 15)
math(x + 14, y + 174, "w_{ij} = `max`(0, `cos` θ_{ij}) e^{−d_{ij}/ℓ}", 15)
math(x + 14, y + 198, "τ_{ij} = d_{ij} / |u|", 15)
text(x + w - 10, y + 198, "u: anemometer, else NWP", 11.5, 400, MUTED, "end")
B_UP = y + uh / 2
y += uh + 12
module(x, y, w, 64, "ctx", "Fireworks-night flag", title_size=14.5)
lines_of(x + 14, y + 42, ["18:00–02:00 IST, Diwali −2 … +2"], 12.5)
B_FW = y + 32
y += 76
module(x, y, w, COL_BOTTOM - y, "ctx", "Past-only context", title_size=14.5)
lines_of(x + 14, y + 41, ["target history, neighbour median /", "nearest, co-pollutants, station", "wind; never read beyond T0"], 12.5)
B_PO = y + 40

# arrows data -> context
x1, x2 = COLS[0][0] + COLS[0][1] + 3, COLS[1][0] - 3
pairs = [(DATA_Y[0], B_CLEAN, None), (DATA_Y[1] - 12, B_CLEAN + 10, None), (DATA_Y[2], B_REL, None), (DATA_Y[1], B_UP - 30, None), (DATA_Y[3], B_UP + 10, "4 3"),
         (DATA_Y[4], B_FW, None), (DATA_Y[3] + 14, B_PO, "4 3")]
for ya, yb, dash in pairs:
    xm = (x1 + x2) / 2
    add(f'<path d="M{x1},{ya:.1f} C{xm},{ya:.1f} {xm},{yb:.1f} {x2 - 2},{yb:.1f}" fill="none" stroke="{INK2}" '
        f'stroke-width="1.4"{" stroke-dasharray=" + chr(34) + dash + chr(34) if dash else ""} marker-end="url(#ar)"/>')

# ------------------------------------------------------------------ column 3: members
x, w = COLS[2][0], COLS[2][1]
y = TOP
gh = 168
module(x, y, w, gh, "fm", "Zero-shot time-series foundation models")
text(x + 14, y + 40, "past + known-future covariates · mean of predicted quantiles", 12.5, 400, INK2)
cx0, cw = x + 136, 76
for k, hdr in enumerate(["ctx 1024 h", "ctx 2048 h", "+ upwind Sᵢ"]):
    text(cx0 + k * (cw + 8) + cw / 2, y + 66, hdr, 12, 700, INK2, "middle")
for r, mname in enumerate(["Chronos-2", "Timestral-209M"]):
    yy = y + 78 + r * 38
    text(x + 16, yy + 18, mname, 13.5, 700, INK)
    for k in range(3):
        rect(cx0 + k * (cw + 8), yy, cw, 26, fill=SURF, stroke=FAM["fm"][0], rx=13, sw=1.3)
        add(f'<circle cx="{cx0 + k * (cw + 8) + cw / 2}" cy="{yy + 13}" r="4.5" fill="{FAM["fm"][0]}"/>')
text(x + w - 12, y + gh - 10, "6 members", 12, 400, MUTED, "end")
C_FM = y + gh / 2
y += gh + 12
module(x, y, w, 76, "fm", "Chronos-2, fine-tuned")
lines_of(x + 14, y + 41, ["all Mumbai series · 800 steps · data ≤ T0 only",
                          "learns the fireworks response from past Diwalis"], 12.5)
C_FT = y + 38
y += 88
module(x, y, w, 76, "gbm", "LightGBM, direct multi-horizon")
lines_of(x + 14, y + 41, ["raw + ratio target (2 members) · 3 seeds",
                          "relative CAMS, lags, neighbours, co-pollutants"], 12.5)
C_GBM = y + 38
y += 88
ph = COL_BOTTOM - y
module(x, y, w, ph, "phys", "PIBM: physics-informed box model", novel=True)
# glyph: boundary-layer box
bx, by, bw, bh = x + 18, y + 38, 104, 86
add(f'<rect x="{bx}" y="{by + 14}" width="{bw}" height="{bh - 14}" fill="{SURF}" stroke="{FAM["phys"][0]}" '
    f'stroke-width="1.4" stroke-dasharray="5 3" rx="3"/>')
line(bx - 6, by + bh, bx + bw + 6, by + bh, INK2, 1.6)
line(bx + 8, by + 18, bx + 8, by + bh - 4, MUTED, 1.0, marker="arg")
line(bx + 8, by + bh - 4, bx + 8, by + 18, MUTED, 1.0, marker="arg")
math(bx + 14, by + 50, "H_t", 13, INK2)
line(bx + 46, by + bh - 2, bx + 46, by + 50, FAM["phys"][0], 2.2, marker=None)
add(f'<path d="M{bx + 41},{by + 56} L{bx + 46},{by + 47} L{bx + 51},{by + 56} z" fill="{FAM["phys"][0]}"/>')
math(bx + 52, by + bh - 8, "E_s(h)", 12, INK2)
line(bx + bw - 14, by + 40, bx + bw + 16, by + 40, FAM["phys"][0], 2.0)
add(f'<path d="M{bx + bw + 12},{by + 35} L{bx + bw + 21},{by + 40} L{bx + bw + 12},{by + 45} z" fill="{FAM["phys"][0]}"/>')
math(bx + bw + 4, by + 32, "κ_t", 12.5, INK2)
math(bx + 62, by + 30, "C_t", 13, INK)
ex = x + 148
math(ex, y + 56, "C_{t+1} = C_t + e^{ε_s(h,wk)} / H_t − κ_t (C_t − B_t)", 14.5)
math(ex, y + 82, "κ_t = 1 − `exp`(−(u_t/L_s + k_s + λ_s P_t))", 14.5)
math(ex, y + 108, "B_t = β_s · `NB`_0 · (`CAMS`_t / `CAMS`_{72})^{γ_s}", 14.5)
lines_of(ex, y + 134, ["hourly mass balance per station, fitted", "by backpropagation through time; error",
                       "corr. with FMs 0.65 (between FMs: 0.89)"], 12.5)
C_PIBM = y + ph / 2

# bus between context and members
BUS = (COLS[1][0] + COLS[1][1] + COLS[2][0]) / 2
ys_in = [B_CLEAN, B_REL, B_UP, B_FW, B_PO]
ys_out = [C_FM, C_FT, C_GBM, C_PIBM]
line(BUS, min(ys_in + ys_out), BUS, max(ys_in + ys_out), INK2, 2.2)
for ya in ys_in:
    line(COLS[1][0] + COLS[1][1] + 1, ya, BUS, ya, INK2, 1.4)
    add(f'<circle cx="{BUS}" cy="{ya:.1f}" r="2.8" fill="{INK2}"/>')
for yb in ys_out:
    line(BUS, yb, COLS[2][0] - 3, yb, INK2, 1.6, marker="ar")

# ------------------------------------------------------------------ column 4: blending
x, w = COLS[3][0], COLS[3][1]
y = TOP
nh = 266
module(x, y, w, nh, "blend", "NNLS stacking on clean OOF forecasts", title_size=14.5)
lines_of(x + 14, y + 40, ["fit on out-of-fold forecasts at the private",
                          "lead range (3–51 h); one weight set per 3-h", "target-hour block, shrunk 50/50 to global"], 12.3,
         15.5)
names = ["GBM raw", "GBM ratio", "C2", "TS", "C2 fine-tuned", "PIBM", "C2 ctx 2048", "TS ctx 2048", "C2 upwind",
         "TS upwind"]
fams = ["gbm", "gbm", "fm", "fm", "fm", "phys", "fm", "fm", "fm", "fm"]
blocks = D["blocks"]
hx, hy, cwid, chei = x + 104, y + 104, 18.5, 13.2
for c, b in enumerate(sorted(blocks, key=int)):
    text(hx + c * (cwid + 1.5) + cwid / 2, hy - 6, f"{int(b):02d}", 10.5, 400, MUTED, "middle")
vmax = 0.52
for r_, nm in enumerate(names):
    yy = hy + r_ * (chei + 1.5)
    add(f'<circle cx="{x + 16}" cy="{yy + chei / 2}" r="3.6" fill="{FAM[fams[r_]][0]}"/>')
    text(x + 24, yy + chei / 2 + 4, nm, 11.2, 400, INK2)
    for c, b in enumerate(sorted(blocks, key=int)):
        v = blocks[b][r_]
        if v < 0.005:
            fill = "#f1f1ef"
        else:
            fill = BLUE_RAMP[min(len(BLUE_RAMP) - 1, int(v / vmax * len(BLUE_RAMP)))]
        rect(hx + c * (cwid + 1.5), yy, cwid, chei, fill=fill, stroke="none", rx=2)
text(hx, hy + 10 * (chei + 1.5) + 12, "target hour (IST) →", 10.5, 400, MUTED)
# colour key
kx, ky = x + w - 92, hy + 10 * (chei + 1.5) + 4
for k, cc in enumerate(BLUE_RAMP):
    rect(kx + k * 7, ky, 7, 8, fill=cc, stroke="none", rx=0)
text(kx - 4, ky + 8, "0", 10, 400, MUTED, "end")
text(kx + 7 * len(BLUE_RAMP) + 4, ky + 8, "0.5", 10, 400, MUTED)
D_NNLS = y + nh / 2
y += nh + 12
th = COL_BOTTOM - y
module(x, y, w, th, "blend", "Festival tilt", novel=True, title_size=14.5)
math(x + 14, y + 46, "T0 ∈ [D, D+4]:   p̂ = ½ p_{`NNLS`} + ½ p_{`FT`}", 15)
lines_of(x + 14, y + 68, ["post-Diwali the blend over-forecasts ×1.16–1.82;",
                          "the fine-tuned member is the least biased"], 12.3, 15.5)
ex_ = D["examples"]["2024-11-03 23:00"]
px0, py0, pw, ph_ = x + 40, y + 124, w - 56, th - 168
hh = np.array(ex_["h"], float)
ser = {k: np.array(ex_[k], float) for k in ("y", "nnls", "ft", "tilt")}
ymin, ymax = 30, 125
X = lambda h: px0 + (h - 3) / 48 * pw
Y = lambda v: py0 + ph_ - (v - ymin) / (ymax - ymin) * ph_
for gv in (40, 80, 120):
    line(px0, Y(gv), px0 + pw, Y(gv), "#ecebe8", 1)
    text(px0 - 5, Y(gv) + 3.5, str(gv), 10, 400, MUTED, "end")
line(px0, py0 + ph_, px0 + pw, py0 + ph_, MUTED, 1)
for hv in (3, 27, 51):
    text(X(hv), py0 + ph_ + 13, f"{hv} h", 10, 400, MUTED, "middle")
text(px0 - 30, py0 - 8, "µg/m³", 10, 400, MUTED)
sty = {"nnls": (FAM["blend"][0], 1.8, "5 3"), "ft": (FAM["fm"][0], 1.8, None), "y": (INK, 2.0, None),
       "tilt": (FAM["blend"][0], 2.8, None)}
for k in ("nnls", "ft", "y", "tilt"):
    col, sw, da = sty[k]
    d = "M" + " L".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(hh, ser[k]))
    add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"'
        f'{" stroke-dasharray=" + chr(34) + da + chr(34) if da else ""}/>')
# legend row with RMSE (2024, Diwali+2 origin, 5 stations)
ly = y + 104
items = [("obs", INK, 2.0, None, ""), ("NNLS", FAM["blend"][0], 1.8, "5 3", " 19.9"),
         ("FT", FAM["fm"][0], 1.8, None, " 23.6"), ("tilt", FAM["blend"][0], 2.8, None, " 17.0")]
lx_ = x + 14
for lab, col, sw, da, val in items:
    line(lx_, ly - 4, lx_ + 16, ly - 4, col, sw, dash=da)
    text(lx_ + 20, ly, lab + val, 11.2, 700 if lab == "tilt" else 400, INK2)
    lx_ += 20 + 6.1 * len(lab + val) + 10
text(x + 14, y + 94 - 4 + 0, "", 1)
text(x + w - 12, y + th - 10, "example: Diwali+2 origin, 2024, 5 stations", 11, 400, MUTED, "end")
D_TILT = y + 40

# arrows members -> blend
x1, x2 = COLS[2][0] + COLS[2][1] + 3, COLS[3][0] - 3
xm = (x1 + x2) / 2
for ya in (C_FM, C_FT, C_GBM, C_PIBM):
    add(f'<path d="M{x1},{ya:.1f} C{xm},{ya:.1f} {xm},{D_NNLS:.1f} {x2 - 2},{D_NNLS:.1f}" fill="none" stroke="{INK2}" '
        f'stroke-width="1.4" marker-end="url(#ar)"/>')
add(f'<path d="M{x1},{C_FT + 8:.1f} C{xm},{C_FT + 8:.1f} {xm},{D_TILT + 30:.1f} {x2 - 2},{D_TILT + 30:.1f}" fill="none" '
    f'stroke="{FAM["fm"][0]}" stroke-width="1.8" marker-end="url(#arb)"/>')

# NNLS -> tilt (vertical)
line(x + w / 2, TOP + nh + 1, x + w / 2, TOP + nh + 11, FAM["blend"][0], 1.8, marker="arv")

# ------------------------------------------------------------------ column 5: decision
x, w = COLS[4][0], COLS[4][1]
y = TOP
hh5 = 326
module(x, y, w, hh5, "dec", "Best-of-two hedge", novel=True, title_size=14.5)
lines_of(x + 14, y + 40, ["the private score of a team is the better", "of its 2 selected submissions"], 12.3, 15.5)
math(x + 14, y + 84, "(a, b) = `arg min`_{a, b}", 15)
math(x + 34, y + 106, "𝔼_w[ `min`( ε_w(a p̂), ε_w(b p̂) ) ]", 15)
math(x + 14, y + 128, "ε_w: `RMSE of one backtest window`", 12.5, INK2)
lines_of(x + 14, y + 144, ["(one origin, all stations, leads 3–51 h)"], 12, 15)
# histogram of per-window optimal scale
sc = np.array(D["scales"])
edges = np.arange(0.55, 1.551, 0.05)
cnt, _ = np.histogram(np.clip(sc, 0.55, 1.549), edges)
hx0, hy0, hw, hh_ = x + 22, y + 178, w - 36, 92
Xs = lambda s: hx0 + (s - 0.55) / 1.0 * hw
cmax = cnt.max()
for k, cn in enumerate(cnt):
    if cn == 0:
        continue
    bw_ = hw / len(cnt) - 2
    bh_ = cn / cmax * hh_
    xx = Xs(edges[k]) + 1
    add(f'<path d="M{xx:.1f},{hy0 + hh_:.1f} v{-(bh_ - 3):.1f} q0,-3 3,-3 h{bw_ - 6:.1f} q3,0 3,3 v{bh_ - 3:.1f} z" '
        f'fill="#c9c8c3"/>')
line(hx0, hy0 + hh_, hx0 + hw, hy0 + hh_, MUTED, 1)
for tv in (0.6, 0.8, 1.0, 1.2, 1.4):
    text(Xs(tv), hy0 + hh_ + 13, f"{tv:.1f}", 10, 400, MUTED, "middle")
for v, lab in ((0.94, "a = 0.94"), (1.09, "b = 1.09")):
    line(Xs(v), hy0 - 6, Xs(v), hy0 + hh_, FAM["dec"][0], 2.2)
    text(Xs(v) + (-4 if v < 1 else 4), hy0 - 10, lab, 11.5, 700, INK, "end" if v < 1 else "start")
math(hx0 + hw, hy0 + hh_ + 30, "s*_w = ⟨y, p̂⟩ / ‖p̂‖²   `(127 windows)`", 12.5, MUTED, "end")
D_H = y + hh5 / 2
y += hh5 + 12
for k, (lab, kv) in enumerate((("Final 1", "0.94"), ("Final 2", "1.09"))):
    yy = y + k * 56
    rect(x, yy, w, 46, fill=SURF, stroke=FAM["dec"][0], rx=8, sw=1.6)
    # document icon
    add(f'<path d="M{x + 14},{yy + 9} h16 l6,6 v22 h-22 z" fill="{FAM["dec"][1]}" stroke="{FAM["dec"][0]}" '
        f'stroke-width="1.2" stroke-linejoin="round"/>')
    text(x + 46, yy + 20, lab, 13.5, 700, INK)
    text(x + 46, yy + 37, "192 private rows", 11.5, 400, MUTED)
    math(x + w - 14, yy + 29, f"{kv} · p̂", 16, INK, "end")
    line(x + w / 2, yy - 11 if k == 0 else yy - 9, x + w / 2, yy - 1, FAM["dec"][0], 1.8, marker="ard") if k == 0 else None
y += 112
rect(x, y, w, COL_BOTTOM - y, fill=FAM["dec"][1], stroke="none", rx=8)
text(x + 12, y + 20, "LOFO window RMSE, single → best of two", 12.0, 700, INK)
for k, (f, a_, b_) in enumerate((("fall 2024", "15.24", "14.64"), ("fall 2025", "21.49", "20.84"),
                                 ("Sep–Nov 2026", "8.45", "8.09"))):
    text(x + 12, y + 40 + k * 17, f, 12.3, 400, INK2)
    text(x + w - 12, y + 40 + k * 17, f"{a_} → {b_}", 12.3, 700, INK, "end")

# arrows blend -> decision
x1, x2 = COLS[3][0] + COLS[3][1] + 3, COLS[4][0] - 3
xm = (x1 + x2) / 2
add(f'<path d="M{x1},{D_TILT + 10:.1f} C{xm},{D_TILT + 10:.1f} {xm},{D_H:.1f} {x2 - 2},{D_H:.1f}" fill="none" '
    f'stroke="{FAM["blend"][0]}" stroke-width="2.2" marker-end="url(#arv)"/>')
text(xm, D_H - 8, "p̂", 14, 400, FAM["blend"][0], "middle", SERIF, True)

# ------------------------------------------------------------------ bottom band
BY = COL_BOTTOM + 30
line(20, BY - 14, 1580, BY - 14, RULE, 1.2)
x, w = 20, 700
text(x, BY + 10, "Validation protocol", 15, 700)
val = ["Frozen data snapshots (md5); identical folds and daily 23:00 origins:",
       "   fall 2024, fall 2025, Sep–Nov 2026 · one clean out-of-fold set per experiment",
       "Leave-one-fold-out (LOFO) blending; a change is kept only if the LOFO mean",
       "   improves and the most recent fold does not get worse",
       "Rejected by this rule: TimesFM-3, t0-gamma, TimeH2O-v1, Aurora air-pollution,",
       "   multi-NWP ensemble, LightGBM stacker, Kalman bias correction,",
       "   festival-smoke context masking, lead-bucket weights"]
for k, s in enumerate(val):
    if not s.startswith("   "):
        add(f'<circle cx="{x + 4}" cy="{BY + 30 + k * 18 - 4}" r="2.6" fill="{INK2}"/>')
    text(x + 14, BY + 30 + k * 18, s.strip(), 12.8, 400, INK2)

# results bars
rx0, rw = 760, 820
text(rx0, BY + 10, "LOFO RMSE (µg/m³), lead 3–51 h, mean of 3 folds", 15, 700)
text(rx0 + rw, BY + 10, "recent fold", 12.3, 700, INK2, "end")
res = [("Persistence (last value)", 35.364, 13.759, "data"), ("PIBM (physics)", 27.932, 11.657, "phys"),
       ("LightGBM, ratio target", 23.405, 9.603, "gbm"), ("Chronos-2, fine-tuned", 22.847, 9.258, "fm"),
       ("Chronos-2, ctx 2048", 22.729, 9.538, "fm"), ("Timestral + upwind (best single)", 22.266, 9.260, "fm"),
       ("NNLS blend of 10", 21.772, 8.726, "blend"), ("+ festival tilt (final central)", 21.542, 8.726, "blend")]
lab_w, bx0, bmax = 230, rx0 + 232, 470
for k, (lab, v, rv, fam) in enumerate(res):
    yy = BY + 22 + k * 20.5
    bold = k == len(res) - 1
    text(rx0, yy + 11.5, lab, 12.3, 700 if bold else 400, INK if bold else INK2)
    L = v / 36 * bmax
    col = FAM[fam][0] if fam != "data" else "#b9b8b3"
    add(f'<path d="M{bx0},{yy + 1} h{L - 3:.1f} q3,0 3,3 v8 q0,3 -3,3 h{-(L - 3):.1f} z" fill="{col}"'
        f'{" opacity=" + chr(34) + "0.55" + chr(34) if fam == "blend" and not bold else ""}/>')
    text(bx0 + L + 6, yy + 11.5, f"{v:.2f}", 12.3, 700 if bold else 400, INK)
    text(rx0 + rw, yy + 11.5, f"{rv:.2f}", 12.3, 700 if bold else 400, INK, "end")
line(bx0, BY + 18, bx0, BY + 22 + len(res) * 20.5, MUTED, 1)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
       f'<defs>{"".join(defs)}</defs><rect width="{W}" height="{H}" fill="{SURF}"/>' + "\n".join(out) + "</svg>")
(HERE / "pipeline_poster.svg").write_text(svg, encoding="utf-8")
print("wrote", HERE / "pipeline_poster.svg", "bottom of columns", COL_BOTTOM, "band", BY)
