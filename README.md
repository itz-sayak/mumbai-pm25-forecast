# Mumbai hourly PM2.5 forecasting — Google Mumbai Data Challenge

Forecast hourly PM2.5 at 4 CPCB/OpenAQ stations (Malad W, Shivaji Nagar, Borivali E, Byculla), 1–3 days ahead.
Metric: RMSE. Private window = 11–12 Nov 2026 observations.

## Final recipe (`src/predict.py` `RECIPE`, selected by clean ablations, see `experiments.csv`)

| Member | What it is |
|---|---|
| GBDT raw and ratio target | Global direct multi-horizon LightGBM on 46 Mumbai series, using **relative CAMS** (change versus the last 72 h), no season proxies, neighbour and co-pollutant features at the origin |
| Timestral-209M, Chronos-2 (TimesFM-3 evaluated, removed 8 Nov) | Zero-shot, with known-future weather + CAMS + fireworks-night covariates, and past-only neighbour context |
| … 2048-h context variants | Chronos-2 and Timestral with a longer history window |
| … Lagrangian upwind variants | Plus the upwind-neighbour signal (see below) |
| Chronos-2 fine-tuned | 800 steps on all Mumbai series, data up to the origin only |
| **PIBM: physics-informed box model** | An hourly mass-balance ODE per station, fitted by backpropagation through time (see below) |

**Blending:**
- Weights are fit by NNLS on clean out-of-fold (OOF) forecasts, over the lead range the private window has from the actual origin.
- There are separate weights per 3-hour block of target hour, shrunk 50/50 toward the global weights. This fixes the blend's flattened diurnal cycle.

**Leave-one-fold-out blend RMSE**, lead 4–52 h, 3 folds (Sep–Nov 2026, Oct–Nov 2025, Oct–Dec 2024): **21.95**, recent fold **8.78**. For comparison, persistence is 27.7 and the best single model is 22.55.

## Two-pick hedge (`src/hedge.py`)
The challenge ranks each team by the **better** of its 2 selected submissions on the private window. The two finals are therefore two multiplicative legs of the blend, `a·p` and `b·p`, chosen as:

`(a, b) = argmin E_w[ min(RMSE_w(y, a·p), RMSE_w(y, b·p)) ]`

- Each window `w` is one backtest forecast origin (all stations, the private window's lead range).
- Put another way, it is a two-point quantisation of the uncertain window-level multiplier `y/p` under the competition's own loss.
- Optimising squared error instead (2-means) chases spike windows (centroids about 1.0 and 1.5) and does not help the recent fold. So the objective is the window RMSE itself.

| Lead 3–51 h (leave-one-fold-out, mean window RMSE, with the festival tilt below) | Single | Hedged pair |
|---|---|---|
| Fall 2024 | 15.24 | 14.64 |
| Fall 2025 | 21.49 | 20.84 |
| Recent 2026 | 8.45 | 8.09 |

General pair: ×0.94 / ×1.09.

**Festival regime.** The private window (11–12 Nov 2026) is 3–4 days after Diwali.
- From the night of Diwali to 4 days after it, the blend **over-forecasts in every backtest window**: ×1.16 to ×1.82 by origin day in 2024 and 2025. The models carry the fireworks build-up forward while emissions return to normal. GBDT is the worst offender (×1.23 overall, up to ×2.0 in 2025).
- **The observed smoke spike lasts from about 22:00 to 08:00**, peaking at 02:00–04:00 in the morning inversion; the fireworks flag only covers 18:00–02:00. In 2025 the spike reached 7× the pre-festival level at 02:00.
- The **fine-tuned Chronos-2 is the least biased member post-Diwali** (×0.82–1.35; RMSE 18.8 against 23–27 for the others). Fine-tuning on every Mumbai series with the fireworks covariate taught it the fireworks effect from past Diwalis, so it attributes the spike to the nights and returns to baseline when the flag ends.
- **Festival tilt (`hedge.festival_tilt`):** for origins 0–4 days after Diwali, the central forecast is **0.5 × blend + 0.5 × fine-tuned Chronos-2**.
  - It beats the blend in all 10 year × origin-day cells and all 9 stations.
  - Leave-one-fold-out lead 3–51: 21.77 → **21.54** (fall 2024 18.83 → 18.29, fall 2025 37.76 → 37.61, recent unchanged).
  - Mixes of 0.3 and 0.7 give similar results, so 0.5 is not a knife-edge choice.
- **Hedge after the tilt:** the general pair, **×0.94 / ×1.09**.
  - Post-Diwali windows: single 18.1 → hedged 17.1. The earlier setup (blend × festival pair ×0.84 / ×0.99) scored 18.0 on the same comparison, holding out one year at a time.
  - The festival-specific pair is still computed. It now fails its leave-one-year-out test (the bias is gone), so it is not used.
- **Festival member set (`hedge.FEST_SET`, added 9 Nov):** for the same origins (Diwali+0 to +4), the blend before the tilt uses only 5 members: zero-shot Chronos-2 and Timestral (context 1024 and 2048) plus the fine-tuned Chronos-2. GBDT, PIBM and the upwind-neighbour variants are dropped there, since they carry the Diwali smoke forward the most.
  - Post-Diwali, lead 3–51, leave-one-fold-out, tilt on: 17.92 → **17.47**. Better in 8 of 10 year × origin-day cells, including Diwali+2 (the final run's origin day) in both years: 2024 18.22 → 17.22, 2025 15.98 → 14.83.
  - Non-festival origins keep the full recipe, which is better there in 2 of 3 folds. Their forecasts are bit-identical to before.
  - The hedge pair refit at the final lead range is ×0.94 / ×1.11.
  - Found by the pre-COVID rehearsal below; the 5-member set is the set of members whose inputs exist for 2019, not the result of a subset search.
  - Its weights are fit against the ground truth that `blend_select.load` takes from `gbm_base` (no festival-set member stores `y`). That file ends on 2 Nov 2026, so the 2026 Diwali hours are not in the festival weight fit. Refitting with the fresh ground truth (`gbm_rel_nb_cp`, through 10 Nov) was tested on 10 Nov and is worse: the extreme 2026 Diwali-night errors dominate the fit (2024 Diwali+2: 17.2 → 20.0). So the fit is kept as is, and `gbm_base` is included in every run snapshot.
  - Check on real 2026 readings (final Train release, 10 Nov; leads 3–51; tilt on in both):

    | 2026 origin | Rows | 10-member blend | 5-member set |
    |---|---|---|---|
    | Diwali+0 (8 Nov 23:00) | 117 | 82.4 | 99.5 |
    | Diwali+1 (9 Nov 23:00), up to 10 Nov 11:00 | 37 | 44.8 | 44.6 |

    - At Diwali+0, both forecasts are about twice the actual mean (140 and 158 against 75), carrying the Diwali-night context forward.
    - From Diwali+1 both are close to unbiased (97 and 99 against 105) and tie.
    - The final run starts at Diwali+2, where the 5-member set won in both 2024 and 2025. It is kept.
  - Second check at 18:00 on 10 Nov, on the forecasts actually issued by live runs, scored against Train plus OpenAQ readings up to 10 Nov 16:00. Both blends are rebuilt with the current weights; tilt on in both:

    | Run (origin) | Rows | 10-member blend | 5-member set | Issued |
    |---|---|---|---|---|
    | x9fill (9 Nov 14:00, Diwali+1), leads 1–26 | 89 | 45.3 | 44.6 | 45.0 |
    | final_am2 (10 Nov 11:00, Diwali+2), leads 1–5 | 15 | 6.8 | 5.4 | 5.4 |

    - The 5-member set is equal or better at both origins, so it is kept. No method change before the 19:00 freeze.
    - At the Diwali+1 origin the tilt toward the fine-tuned Chronos-2 hurt (no tilt: 41.7), because readings stayed high. Most of that error is one station, Malad, at RMSE 82. At Diwali+2 the tilt helped (no tilt: 6.3). The tilt is kept: it was chosen on the 2024/2025 post-Diwali backtest, and 15 rows can't overturn it.

**Private-window rehearsal on past Diwalis** (research only, a server-side script not included here). Same setup as 11–12 Nov 2026: origin Diwali+2 at 23:00, scored on Diwali+3 and +4, all hours, better of the two hedge legs.

| Year | Stations | Mean actual | Mean forecast | Best of 2 legs | Persistence (24 h) | Previous-week mean |
|---|---|---|---|---|---|---|
| 2025 | the 4 target stations | 20.5 | 30.6 | 13.7 | 41.6 | 65.9 |
| 2024 | XKDR CPCB feeds with the target station names | 65.6 | 60.6 | 16.4 | 21.8 | 29.0 |
| 2019 | 10 MPCB stations (challenge 2018–2020 history) | 33.8 | 46.9 | 22.4 | 42.7 | 27.8 |
| 2018 | Bandra MPCB only | 92.9 | 79.6 | 64.6 | 72.2 | 89.5 |

- Rows use the festival member set. With the full recipe, 2025 scores 14.6 and 2024 17.6.
- 2018 and 2019 are fully out of sample:
  - the fine-tuned Chronos-2 is trained on 2024 to Sep 2026 data with no 2018–2020 data, and without CAMS (no CAMS history for those years);
  - ERA5 weather is used in place of archived forecasts (slightly optimistic);
  - the blend weights, tilt and hedge were all chosen on 2024–2026.
- 2018: a fireworks spike after the Bhai Dooj night (Bandra up to 430 µg/m³ at 01:00 on Diwali+3) dominates the error for every method. No such spike occurred in 2019, 2024 or 2025 (Bhai Dooj night mean 72, 94, 51 against 77, 83, 81 the night before).

**Bhai Dooj scenario: considered and dropped.**
- Calendars disagree on whether Bhai Dooj 2026 falls on 10 Nov or 11 Nov (Diwali +2 or +3).
- An optional high leg with an extra fireworks night on 11 Nov (`run_all.py --fw-hedge`, `FW_EXTRA_NIGHTS`) was tested. It is not used, for two reasons:
  - the foundation models barely respond to an isolated future flag (Chronos-2 night mean 42.6 → 42.7);
  - in 2024 and 2025 the blend still over-forecast Bhai Dooj nights (actual/forecast 0.68–0.84), with no reliable night excess over other post-Diwali nights.

## Novel components
- **Lagrangian upwind neighbour signal:**
  `S_i(t) = Σ_j w_ij C_j(t − τ_ij) / Σ_j w_ij`
  - weight `w_ij = max(0, cos∠(wind, j→i)) · exp(−d_ij/ℓ)`;
  - lag `τ_ij = d_ij/|wind|`;
  - wind comes from the station anemometer (OpenAQ), falling back to the forecast wind.
- **PIBM:**
  `C_{t+1} = C_t + exp(e_s[hour, weekend]) / H_t − κ_t (C_t − B_t)`
  - ventilation `κ_t = 1 − exp(−(u_t/L_s + k_s + λ_s P_t))`;
  - background `B_t = β_s · NB0 · (CAMS_t / CAMS72_0)^γ_s`;
  - H = boundary-layer height, u = wind speed, P = precipitation, all from forecasts.
  - Its errors are only 0.65-correlated with Timestral's, against 0.89 between the foundation models, so it adds diversity to the blend.
- **Relative-CAMS design:** the CAMS level drifted between years (2026: CAMS higher, observations lower). Absolute-CAMS GBDTs overpredicted October 2026 by 10–22 µg/m³; using CAMS as a relative change fixed this (recent-fold GBDT 19.3 → 13.2).
- **Fireworks-night covariate** (Dhanteras … Bhai Dooj, 18:00–02:00).

## Rejected by ablation
| Idea | Result |
|---|---|
| Multi-NWP ensemble (ECMWF/GFS/ICON/AIFS) mean and spread | Worse |
| Sample reweighting | Neutral |
| LightGBM stacker | Overfit |
| Kalman bias correction | Daily-error autocorrelation ≈ 0, so nothing to correct |
| 512-h context | Worse |
| Co-pollutant covariates for the foundation models | Worse on the recent fold |
| Upwind feature in GBDT | Neutral |
| t0-gamma (Oct 2026 open-weights foundation model, with covariates; also with symmetric averaging) | Alone 23.45 / 22.87 versus Timestral 22.27; zero NNLS weight; blend 21.78 → 21.80 |
| Lead-bucket blend weights | 21.78 → 21.84 |
| TimeH2O-v1 (H2O.ai, Oct 2026, fev-bench #1; Python 3.12 worker `src/timeh2o_worker.py`), with 1024-h and 4096-h context | Gets 14–19% NNLS weight (best single model on fall25), but blend 21.772 → 21.785 / 21.804, and the recent fold is worse |
| Microsoft Aurora 0.4° Air Pollution (`src/aurora_ap.py`, from CAMS analyses) | 3–4 Nov RMSE 37.6 (level 1.9× the observations); 9.8 even with an oracle level fix, versus 8.3 for the blend |
| TimesFM-3 members (removed 8 Nov) | Zero weight at lead 3–51; blend 21.784 → 21.772 without them |
| Extra fireworks night for Bhai Dooj (`--fw-hedge`) | No effect: the models ignore an isolated future flag |
| TabPFN-TS (Prior Labs, tabular in-context regression on known covariates; `src/tabpfn_worker.py`), TabPFN-3.5 and the forecasting-tuned TabPFN-TS-3 | Mean error 24.80 / 23.87 versus 22.27 for Timestral; added to the blend: recent fold 8.726 → 8.739 / 8.785, so rejected |
| AutoGluon-TimeSeries 1.6 `best_quality` (offline research probe `src/ag_probe.py`, never part of the solution; zoo: Chronos-2 incl. fine-tune, Toto 2.0, TFT, DeepAR, tabular, ETS/Theta) | Its ensemble scores 24.45 mean / 11.87 recent versus our blend's 21.54 / 8.73; its own pick (TFT, chosen on one validation window) scores 33.25. No AutoGluon model improves our blend when added (best: Toto 2.0, unchanged at 21.542) |
| Best-of-two *portfolio*: each final chosen as (base, scale) from {tilted blend, NNLS blend, fine-tuned Chronos-2, GBDT, foundation-model mean, PIBM} × 0.66–1.30, solved exactly over all pairs | Selects the same pair as the scalar hedge (tilted blend ×0.94 / ×1.09) in every fold and both festival years: the current hedge is already optimal in the larger family |
| Holiday flag for Bhaubeej (11 Nov 2026, Diwali+3, a state holiday) treated like a Sunday | Dry-season weekend effect in Mumbai is +3 % (Sat/Sun vs weekdays, ±1 %), so the flag would move forecasts by about 3 %: not pursued |
| Fine-tuned Chronos-2 also trained on the 2018–19 MPCB history (2 extra pre-COVID Diwalis; `src/k18.py`, configs `ft_fw_nb_nocams`, `ft_fw_nb_nocams_k18`) | CAMS has no 2018–19 history, so CAMS had to be dropped, which costs about 5 points after Diwali (member 18.6 → 23.7). The extra history doesn't recover it (24.9). 2019 rehearsal with the 2019 Diwali left out of fine-tuning: 22.6 → 22.1 |
| Simple models alone: last-24 h mean, 3- and 7-day same-hour means, 48-h EWMA, same hour yesterday, CAMS MOS, their median | Post-Diwali (2024–25, origins Diwali+0…4, leads 3–51) 31.5–55.5 against 16.8 for our forecast; non-festival also worse in every fold |
| Simple models as extra blend members (general and festival member sets) | Festival: worse in every case (17.1–17.3 against 16.7). General: 3-day same-hour mean helps fall 2025 and recent but not fall 2024, and doesn't reach the private window (a festival-regime forecast) |
| From-scratch neural nets: N-HiTS, TiDE, NBEATSx (`src/nf_prep.py`, `src/nf_cv.py`; neuralforecast, fixed hyper-parameters, global over all Mumbai series, same covariates as the fine-tuned Chronos-2) | Alone: post-Diwali 31–42, recent 12.7–15.1 against 16.7 / 9.5. As extra members: post-Diwali worse in every case (17.0–17.3), Diwali+2 2024 worse (19.7 against 17.2); NBEATSx helps non-festival slightly (fall 2024 18.38 → 18.30), which can't reach the private window |
| NASA GEOS-CF composition forecasts (`src/geoscf.py`; v1 archive 2019 to Jan 2026, v2 Aug 2025 to now, OPeNDAP point extracts; runs at least 30 h old at the origin): raw, observed-level × forecast change (`rel`), and its square-root damped version (`rel05`) | Alone: post-Diwali 39–67 against 16.7. As extra members: non-festival fall 2024 18.37 → 18.20 and fall 2025 39.77 → 39.46 (recent equal), but post-Diwali within ±0.05 (16.67–16.76), since GEOS-CF has no fireworks emissions. No gain for the private window, so not used. The live hook in `predict.py` stays inactive unless a `gcf:` member is listed |
| Festival-regime online correction: scale the forecast by (observed ÷ previous-day forecast over the last 24 h)^β, pooled or per station | Yesterday's ratio doesn't predict today's (2019 Diwali+2: 1.22 against realised 0.70; 2024: 0.93 against 1.08). Leave-one-year-out 17.24–17.31 against 17.05 |
| Return-to-baseline festival model: pre-Diwali diurnal profile × decaying post-Diwali excess, τ 12–72 h, as a third component of the festival central forecast | Alone 22–35 against 13–24. Leave-one-year-out (2019, 2024, 2025) picks weight 0 in every fold |
| Festival-smoke hours (20:00–08:00 on the nights Diwali −2 … +2) masked in the foundation models' context (`_mfw` configs, `tsfm.smoke_mask`) | Removes the post-Diwali bias (Chronos-2 ×1.14 → ×0.89) but raises RMSE: zero-shot post-Diwali 27.5 → 28.1, fine-tuned 18.8 → 32.4. Added to the blend: LOFO 21.54 → 21.42, but the recent fold is worse (8.73 → 8.77), so rejected |

## Data
- **Challenge data:** Train, Test, AdditionalHistory (not redistributed here). The 2024 XKDR CPCB series and other Mumbai stations serve as auxiliary series.
- **Cleaning:** values ≤0 or >1000 are dropped, along with stuck sensors (≥6 identical consecutive values) and isolated spikes (>4× the local median and >150).
- **Open-Meteo** (open, no key, listed on ClimateVerse):
  - Previous-Runs API: weather forecasts **as issued**, lead-matched day 1–3;
  - Air-Quality API: CAMS global PM2.5.
- **OpenAQ v3** (free key in `OPENAQ_API_KEY`; the same upstream source as Train.csv):
  - neighbour stations;
  - co-located PM10, NO₂, CO, O₃ and the station anemometer;
  - optional fresh target-station hours newer than Train.csv (`--live`);
  - optional `--fill-gaps` (with `--live --past-from <Train-only tag>`): the target-station hours Train.csv leaves out are also used as context. These are the withheld public window, 7–8 Nov 2026, i.e. Diwali eve and Diwali day. All of them were observed more than two days before the forecast origin.
    - Every row at or before the Train-only run's origin, including the 7–8 Nov public rows, is copied from that run. So gap-filled data only reaches the future (private) rows.
    - Backtest with the same 48-h hole cut into 2024 and 2025 (`_hole` configs, `data.diwali_hole_mask`), post-Diwali origins D+1..4, leads 3–51, tilt on:

      | Context | RMSE, D+1..4 | RMSE, D+2 |
      |---|---|---|
      | Full | 14.55 | 17.26 |
      | With the hole | 16.28 | 18.47 |

      Non-festival origins are unchanged (24.26 vs 24.31). With the hole, the fine-tuned Chronos-2 degrades from 16.5 to 20.0, since it reacts to the spike it can no longer see.
  - All of these are **past-only**: never read beyond the forecast origin (`upto=T0`). There is a **hard cutoff of 2026-11-11 00:00 IST**, also capped at fetch time.
  - The hourly aggregation reproduces Train.csv exactly (MAE 0.000).
- Observed targets are never pasted into the submission.
- Rows before the origin get day-ahead forecasts issued at 23:00 the previous day, blended with weights fitted for leads 1–24 h. This lifts the 5–6 Nov public rows from 10.0 to 8.5.
- The two hedge scales apply only to the private-window rows; every other row keeps the central forecast.

## Validation protocol
- Frozen snapshots (`data/processed/exp/SNAPSHOT*.json`, with md5s), identical folds and origins, one OOF set per experiment.
- GBDT boosting rounds are chosen on an inner window, 30 days before each fold.
- Acceptance rule: the LOFO blend mean improves, and the recent fold is not worse.

## Run
```
pip install -r requirements.txt          # Python 3.11, CUDA GPU (8 GB is enough)
set CHALLENGE_API=... & set CHALLENGE_SLUG=... & set CHALLENGE_USER=... & set CHALLENGE_PASS=... & set OPENAQ_API_KEY=...
python run_all.py --tag final_am2 --download --live   # Train-only: refresh data -> re-run selected experiments -> predict
python run_all.py --tag final_am2f --live --skip-exp --fcst-tag final_am2 --fill-gaps --past-from final_am2   # fallback pair
python run_all.py --tag final --live --skip-exp --fill-gaps --past-from final_am2   # late refresh, freshest OpenAQ hours (~22 min)
python run_all.py --tag rerun --skip-exp --fcst-tag <snapshot tag>   # reproduce from cached inputs
```
- Outputs `submissions/<tag>_PICK_lo.csv` and `<tag>_PICK_hi.csv`, the two submissions to select.
- `<tag>.csv` is the unhedged central forecast.
- OpenAQ hourly data lags about 2 h. A run started around 21:30 IST on 10 Nov forecasts the private window from about 3 h ahead.
- **10 Nov final release:** Train runs to 10 Nov 09:00 and includes 7–8 Nov. On it, `--fill-gaps` adds 0 hours: OpenAQ has no target-station hour that Train lacks, and on the 805 overlapping hours the two agree within 5 µg/m³. So `final_am2f` equals `final_am2` (max difference 3e-14). The tag `final_am2` replaced `final_am`, whose morning inputs were overwritten by a duplicate launch.
- **Fix, 10 Nov:** `tsfm.build_items` skips origins whose 84-h horizon runs past the grid. With `--live`, the cross-validation could otherwise pick an origin after the newest reading (10 Nov 23:00) and fail. PIBM already had the same guard, and every earlier run is unchanged.
- **Fix, 10 Nov 21:47 (final run):** `predict.run` now runs every `RECIPE` member.
  - Before, it ran only members with a non-zero blend weight. At the final run's lead range (5–52), the leave-one-fold-out check picked one global weight vector over the 3-hour blocks (27.45 against 27.58), and both `tsfm_fw_nb` members got weight 0.
  - So they were never run, and the festival member set, which needs them, was silently skipped. The backfill re-blend was skipped as well.
  - That first output was discarded and the run repeated. Runs with block weights (all earlier runs) use the same members in the same order, so their output is unchanged.
- Seeds: LightGBM seeds (42, 7, 2026) with deterministic mode; torch and numpy seed 0.
- The foundation models are zero-shot, except the seeded Chronos-2 fine-tune.

**Reproducibility (code review).**
- **Deterministic fine-tune:** `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms`, cuDNN deterministic, and seeds for python, numpy and torch right before training. Two fine-tunes on the same data gave identical forecasts (max difference 0.000000, 15,372 rows).
- **Frozen inputs:** every `run_all.py` run copies the exact inputs it used (raw data, forecast snapshot, OpenAQ files, and the out-of-fold files behind the blend weights) to `data/snapshots/<tag>/`, with an md5 manifest.
- To reproduce a submission from those frozen inputs, without re-downloading anything:
  ```
  python run_all.py --tag <tag> --replay    # writes submissions/<tag>_replay*.csv and prints the difference to the originals
  ```
- `src/ag_probe.py` is a **research-only** AutoGluon probe, used offline to see which model families work. It is never imported by the pipeline, because AutoML is not permitted in solutions.

## Files
- `src/data.py` — loading and cleaning
- `src/external.py` — Open-Meteo
- `src/openaq_{live,neighbors,copoll}.py` — OpenAQ
- `src/neighbors.py` — neighbour, upwind and co-pollutant context
- `src/features.py` — GBDT samples
- `src/train.py` — GBDT
- `src/tsfm.py` — foundation models and fine-tuning
- `src/pibm.py` — box model
- `src/experiments.py` — ablations
- `src/blend_select.py` — NNLS blend with hour blocks
- `src/hedge.py` — two-pick hedge (general and festival regime)
- `src/predict.py` — final submission
- `src/leaderboard_io.py` — challenge platform API (data download, submit, leaderboard)
