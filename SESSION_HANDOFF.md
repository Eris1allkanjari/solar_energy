# Session handoff — solar forecasting thesis

Everything established in the previous working session, so a new session can
pick up without the history. Read "Current state" first.

## Project in one paragraph

Master's thesis (Eris Allkanjari, advisor Zsolt Saffer): comparing
autoregressive models (ARIMA, SARIMA, ARIMAX, SARIMAX) with recurrent networks
(LSTM, GRU) for **one-hour-ahead** PV energy forecasting, against persistence
and seasonal-naive-24h baselines. PV target: aggregated rooftop production in
Utrecht (Visser et al.), 175 systems at 1-minute resolution, 2014-2017. Weather:
KNMI station De Bilt (`0-20000-0-06260`). Temporal split 65 / 15 / 20 (train /
validation / test), no shuffling; model selection on validation only, test used
once.

---

## Current state — read this first

### All results are stale; everything must be rerun

Two changes invalidated every existing result:

1. **Timestamp alignment fix.** The weather was one hour behind the PV target
   (see "Alignment bug"). Protocol tags were bumped, so old tuning results are
   refused.
2. **Switch to datasets built from KNMI 10-minute data** (see "Datasets").
   Dataset names are now part of every protocol tag.

Every number in "Findings" below comes from the **old validated-hourly dataset,
before the alignment fix**. Use them only as expectations, not as results.

### Datasets

| name | weather source | resolution | status | results folder |
|---|---|---|---|---|
| `hourly_from_10min` | KNMI 10-min, aggregated to hours by us | 1 h | **in use, default** | `src/experiments/results/` |
| `10min` | KNMI 10-min | 10 min | **in use** | `src/experiments/results_10min/` |
| `hourly_validated` | KNMI validated hourly product | 1 h | kept, not used | `src/experiments/results_hourly_validated/` |

- Select with `--dataset NAME` on any script, or `$env:SOLAR_DATASET = "10min"`.
  The flag overrides the environment variable. Implemented in
  `src/configs/dataset_selection.py`, which reads `sys.argv` at import time
  because every experiment binds its constants before argparse runs.
- `src/experiments/results/` still holds stale pre-fix outputs from the
  validated dataset; reruns overwrite them.
- **At 10 minutes, models forecast 10 minutes ahead** (one step), which the user
  chose deliberately. That is an easier task than the hourly one, so 10-minute
  errors are **not comparable** with hourly ones.

### What to run next (the user runs these, not the assistant)

```
python script.py
python -m src.experiments.ar_parameter_tuning_run --dataset 10min --models sarimax sarima arima --probe-steps 200
```

Step 1 builds both datasets; it is required, because the existing 10-min file
lacks the new `minute_rows` column. Expect alignment peaks at lag 0, a PV
difference around 1e-13, and the agreement table (temperature 100%, radiation
99.7%, wind speed 99.7%, cloud 93.5%, humidity 64.8%). Step 2 times SARIMAX at
`s=144` before committing days of compute.

Then the README's run order twice: once without a flag (`hourly_from_10min`),
once with `--dataset 10min` on each command. Core steps: `parameter_tuning_run`,
`ar_parameter_tuning_run`, `feature_significance_run`,
`final_model_comparison_run`, `summary_plots_run`. The others (feature ablation,
daily periodicity, periodicity ablation, loss comparison, computational
efficiency, seed sweep, and the three MAPE diagnostics in `src/analysis/`) also
support both datasets.

### Git

Last commit `3aec380` ("updated mape category breakdown"). About 33 modified and
3 new files are **uncommitted**, covering the alignment fix, the 10-minute
pipeline and dataset selection. `README.md` and every `*.csv` are gitignored;
result PNGs are tracked. `utrecht/utrecht.py` shows as modified only because of
line endings.

---

## Working preferences (from the user)

- **Do not run pipeline commands** (dataset builds, experiments, long runs); show
  the commands and let the user run them. Code-level checks are fine.
- Keep answers short; when asked for emails, keep them short and simple.
- Prefer explicit names over ambiguous ones. A past confusion came from
  `day_nmae` meaning two things.
- Verify claims against code and data rather than from memory.

---

## Methodology as it stands

### Data pipeline

| step | script | output |
|---|---|---|
| PV 1-min → interval energy | `utrecht/pv_utils.py [--freq 1h 10min]` | `pv_hourly_2014_2017.csv`, `pv_10min_2014_2017.csv` |
| KNMI hourly (legacy) | `utrecht/utrecht.py` | `knmi_hourly_2014_2017.csv` |
| KNMI 10-min | `utrecht/knmi_10min.py` (needs `KNMI_API_KEY`) | `knmi_10min_2014_2017.csv` |
| merge | `python script.py [--resolution ...]` | `utrecht/processed_data/utrecht_pv_data_{10min,hourly_from_10min}.csv` (+ legacy `utrecht_pv_data.csv`) |

- PV power (W per system per minute) → `/60` → Wh → sum → `/1000` → kWh.
  Missing minutes count as zero, which makes outages look like low production;
  `observation_fraction` records the coverage.
- **Causal preprocessing:** `clean_data` uses forward fill only. `bfill` was
  removed after confirming no leading gap exists, so the thesis sentence on
  causal imputation now matches the code exactly.
- **`hourly_from_10min` aggregation** mirrors KNMI's own hourly definitions:
  PV and sunshine are summed, radiation averaged, snapshot variables take the
  last 10-min value of the hour, and coverage is weighted by minute rows. It is
  aggregated from the finished 10-min dataset, so both share identical readings
  and gap fills.
- **10-min gap fill:** station gaps are filled from the latest *completed*
  hourly value at forecast issue time, which is causal and at most about 70
  minutes stale. Flagged in `weather_from_hourly`. 549 of 567 gaps filled;
  18 remain (the 2016-03-07 outage and the final interval) and are forward
  filled by `clean_data`.

### Selection and evaluation

- Rolling-origin one-step-ahead forecasting.
- Metrics aggregated per **calendar month** and averaged (partition, not
  smoothing), so sunny months don't dominate. No smoothing anywhere (verified).
- `select_robust_candidate`: within 2% of the best monthly MAE → lower RMSE
  (plus BIC for AR within a window length) → prefers the **longer** `seq_len`.
  So describe it in the thesis as favouring stability, *not* simplicity.
- AR fits that don't converge are rejected. RNNs use seeds (42, 123, 2026),
  with results averaged across seeds.
- Loss = MAE, chosen over Huber on validation (−13.5% validation MAE, about 4×
  lower seed variance). Scaling: MinMax target, StandardScaler features, fitted
  on train only.
- High-quality mask: `observation_fraction ≥ 0.5`.

### Reported metric set (advisor's specification)

File: `final_model_reported_metrics.csv`, with periods defined by **clock hour**:

| period | hours | reported |
|---|---|---|
| ramp | 06-09 + 15-17 | MAPE |
| midday | 10-14 | MAPE |
| day / night | 06-17 / 18-05 | MAE, NMAE |
| whole | all | MAE, NMAE, RMSE |

- MAPE is never reported for night or the whole period.
- MAPE only scores hours above 5 kWh per hour (0.83 kWh per 10-min interval).
- A clock-hour night is not dark: hours 04, 05 and 18 carry summer production,
  so `night_nmae` is about 0.22, not about 0.06.
- **NMAE** = MAE / (training max − training min) × 100. Min is exactly 0 (half
  the hours are dark), so the min-max form equals MAE / training peak. Hourly
  divisor: 313.07 kWh. The same constant is used for every split, taken from
  training only. At 10 minutes the divisor is the peak 10-minute energy.
- Retired: all solar-elevation-based metrics (`day_mape`, `day_period_*`,
  `complement_*`, `day_wmape`, `day_mdape`) and four CSVs. Solar elevation
  survives only in the three diagnostic scripts in `src/analysis/`.

### Other code facts

- Features: 5 weather columns (radiation, cloud, temperature, wind speed,
  humidity) plus `hour_sin`/`hour_cos`, now built from the **fractional hour**.
  The feature-significance group `hour_of_day` is exactly that pair.
- Wind direction: code `990` (variable) and calm (speed 0) → encoded as no
  direction. Used only by feature ablation.
- Neural windows are cut on demand by `WindowDataset`: 13 MB instead of
  13.1 GB at 10 minutes. Training at 10 minutes samples 1/6 of windows per
  epoch. Proven bit-identical to the old hourly training.
- AR tuning checkpoints after every configuration and resumes;
  `--probe-steps N` estimates runtime.
- Seed-sweep cache is reused only under matching protocols.
- `--plot-only` (feature significance) and `--tables-only` (final comparison)
  redraw outputs without refitting.

---

## Findings (pre-fix, validated hourly data — expectations only)

### Model comparison, last full run

| model | MAE | RMSE | ramp MAPE | midday MAPE | NMAE | day NMAE | night NMAE |
|---|---|---|---|---|---|---|---|
| gru | 6.329 | 13.265 | 28.41 | 25.92 | 2.02 | 3.81 | 0.24 |
| lstm | 6.358 | 12.893 | 28.60 | 25.41 | 2.03 | 3.84 | 0.22 |
| sarima | 7.076 | 14.357 | 35.90 | 28.09 | 2.26 | 4.30 | 0.23 |
| sarimax | 7.239 | 14.448 | 36.06 | 28.15 | 2.31 | 4.32 | 0.30 |
| arima | 7.914 | 16.026 | 37.95 | 27.14 | 2.53 | 4.78 | 0.28 |
| arimax | 8.082 | 15.987 | 38.18 | 26.44 | 2.58 | 4.82 | 0.35 |
| persistence | 11.370 | 20.359 | 65.18 | 28.48 | 3.63 | 6.62 | 0.65 |
| seasonal_naive_24h | 17.213 | 35.664 | 76.49 | 87.56 | 5.50 | 10.80 | 0.20 |

- **GRU ≈ LSTM** (seed sweep p = 0.154); both beat every AR model and the
  baselines. GRU is about 44% better than persistence on NMAE.
- `ramp_mape > midday_mape` for every model except `seasonal_naive_24h`.
- NMAE is about 21% higher if divided by the test-era peak (259.5 kWh) rather
  than the training peak, because the fleet shrinks from 152 to 96 systems.
  State the divisor wherever NMAE is quoted, and quote **day NMAE**, not whole.

### Why MAPE is high (resolved)

- **Arithmetic, not model failure.** APE falls monotonically with production
  (≈60% at 5-10 kWh down to 7% at 160+ kWh), while absolute error *rises*.
  Hours with 5-20 kWh are 19% of hours but 36% of MAPE.
- **Not a time-of-day effect:** per-clock-hour contributions are flat
  (1.1-2.3 points).
- Category breakdown (`mape_category_breakdown_run.py`):

  | category | % of hours | % of MAPE |
  |---|---|---|
  | cloudy day | 71.8 | 68.5 |
  | dawn | 6.5 | 12.9 |
  | technical failure | 5.8 | 9.8 |
  | clear day | 15.2 | 7.3 |
  | dusk | 0.7 | 1.5 |

  Cloudy hours are *not* disproportionate (ratio 0.95); they dominate by count.
  Dawn and dusk are disproportionate but few.
- If cloudy hours reached clear-day accuracy, MAPE would drop from 27.8% to
  18.4%. Most of that gap is irreducible with whole-sky okta cloud data.
- Excluding dawn and dusk (sun > 10°) lowered MAPE by only about 3 points;
  above 25° it stays above 20%.
- `day_wmape` was about 17.4% and `day_mdape` 16.5-18.2%; both metrics are now
  retired.

### Error drivers (`error_driver_analysis_run.py`)

- Cloud is a real but small driver: partial ρ +0.125, controlling for
  production. It is *hidden* in the raw correlation (ρ 0.003, a suppression
  effect).
- Data coverage is **not** a driver: partial ρ 0.024, p = 0.19. Outage hours
  show *lower* error, and the worst errors occur at higher coverage.
- `observation_fraction` ≡ `mean_reporting_systems` (ρ = 1, ratio 175).
- Radiation had no residual association with error. The earlier claim that
  forecast-hour cloud barely helps (0.125 vs 0.124) came from **misaligned
  data** and needs rechecking.

### Feature significance (LSTM, pre-fix)

Increase in MAE (kWh) when each group is removed: hour_of_day 0.398 (CI
0.307-0.487), wind speed 0.209, cloud 0.150, humidity 0.136, radiation 0.128,
temperature 0.114. Radiation ranking this low is plausibly an artefact of the
alignment bug; expect it to rise.

### Alignment bug (fixed)

PV hours are labelled by interval **start** (`resample`), KNMI by interval
**end**. The correlation between PV and radiation peaked at a +1 h lag (0.932)
instead of 0 (0.881). This leaked nothing — the weather was older than
available — but all models got radiation one hour stale. Fixed by shifting
KNMI back one step in `script.py` (`to_interval_start`); `check_alignment`
fails the build if the peak isn't at lag 0.

### 10-minute KNMI data

- Collection `10-minute-in-situ-meteorological-observations`; covers 2012
  onward; De Bilt present. The EDR API with a personal key (keys don't expire);
  limits 200 requests/s and 1000 requests/hour.
- Parameter codes differ from the hourly ones: `ta rh pp ff dd ss qg qgn qgx
  nc`. Cloud is `nc` (matches hourly 93.5%, versus `n` at 67.7%); `n` dropped.
  Radiation is already in W/m². Sunshine was converted from minutes to hours.
- 10-min labels mark interval **ends** (radiation check: 0.78 W/m² error
  versus 16.36 for the alternative).
- Agreement with the validated hourly product, after rounding: temperature
  100%, radiation 99.7%, wind speed 99.7%, wind direction 100% (excluding
  codes), pressure 97.5%, sunshine 95.2%, **cloud 93.5%, humidity 64.8%**. The
  differences shrink by year (≈1% in 2017), consistent with validation.
- Station gaps: 136 stamps in 48 gaps; the longest is 7h50 on 2017-07-24 (test
  period).
- SARIMAX at `s=144`: estimated about 37 h per test run; the probe gives the
  real figure.

### Other conclusions

- Explicit time encoding earns its place (+4.7% validation, 3-4σ).
- Weather helps only nonlinearly: ARIMAX/SARIMAX don't beat ARIMA/SARIMA.
- 15-min data (discussed): only worth it with finer *weather*, which led to the
  10-min work.

---

## Open items

**Advisor's list**
- [done] day/night periods and metrics, scaled-MAE diagrams, MAPE diagnosis,
  no smoothing, hour_of_day label.
- [**not started**] extend parameter ranges and rerun feature significance.
- [**not started**] use the cloud-cover analysis to decide which SARIMAX
  parameters to change.

**Before quoting anything in the thesis**
- Rerun all experiments on both datasets.
- Recheck the cloud lag test and radiation's feature importance on aligned data.
- Decide which hourly dataset carries the main thesis results (currently
  `hourly_from_10min`).

**Thesis text corrections identified**
- Tie-break: "stability and simplicity" → the rule prefers the *longer* window;
  drop "simplicity".
- Delete the wMAPE/MdAPE sentence: those metrics were removed from the code.
- "Daylight-specific evaluation" → clock-hour day (06-17) / night (18-05).
- NMAE "consistent across partitions": the numeric NMAE exists only for test;
  validation uses the same divisor only in figures.
- Causal imputation paragraph: now exactly true, since `bfill` was removed.

**Stale documents**
- [done] `METHODOLOGY_NOTES.md` updated to the clock-hour periods, current
  metric set, datasets, alignment fix and actual tie-break rule.
- The MAPE email draft to the advisor used pre-fix numbers.
