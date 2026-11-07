# Methodology notes

Reasoning behind the evaluation decisions in this project. Design and
justification only — no result figures.

## Scope

One-step-ahead forecasting of aggregate PV production for a site in Utrecht.
Temporal split 65% train / 15% validation / 20% test, with no shuffling. Two
model families are compared: autoregressive (ARIMA, SARIMA, ARIMAX, SARIMAX)
and recurrent (LSTM, GRU), against persistence and seasonal-naive-24h baselines.

Two datasets are evaluated, both built from KNMI 10-minute observations at
De Bilt:

- **`hourly_from_10min`** (default): the 10-minute data aggregated to hours
  following KNMI's own hourly definitions: PV and sunshine are summed,
  radiation is averaged, and snapshot variables take the last value of the
  hour. One step is one hour ahead.
- **`10min`**: native 10-minute resolution. One step is ten minutes ahead,
  which is an easier task, so its errors are not comparable with the hourly
  ones.

Split boundaries fall on whole hours, so both datasets split at the same
timestamps. The KNMI validated hourly product is kept as a third dataset but is
not used.

## Evaluation design

### No smoothing

No moving average, spline, interpolation or filter is applied to any predicted
or observed value. Plot lines connect measured points directly. Two things in
the pipeline can be mistaken for smoothing and are not:

- **Monthly block metrics** partition the hours; they do not slide a window over
  them. Each hour belongs to exactly one month, the error is computed inside
  each month independently, and those per-month values are then averaged. No
  error is blended into its neighbours.
- **Forward-fill imputation** fills gaps in the raw series. It replaces missing
  entries and never modifies an observed one.

The only transformation applied to the target is clipping at zero, which removes
physically impossible negative production.

### Why metrics are aggregated per month

Solar output swings enormously by season. A single pooled figure over the whole
evaluation period is dominated by the summer months simply because there are
more kWh there to be wrong about, so a model that performs well in July and
badly in February can still post a good pooled score. Computing the error inside
each calendar month and averaging those values gives each month equal weight.

This departs deliberately from a pooled metric, and the two are not
identical: months differ in length and the partial months at each end of a split
carry the same weight as full ones. The per-month spread is kept as a standard
deviation and reported. It shows whether a model is consistently decent or
merely seasonally lucky, but it does not enter the selection rule.

### Timestamp alignment

PV intervals are labelled by their start and KNMI observations by their end.
Left as-is, every weather reading would be one step older than it needs to be.
KNMI timestamps are therefore shifted back one step to interval-start labels,
and the build fails unless the PV–radiation cross-correlation peaks at lag 0.
The misalignment leaked no future information, since the weather was older than
it needed to be, not newer, but it cost the models the latest radiation reading.

Exogenous inputs are lagged by one step, so a forecast only uses the latest
*completed* interval.

### Causal imputation

Missing values are forward-filled only; there is no backward fill and no
interpolation. Forward-fill only carries past values across a gap, so imputation
can never reach backward over a train/test boundary and leak future information
into an earlier split.

At 10 minutes, gaps in the station record are first filled from the latest
completed hourly KNMI value available at forecast time, which is also causal.
The few gaps that remain are forward-filled like any other.

### Rolling-origin evaluation

Forecasts are produced one step ahead in a rolling fashion with a bounded
history window, rather than fitting once and predicting the whole horizon. This
matches how the model would actually be used.

### Protocol version tags

Each stage writes a protocol string into its results, and the dataset name is
part of every tag. Downstream stages check for the columns and protocol they
expect and refuse to proceed against outputs produced under an older
methodology or a different dataset. This guard prevents silently mixing results
from different stages of the pipeline, which is a real risk when experiments are
expensive and get rerun at different times.

## Feature encoding

The recurrent feature set is five weather measurements (radiation, cloud cover,
temperature, wind speed, humidity) plus a cyclical encoding of the time of day.
The time of day is the fractional hour, represented as a sine and cosine pair
rather than an integer, for two reasons:

- Either function alone is ambiguous: two different times of day share the same
  sine value. The pair pins the time to a unique point on a circle.
- A raw integer hour would place hour 23 and hour 0 twenty-three units apart
  despite being adjacent in time, forcing the model to spend capacity undoing an
  artificial discontinuity at midnight.

In feature-importance experiments the two columns are always removed together,
since dropping only one would leave a broken encoding and a meaningless result.
The group is called `hour_of_day`, and figures list its columns (`hour_sin`,
`hour_cos`) so it is not mistaken for a single feature.

### Exogenous inputs of the autoregressive models

ARIMAX and SARIMAX receive measured weather lagged by one interval, because a
measurement is only available once its interval has ended. Once the previous
interval's production is known, that lagged weather adds little: on the hourly
training data its partial correlation with production is about 0.07 for
radiation and 0.01 for cloud cover. This is why weather inputs have not improved
the linear models.

Two further inputs are therefore offered to the tuning grid, each in its own
input sets:

- **Humidity** (lagged), the one weather variable that still adds some
  information given the previous production (partial correlation about 0.2).
- **Solar geometry**, `solar_elevation_sin`: the sine of the sun's elevation at
  the middle of the interval being forecast, floored at zero. It is proportional
  to clear-sky irradiance on a horizontal surface. It is computed from the
  timestamp and the site coordinates alone, so it is known before the interval
  starts and enters **unlagged** without leaking anything.

Two more input sets follow from the feature significance study, where removing
cloud cover improved the selected SARIMAX:

- **Radiation alone**, so the grid contains the set that study pointed to.
- **Daylight cloud cover**, `cloud_cover_daylight`: the previous interval's cloud
  cover times the `solar_elevation_sin` of the interval being forecast. A linear
  model gives plain cloud cover one coefficient for every hour, and at night,
  where production is zero but cloud cover still varies, that term moved the
  forecasts away from zero. On validation, all of the extra error from cloud
  cover came from night hours (night MAE 0.76 against 0.36 kWh without it; day
  MAE 10.88 against 10.90 kWh). Weighted by the sun, cloud cover is zero at
  night. It is built from the already lagged cloud cover and is not lagged again.

These inputs are offered to the autoregressive models only. The recurrent
models keep their feature set, so when the two families are compared the
difference in inputs must be stated.

## Reporting periods

### Defined by clock hour

Reported metrics split each day by clock hour:

| period | hours |
|---|---|
| day | 06–17 |
| night | 18–05 (the complement of day) |
| upwards/downwards | 06–09 and 15–17 |
| by day | 10–14 |

Clock hours need no site geometry to interpret, so a reported figure can be
reproduced from the timestamp alone. The cost is that clock periods do not
follow the sun: at this latitude, day length varies by more than eight hours
between December and June, and in summer the "night" hours 04, 05 and 18 carry
real production. Night error is therefore not close to zero, and it should not
be described as a dark-hours figure.

Solar elevation (NOAA solar-position equations at the site coordinates) is used
in the diagnostic scripts in `src/analysis/`, where the question is physical and
a mask that follows the sun is the right tool, and as an input the
autoregressive models can select (see "Exogenous inputs of the autoregressive
models"). It never defines a reported metric period.

### Which metric is reported where

| period | reported (`final_comparison/tables/reported_metrics.csv`) |
|---|---|
| whole | NMAE (`nmae`) |
| day / night | NMAE (`day_nmae`, `night_nmae`) |
| upwards/downwards | MAPE (`upwards_downwards_mape`) |
| by day | MAPE (`by_day_mape`) |

In the reported tables (`reported_metrics.csv`, `high_quality_metrics.csv`,
`block_metrics.csv`) each column name carries its unit, e.g.
`day_nmae [% of peak production]`, `by_day_mape [%]`, `test_block_mae_mean [kWh]`.
NMAE is the MAE divided by the peak production in the training period (313.1 kWh
per hour hourly, 52.8 kWh per 10 minutes), in %, so 3.49 means 0.0349 as a
fraction. `comparison.csv` keeps the plain names because
other scripts read it.

MAPE is never reported for night or for the whole period, since a denominator
near zero makes it meaningless there. Unscaled MAE, RMSE and the NMAE divisor
remain available in `final_comparison/tables/comparison.csv`.

### Why MAE is split into day and night

Every model scores well at night because predicting little or no production
after sunset is easy. Pooling those hours lowers the reported MAE without
reflecting any forecasting skill. Reporting day and night separately makes the
figure honest: the day number is not "worse", it is the undiluted one, and the
pooled number was flattered by the easy part of the dataset.

The error tracks the production curve through the day. It is largest in absolute
terms at midday, simply because that is when there is the most electricity to be
wrong about.

## The MAPE problem

### Why the percentage error is inflated

MAPE divides each hour's absolute error by that hour's own production. When
production is small, a modest absolute miss becomes an enormous percentage, and
the mean is dragged upward by a thin tail of such hours. The result is a
percentage that looks alarming while the actual and predicted curves track each
other closely.

MAPE only scores intervals above a production threshold of 5 kWh per hour
(the same rate per interval at 10 minutes). Two clarifications matter, because
both are easy to get wrong:

- **It is not a night-time artifact.** The production threshold already removes
  night by itself; restricting MAPE further to sunlit hours changes little.
- **The primary cause is the upwards and downwards hours (06–09, 15–17).**
  Production is small while it rises in the morning and falls in the afternoon,
  so these hours carry the largest share of the percentage error.
- **The secondary cause is cloudy by-day hours (10–14).** Within the
  high-production plateau, overcast hours have low production and therefore a
  much higher percentage error than clear hours.

So the inflation comes from the arithmetic of small denominators, not from a
model failure, and the hours causing it are daylight hours that no daylight mask
can remove.

### The response

MAPE is kept, because published work reports it, but only over the
upwards/downwards and by-day periods and reported for each separately. The split separates the
low-production hours, where the percentage is inflated, from the high-production
plateau, where it means what it appears to mean.

The inflation itself is explained by three diagnostic scripts in `src/analysis/`
rather than by extra reported metrics. They break down the error by production
level, by clock hour and by sky condition (clear, cloudy, dawn, dusk, technical
failure). The earlier alternatives, wMAPE and MdAPE, were removed from the
reported set.

## Normalised MAE

An absolute error in kWh is meaningless without knowing the size of the
installation, so every reported MAE also comes as NMAE: min-max-scaled MAE,
expressed as a percentage.

    NMAE = MAE / (training max − training min) × 100

The training minimum is exactly zero, because roughly half of all hours are
dark, so this equals MAE divided by the training peak. The code keeps the
general form so that the reported columns show why the two coincide.

Two choices in the implementation are deliberate:

- **The divisor comes from the training split alone**, so no test information
  enters the definition of a reported metric.
- **A single constant is applied to every split.** Dividing each split by its own
  peak would be actively misleading here: the validation period is winter-heavy
  and peaks far below training, so its normalised error would be inflated and no
  longer comparable with test.

Each dataset has its own divisor: the peak hourly energy for `hourly_from_10min`
and the peak 10-minute energy for `10min`.

Because the divisor is one constant, normalising does not change the shape of any
curve or the selected hyperparameter — only the axis units change. That is the
intent; the scaled figures exist for comparability, not to alter conclusions.
Numeric NMAE values are reported for test only; validation uses the same divisor
only in figures.

### Caveat on capacity

The aggregate target sums the systems reporting at each timestamp, and that fleet
shrinks substantially over the record. There is therefore no true fixed installed
capacity, and the training peak is a stable, reproducible reference level rather
than a nameplate rating. Dividing by the smaller test-era peak would raise NMAE
noticeably. The divisor should be stated wherever NMAE is quoted, and the
difference should be noted when comparing against published NMAE values that
divide by a real rated capacity.

## Data quality position

The raw source contains acquisition outages and changing system coverage.
Missing minutes count as zero production, so an outage looks like a dim hour.
The target is not a perfectly observed fixed fleet, and it should not be
described as one. Redefining or imputing the target would be a separate
methodological decision requiring a complete rerun. Instead, the primary metrics
keep the established target definition, and a parallel sensitivity analysis
reports the same comparison on high-quality hours only (observation coverage of
at least 50%).

The changing fleet size also confounds the level of the target over time, which
matters when interpreting any trend in absolute production.

## Fairness of the comparison

- Both families are selected on validation data only; the test period is touched
  once, at the end.
- Recurrent models are run under several fixed seeds and reported as the mean
  across them, so a favourable initialisation cannot flatter a result. Seed
  spread is reported alongside.
- Scalers are fitted on the training split only.
- For the test, recurrent models are retrained on train+validation for the
  epoch count early stopping chose during tuning. Nothing checks that
  retraining, so the learning rate decays by a fixed factor per epoch in every
  training run. Without it, the 10-minute GRU seeds ended their retraining with
  different output scales, up to about ±3 kWh at high production. The schedule
  depends only on the epoch number, so a retraining follows exactly the path
  tuning took.
- Autoregressive fits that do not converge are rejected rather than silently
  accepted. Tuning allows 200 optimiser iterations. The final refit of a selected
  setting allows 1000, because it fits a new window and must still reach an
  optimum. Settings rejected during tuning stay rejected, and a fit that converges
  within 200 iterations gives the same result under either limit.
- The selection rule does not take the raw minimum, which is sensitive to
  noise. Every candidate within 2% of the best monthly-block validation MAE is
  eligible. Among those, the rule prefers lower BIC (autoregressive models,
  within one window length), then lower monthly-block RMSE, then the **longer**
  history window. It favours stability, not simplicity: a longer window is not
  a simpler model.
- Baselines are included so that reported skill is anchored to something
  interpretable rather than only to other trained models.
- The search ranges were extended where the best setting of the original grid
  sat on an edge. For the autoregressive models the extended grid is crossed in
  full. For the recurrent models, where a full cross would multiply the grid
  about twentyfold, each window length's best grid configuration is varied one
  parameter at a time through the new values, and selection runs over grid and
  variations together.

## Open questions

- Which hourly dataset carries the main thesis results. Currently
  `hourly_from_10min`.
- Whether the capacity reference should be revisited given the fleet drift, for
  instance by normalising against a rolling estimate of active capacity rather
  than a fixed peak. This would change the metric's interpretation and is not a
  purely cosmetic decision.
