# Methodology notes

Reasoning behind the evaluation decisions in this project. Design and
justification only — no result figures.

## Scope

One-hour-ahead (horizon = 1) forecasting of aggregate PV production for a site
in Utrecht. Hourly data, temporal split 65% train / 15% validation / 20% test.
Two model families are compared: autoregressive (ARIMA, SARIMA, ARIMAX, SARIMAX)
and recurrent (LSTM, GRU), against persistence and seasonal-naive-24h baselines.

## Evaluation design

### No smoothing

No moving average, spline, interpolation or filter is applied to any predicted
or observed value. Plot lines connect measured points directly. This is worth
stating explicitly because two things in the pipeline can be mistaken for
smoothing and are not:

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

This is a deliberate departure from a pooled metric, and the two are not
identical: months differ in length and the partial months at each end of a split
carry the same weight as full ones. The per-month spread is retained as a
standard deviation, which reveals whether a model is consistently decent or
merely seasonally lucky. The selection rule uses that spread when two candidates
score within tolerance of each other.

### Causal imputation

Missing values are forward-filled rather than interpolated. Forward-fill only
carries past values across a gap, so imputation can never reach backward over a
train/test boundary and leak future information into an earlier split.

### Rolling-origin evaluation

Forecasts are produced one step ahead in a rolling fashion with a bounded
history window, rather than fitting once and predicting the whole horizon. This
matches how the model would actually be used.

### Protocol version tags

Each stage writes a protocol string into its results. Downstream stages check
for the columns and protocol they expect and refuse to proceed against outputs
produced under an older methodology. This is the guard that prevents silently
mixing results from different eras of the pipeline, which is a real hazard when
individual experiments are expensive and get rerun at different times.

## Feature encoding

The recurrent feature set is five weather measurements plus a cyclical encoding
of the hour. The hour is represented as a sine and cosine pair rather than an
integer, for two reasons:

- Either function alone is ambiguous — two different hours of the day share the
  same sine value. The pair pins the hour to a unique point on a circle.
- A raw integer hour would place hour 23 and hour 0 twenty-three units apart
  despite being adjacent in time, forcing the model to spend capacity undoing an
  artificial discontinuity at midnight.

In feature-importance experiments the two columns are always removed together,
since dropping one would leave a half-crippled encoding whose result means
nothing. Figures label such groups with their constituent columns so a group
name is not mistaken for a single feature.

## The day/night distinction

### Defining the day period

Daylight is determined by solar geometry rather than a fixed clock window: the
NOAA solar-position equations give the sun's elevation at the site coordinates
for each timestamp, and an hour counts as daylight when the sun is above the
horizon. A fixed clock window would be wrong at this latitude, where day length
varies by more than eight hours between December and June.

### Why MAE is reported twice

Roughly half of all hours are dark, and every model scores near zero on them
because predicting no production after sunset is trivial. Pooling those hours
therefore roughly halves the reported MAE without reflecting any forecasting
skill. Reporting the daylight and night periods separately makes the figure
honest: the daylight number is not "worse", it is the undiluted one, and the
pooled number was flattered by the easy half of the dataset.

Reported in context, the daylight error is a modest fraction of mean daylight
production, and the error tracks the production curve through the day — largest
in absolute terms at midday, simply because that is when there is the most
electricity to be wrong about. The proportional error stays roughly constant
across the productive hours.

## The MAPE problem

### Why the percentage error is inflated

MAPE divides each hour's absolute error by that hour's own production. When
production is small, a modest absolute miss becomes an enormous percentage, and
the mean is dragged upward by a thin tail of such hours. The result is a
percentage figure that looks alarming while the actual and predicted curves
track each other closely.

Two clarifications matter here, because both are easy to get wrong:

- **It is not a night-time artifact.** MAPE already excludes hours below a
  production threshold, and that threshold removes night by itself. Restricting
  MAPE to an explicit daylight mask changes almost nothing. It is retained only
  as a cross-check confirming the threshold is a sound proxy for daylight.
- **It is not confined to dawn and dusk.** Low production occurs throughout the
  day whenever cloud cover is heavy, and because there are many more midday
  hours than twilight hours, overcast middays contribute more to the inflation
  than the sunrise and sunset ramps do.

So the inflation is arithmetic, not a model failure, and the hours causing it are
daylight hours that no daylight mask can remove.

### The response

Rather than discard the percentage metric, two alternatives are computed over
the same period:

- **wMAPE** divides summed error by summed production. The denominator is a
  period total that no single small hour can distort, so it needs no production
  threshold at all, and it answers a question worth asking: what fraction of the
  energy generated did the forecast get wrong.
- **MdAPE** keeps the same mask and threshold as MAPE and swaps the mean for a
  median, reporting the typical hour instead of one dominated by the tail.

The two approach the problem from different directions — a ratio of totals and a
median of ratios. When they agree, that is the evidence that the original figure
was an artifact of small denominators rather than a genuine accuracy level.

## Normalised MAE

An absolute error in kWh is meaningless without knowing the size of the
installation, so every MAE is also reported as NMAE: the error divided by a
reference capacity and expressed as a percentage. This is the convention that
allows comparison against published results from other sites.

Two choices in the implementation are deliberate:

- **The divisor comes from the training split alone**, so no test information
  enters the definition of a reported metric.
- **A single constant is applied to every split.** Dividing each split by its own
  peak would be actively misleading here: the validation period is winter-heavy
  and peaks far below training, so its normalised error would be inflated and no
  longer comparable with test.

Because the divisor is one constant, normalising does not change the shape of any
curve or the selected hyperparameter — only the axis units change. That is the
intent; the scaled figures exist for comparability, not to alter conclusions.

### Caveat on capacity

The aggregate target sums the systems reporting at each timestamp, and that fleet
shrinks substantially over the record. There is therefore no true fixed installed
capacity, and the observed training peak is a stable, reproducible reference
level rather than a nameplate rating. This distinction should be stated when
comparing against published NMAE values that divide by a real rated capacity.

## Data quality position

The raw source contains acquisition outages and changing system coverage. The
target is not a perfectly observed fixed fleet, and it should not be described as
one. Rather than redefine or impute the target, which would be a separate
methodological decision requiring a complete rerun, the primary metrics preserve
the established target definition and a parallel sensitivity analysis reports the
same comparison restricted to hours with adequate raw observation coverage.

The changing fleet size also confounds the level of the target over time, which
matters when interpreting any trend in absolute production.

## Fairness of the comparison

- Both families are selected on validation data only; the test period is touched
  once, at the end.
- Recurrent models are run under several fixed seeds and reported as the mean
  across them, so a favourable initialisation cannot flatter a result. Seed
  spread is reported alongside.
- Non-converged or fallback-assisted autoregressive fits are rejected rather than
  silently accepted.
- The selection rule prefers a simpler or more stable candidate when accuracy is
  within a small tolerance, rather than taking the raw minimum, which is noise-
  sensitive.
- Baselines are included so that reported skill is anchored to something
  interpretable rather than only to other trained models.

## Open questions

- Whether to report MdAPE alongside mean MAPE in the thesis, or replace the mean
  entirely. Replacing it loses comparability with published work that reports
  mean MAPE; keeping both invites the question of which one is "the" number.
- Whether the capacity reference should be revisited given the fleet drift, for
  instance by normalising against a rolling estimate of active capacity rather
  than a fixed peak. This would change the metric's interpretation and is not a
  purely cosmetic decision.
