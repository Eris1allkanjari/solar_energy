"""Reference capacity for expressing forecast errors as a percentage of peak."""

from src.configs.evaluation import TRAIN_RATIO


def training_peak_capacity(df, target_column="pv_total_kWh"):
    """Peak production observed in the training split, in kWh.

    Normalising by peak output is how PV forecast errors are usually made
    dimensionless and comparable across sites. Two choices are deliberate here.

    The peak is taken from the training split alone, so no test information
    enters the definition of the metric. And the same single constant is applied
    to every split, rather than each split being divided by its own peak: the
    validation period is winter-heavy and peaks far lower than training, so a
    per-split divisor would inflate its scaled error and make validation and
    test figures impossible to compare.

    The fleet behind the aggregate target shrinks over the record, so this peak
    is a reference level for scaling rather than a true installed-capacity
    rating. It is a fixed, reproducible divisor, which is what the metric needs.
    """
    train_end = int(len(df) * TRAIN_RATIO)

    return float(
        df[target_column].iloc[:train_end].max()
    )
