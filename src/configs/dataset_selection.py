"""Which dataset the experiments run on.

The datasets share one PV record and differ in their weather source or their
resolution:

    hourly_from_10min  KNMI 10-minute observations aggregated to hours (default)
    10min              KNMI 10-minute observations
    hourly_validated   KNMI's validated hourly product, kept but no longer used

Select one with --dataset NAME on an experiment's command line, or with the
SOLAR_DATASET environment variable. The command line wins.

The flag is read from sys.argv when this module is first imported. Every
experiment binds its data path, result folders and step-dependent settings as
module-level constants, which happens before its own argparse runs, so reading
the flag here is what lets an ordinary per-script option reach those constants.
Scripts still register the option through add_dataset_argument, so --help lists
it and argparse accepts it.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "utrecht" / "processed_data"
EXPERIMENTS_DIR = PROJECT_ROOT / "src" / "experiments"

ENVIRONMENT_VARIABLE = "SOLAR_DATASET"
DEFAULT_DATASET = "hourly_from_10min"
HOUR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class Dataset:
    name: str
    path: Path
    step: pd.Timedelta
    freq: str
    results_root: Path
    # Datasets built from the 10-minute record carry their own PV coverage
    # column. The validated hourly dataset predates it and relies on the
    # separate PV quality audit instead.
    coverage_in_file: bool
    description: str


DATASETS = {
    "hourly_from_10min": Dataset(
        name="hourly_from_10min",
        path=PROCESSED_DIR / "utrecht_pv_data_hourly_from_10min.csv",
        step=HOUR,
        freq="h",
        results_root=EXPERIMENTS_DIR / "results",
        coverage_in_file=True,
        description="KNMI 10-minute observations aggregated to hours"
    ),
    "10min": Dataset(
        name="10min",
        path=PROCESSED_DIR / "utrecht_pv_data_10min.csv",
        step=pd.Timedelta(minutes=10),
        freq="10min",
        results_root=EXPERIMENTS_DIR / "results_10min",
        coverage_in_file=True,
        description="KNMI 10-minute observations"
    ),
    "hourly_validated": Dataset(
        name="hourly_validated",
        path=PROCESSED_DIR / "utrecht_pv_data.csv",
        step=HOUR,
        freq="h",
        results_root=EXPERIMENTS_DIR / "results_hourly_validated",
        coverage_in_file=False,
        description="KNMI validated hourly product (kept, not in use)"
    ),
}


def requested_dataset_name(argv=None, environ=None):
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ

    for index, token in enumerate(argv):
        if token == "--dataset" and index + 1 < len(argv):
            return argv[index + 1]

        if token.startswith("--dataset="):
            return token.split("=", 1)[1]

    return environ.get(ENVIRONMENT_VARIABLE, DEFAULT_DATASET)


def resolve_dataset(name):
    if name not in DATASETS:
        raise ValueError(
            f"unknown dataset {name!r}; choose one of {sorted(DATASETS)}"
        )

    return DATASETS[name]


ACTIVE_DATASET = resolve_dataset(requested_dataset_name())
STEP = ACTIVE_DATASET.step
STEPS_PER_HOUR = int(HOUR / STEP)


def hours_to_steps(hours):
    """Convert a duration in hours into rows of the active dataset."""
    steps = hours * STEPS_PER_HOUR

    if steps != int(steps):
        raise ValueError(
            f"{hours} h is not a whole number of {STEP} steps"
        )

    return int(steps)


def add_dataset_argument(parser):
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS),
        default=ACTIVE_DATASET.name,
        help=(
            "Dataset to run on (default: "
            f"{DEFAULT_DATASET}, or ${ENVIRONMENT_VARIABLE}). Results go to "
            "that dataset's own folder."
        )
    )
    return parser


def dataset_arguments():
    """Command-line arguments that select the active dataset in a subprocess.

    A child interpreter re-imports this module from its own argv, so anything
    launching one has to pass the selection on explicitly.
    """
    return ["--dataset", ACTIVE_DATASET.name]
