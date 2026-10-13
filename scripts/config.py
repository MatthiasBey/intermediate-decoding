"""Central path and date configuration for all scripts in this repository.

Every script searches upwards for this file, puts its folder (scripts/) on
``sys.path`` and imports it, so the helper package ``utils`` can be imported as well.

Repository layout
-----------------
data/                          inputs of the example forecast (see data/README.md)
    era5/                      ERA5 daily files and constants
    intermediate_decodings/    tuned/<model>/, no_skip/<model>/
    latent_states/<model>/     latent states (all layers)
    translators/<model>/       weights and training logs (Aurora, ArchesWeather: skip/, no_skip/)
    model_loss/<model>/        direct/, tuned/ loss files of all 48 forecasts
results/paper/                 precomputed diagnostics (see results/README.md): per-date
                               diagnostics of all 48 forecasts, covariance_matrices/ and the
                               increments of the example forecast
results/single_run/            diagnostics of the example forecast computed from data/
figures/paper/                 figures of run_paper_figures.sh (supplement in supplement/)
figures/single_run/            figures of run_single_run_figures.sh
figures/published/             the published figures, for comparison

Pipelines and modes (environment variable ID_MODE)
--------------------------------------------------
run_paper_figures.sh (ID_MODE=paper): only plots. DATES holds the 48 forecasts of the
    paper (valid on the 10th of each month in 2020 at 00, 06, 12 and 18 UTC); figures are
    made from results/paper/ or, for the example forecast, from data/.
run_single_run_calculations.sh (ID_MODE=single_run, the default): computes all
    diagnostics of the example forecast (valid 2020-03-10 12 UTC) from data/ and saves
    them to results/single_run/.
run_single_run_figures.sh (ID_MODE=single_run): plots the results in results/single_run/.
"""
import datetime
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)

MODE = os.environ.get("ID_MODE", "single_run")
if MODE not in ("paper", "single_run"):
    raise ValueError(f"ID_MODE must be 'paper' or 'single_run', not {MODE!r}")
PAPER_MODE = MODE == "paper"

DATA_DIR = os.environ.get("ID_DATA_DIR", os.path.join(REPO_ROOT, "data"))
RESULTS_DIR = os.environ.get("ID_RESULTS_DIR", os.path.join(REPO_ROOT, "results", MODE))
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", MODE)
SUPPLEMENT_DIR = os.path.join(FIGURES_DIR, "supplement")

# --- inputs (DATA_DIR) ---
ERA5_DIR = os.path.join(DATA_DIR, "era5")
DECODINGS_DIR = os.path.join(DATA_DIR, "intermediate_decodings")
LATENT_DIR = os.path.join(DATA_DIR, "latent_states")
TRANSLATOR_DIR = os.path.join(DATA_DIR, "translators")
MODEL_LOSS_DIR = os.path.join(DATA_DIR, "model_loss")

# --- precomputed, used in both modes ---
COVARIANCE_DIR = os.path.join(REPO_ROOT, "results", "paper", "covariance_matrices")

# --- temporary files ---
TMP_DIR = os.path.join(RESULTS_DIR, "tmp")

# --- dates ---
SINGLE_DATE = datetime.datetime(2020, 3, 10, 12, 0)                    # valid time of the example forecast
DATES_48 = [datetime.datetime(2020, month, 10, hour, 0)
            for month in range(1, 13) for hour in (0, 6, 12, 18)]
DATES = DATES_48 if PAPER_MODE else [SINGLE_DATE]
DATES_TAG = "mean_48_dates" if PAPER_MODE else SINGLE_DATE.strftime("%Y-%m-%d-%H-%M")  # figure-name suffix


def analysis_date():
    """Forecast valid time for scripts that show a single forecast.

    Taken from the first command line argument ("%Y-%m-%d-%H-%M"); without
    argument, the example forecast SINGLE_DATE is used.
    """
    if len(sys.argv) > 1:
        return datetime.datetime.strptime(sys.argv[1], "%Y-%m-%d-%H-%M")
    return SINGLE_DATE


def aurora_init_time(valid_time):
    """Aurora files are named after the initial time, 6 h before the valid time."""
    return valid_time - datetime.timedelta(hours=6)


for _d in (RESULTS_DIR, SUPPLEMENT_DIR):
    os.makedirs(_d, exist_ok=True)
