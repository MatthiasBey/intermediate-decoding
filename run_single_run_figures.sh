#!/bin/bash
# Pipeline 3: makes the figures and tables of the paper from the diagnostics that
# run_single_run_calculations.sh saved to results/single_run/ (ID_MODE=single_run), so they show the
# example run (forecast valid 2020-03-10 12 UTC) only: the 48-date means become values of this date.
# Figures of the example run that the paper makes directly from data/ are made the same way here.
# Figures -> figures/single_run/ (supplement: figures/single_run/supplement/), tables ->
# results/single_run/. Not included: the covariance figure, whose matrices need a full year of
# latent states (see run_paper_figures.sh).
#
# Usage (inside the activated environment, see install_env.sh):
#   bash run_single_run_figures.sh                         # everything
#   bash run_single_run_figures.sh plots_main tables       # selected steps only (see STEPS below)
# PYTHON selects the interpreter (default: python3 of the active environment).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/scripts"
export ID_MODE=single_run

DATE=2020-03-10-12-00
PY=${PYTHON:-python3}

run () { echo; echo ">>> $PY $*"; "$PY" -u "$@"; }

# ---------------- plots (scripts/plots/), one script per figure or table ----------------
step_plots_main () {
    run plots/main/latent_state_development.py "$DATE"
    run plots/main/relative_mse_all_variables.py
    run plots/main/encoder_decoder.py "$DATE"
    run plots/main/graphcast_increments_spatial_maps.py "$DATE"
    run plots/main/increment_spectra_geopot_u_10m.py
    run plots/main/physical_balances.py
    run plots/main/physical_balances_500hPa.py
}

step_plots_supplement () {
    run plots/supplement/increments_all_models.py "$DATE"
    run plots/supplement/increment_spectra_all_variables.py
    run plots/supplement/translator_training_loss.py
    run plots/supplement/no_skip_translator_training_loss_aurora.py
    run plots/supplement/no_skip_translator_training_loss_arches.py
    run plots/supplement/tuned_decoding_comparison_aurora.py "$DATE"
    run plots/supplement/tuned_decoding_comparison_arches.py "$DATE"
    run plots/supplement/latent_channel_aurora.py "$DATE"
    run plots/supplement/latent_channel_arches.py "$DATE"
}

step_tables () {
    run plots/tables/encoder_decoder_pattern_correlation.py
    run plots/tables/translator_convergence.py
}

STEPS=${*:-plots_main plots_supplement tables}
for s in $STEPS; do "step_$s"; done
