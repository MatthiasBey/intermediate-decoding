#!/bin/bash
# Pipeline 1: creates all figures and tables of the paper and supplement from the published data
# (ID_MODE=paper). Nothing is recomputed:
#   - the 48-date figures and tables and the covariance figure read the precomputed diagnostics in
#     results/paper/,
#   - the increment maps of the example run (forecast valid 2020-03-10 12 UTC) read the precomputed
#     increments in results/paper/increments/,
#   - the other figures of the example run read data/ directly.
# Figures go to figures/paper/ (supplement: figures/paper/supplement/), tables to results/paper/.
#
# Usage (inside the activated environment, see install_env.sh):
#   bash run_paper_figures.sh                         # everything
#   bash run_paper_figures.sh plots_main tables       # selected steps only (see STEPS below)
# PYTHON selects the interpreter (default: python3 of the active environment).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/scripts"
export ID_MODE=paper

DATE=2020-03-10-12-00
PY=${PYTHON:-python3}

run () { echo; echo ">>> $PY $*"; "$PY" -u "$@"; }

# ---------------- plots (scripts/plots/), one script per figure or table ----------------
step_plots_main () {
    run plots/main/latent_state_development.py "$DATE"
    run plots/main/covariance_similarity.py
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
