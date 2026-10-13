#!/bin/bash
# Pipeline 2: recomputes all diagnostics for the example run (forecast valid 2020-03-10 12 UTC) from
# data/ (ID_MODE=single_run) and saves them to results/single_run/. Makes no figures; plot the results
# with run_single_run_figures.sh. Not included: the covariance matrices, which need a full year of
# latent states (see scripts/methods_without_data/).
#
# Usage (inside the activated environment, see install_env.sh):
#   bash run_single_run_calculations.sh                            # everything
#   bash run_single_run_calculations.sh spectral_filtering spectra # selected steps only (see STEPS below)
# PYTHON selects the interpreter (default: python3 of the active environment).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/scripts"
export ID_MODE=single_run

DATE=2020-03-10-12-00
PY=${PYTHON:-python3}
command -v cdo >/dev/null || echo "WARNING: cdo not found on PATH (needed for spectral filtering)"

run () { echo; echo ">>> $PY $*"; "$PY" -u "$@"; }

# ---------------- calculations (scripts/calculations/), in dependency order ----------------
step_increments () {
    for model in graphcast archesweather aurora; do
        run calculations/1_increments/${model}_increments.py "$DATE"
        run calculations/1_increments/${model}_input_difference.py "$DATE"
    done
}

step_relative_mse () {
    run calculations/2_relative_mse/compute_relative_mse.py
}

step_encoder_decoder_similarity () {
    run calculations/3_encoder_decoder_similarity/compute_encoder_decoder_similarity.py 1
}

# shared preprocessing for the spectra and balances
step_spectral_filtering () {
    for model in arches aurora graphcast; do
        run calculations/4_spectral_filtering/1_filter_intermediate_decodings.py "$DATE" "$model"
    done
    run calculations/4_spectral_filtering/2_filter_era5.py "$DATE"
    run calculations/4_spectral_filtering/3_spectral_power_of_states_and_input.py "$DATE"
}

step_spectra () {   # needs spectral_filtering
    run calculations/5_increment_spectra/compute_increment_spectra_geopot_u10m.py "$DATE"
    run calculations/5_increment_spectra/compute_increment_spectra_all_variables.py "$DATE"
}

step_balances () {  # needs spectral_filtering
    run calculations/6_physical_balances/1_create_spectrally_scaled_era5.py "$DATE"
    run calculations/6_physical_balances/2_compute_balances_full_forecast.py "$DATE"
    run calculations/6_physical_balances/3_summarize_balances_full_forecast.py "$DATE"
    run calculations/6_physical_balances/compute_balances_intermediate_decodings.py "$DATE"
    run calculations/6_physical_balances/compute_balances_era5_scale_separated.py "$DATE"
}

STEPS=${*:-increments relative_mse encoder_decoder_similarity spectral_filtering spectra balances}
for s in $STEPS; do "step_$s"; done
