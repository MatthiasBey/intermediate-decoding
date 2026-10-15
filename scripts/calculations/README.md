# calculations

These scripts compute the diagnostics of a single forecast and save them to `results/<mode>/`.
`run_single_run_calculations.sh` runs all of them for the example forecast, valid 2020-03-10 12 UTC.
The files in `results/paper/` were made with the same scripts for all 48 forecasts of the paper.

The folders are numbered in the order in which they have to run. Within a folder, scripts with a
number prefix have to run in that order. Most scripts take the forecast valid time as their only
argument, for example:

```bash
python 1_increments/graphcast_increments.py 2020-03-10-12-00
```

## 1_increments

Increments between consecutive intermediate decodings (Δx̂ᵢ = x̂ᵢ − x̂ᵢ₋₁), the increment of the
encoder–decoder alone (x̂₀ − Xᵗ), and the difference between the two ERA5 input states
(Xᵗ − Xᵗ⁻¹). The results are written to `results/<mode>/increments/<model>/` and are used for the
increment maps.

| Script | Computes |
|---|---|
| `graphcast_increments.py` | Increments of the GraphCast decodings, processor steps 0–16 |
| `graphcast_input_difference.py` | Difference of the two GraphCast inputs (t−12 h and t−6 h) |
| `archesweather_increments.py` | Increments of the ArchesWeather decodings, all layers and blocks |
| `archesweather_input_difference.py` | Difference of the two ArchesWeather inputs, 24 h apart |
| `aurora_increments.py` | Increments of the Aurora decodings, encoder output and all blocks |
| `aurora_input_difference.py` | Difference of the two Aurora inputs (same times as GraphCast) |

## 2_relative_mse

`compute_relative_mse.py` computes the MSE of every intermediate decoding relative to persistence
(6 h and 24 h), per variable, averaged over all dates of the current mode. Output:
`results/<mode>/relative_mse/`. Without an argument it processes all three models; an optional
comma-separated list (e.g. `arches,aurora`) recomputes only those.

## 3_encoder_decoder_similarity

`compute_encoder_decoder_similarity.py` compares the input difference, the encoder–decoder increment
and the full-model increment with area-weighted pattern correlations and amplitude ratios, for all
variables. Output: `results/<mode>/encoder_decoder_similarity/<date>.pkl`. The argument is the number
of parallel workers, e.g. `python compute_encoder_decoder_similarity.py 1`.

## 4_spectral_filtering

Preprocessing for steps 5 and 6. The fields are transformed to spherical harmonics with `cdo` (winds
via divergence and vorticity), split into large scales (ℓ ≤ 20) and small scales (ℓ > 20), and
transformed back. `cdo` must be on the `PATH`. Temporary files go to `results/<mode>/tmp/`.

| Script | Computes | Output in `results/<mode>/` |
|---|---|---|
| `1_filter_intermediate_decodings.py` | Spectral coefficients and scale-separated fields of all intermediate decodings. Takes the model as a second argument (`arches`, `aurora`, `graphcast` or `all`). | `spectral_filtering/<model>/<date>/` (about 30 GB for GraphCast) |
| `2_filter_era5.py` | The same for the ERA5 state at the valid time | `spectral_filtering/era5_reference/<date>/` |

## 5_increment_spectra

`compute_increment_spectra_geopot_u10m.py` needs the spectral coefficients from step 4;
`compute_increment_spectra_all_variables.py` transforms all fields itself. Increment 0 is the first
decoded state minus the input state Xᵗ, increment k the state of layer k minus that of layer k − 1.

| Script | Computes | Output in `results/<mode>/spectra/<date>/` |
|---|---|---|
| `compute_increment_spectra_geopot_u10m.py` | Power spectra ℓ·P(ℓ) of the increments for geopotential at 500 hPa and 10 m zonal wind | `12_0_increment_spectra_<model>_500hPa.nc` |
| `compute_increment_spectra_all_variables.py` | The same spectra for all variables | `13_0_increment_spectra_<model>_all_vars.nc` |

## 6_physical_balances

Needs step 4. Two measures of balance are computed: the ratio of ageostrophic to geostrophic wind
(r_geo) and the ratio of divergence to vorticity (r_div).

| Script | Computes | Output in `results/<mode>/` |
|---|---|---|
| `1_create_spectrally_scaled_era5.py` | ERA5 rescaled in spectral space so that its energy per wavenumber matches each model | `spectrally_scaled_era5/<date>/` |
| `2_compute_balances_full_forecast.py` | r_geo and r_div on all pressure levels for the final forecasts, ERA5 and the rescaled ERA5 | `balances_full_forecast/<date>/` |
| `3_summarize_balances_full_forecast.py` | Vertical profiles of these ratios | `balances_full_forecast_summary/<date>.pkl` |
| `compute_balances_intermediate_decodings.py` | r_geo and r_div of all intermediate decodings, for large and small scales separately | `balances_intermediate_decodings/<date>.pkl` |
| `compute_balances_era5_scale_separated.py` | The same for ERA5, used as reference | `balances_era5_scale_separated/<date>.pkl` |
