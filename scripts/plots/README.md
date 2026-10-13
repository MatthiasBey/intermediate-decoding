# plots

One script per figure or table of the paper, named after the figure. All scripts work in both modes:

- In **paper** mode (`run_paper_figures.sh`), figures that average over the 48 forecasts read the
  published diagnostics in `results/paper/`. Their names end in `mean_48_dates`.
- In **single_run** mode (`run_single_run_figures.sh`), the same scripts read the results of
  `run_single_run_calculations.sh` in `results/single_run/`. The figures then show the example
  forecast only, and their names end in its date, `2020-03-10-12-00`.

Figures are saved to `figures/<mode>/`, supplementary figures to `figures/<mode>/supplement/` and
tables to `results/<mode>/`.

Scripts that show a single forecast take its valid time as an optional argument, e.g.
`python main/encoder_decoder.py 2020-03-10-12-00`. Without an argument they use the example
forecast. Their figure names end in that date. The other scripts take no argument.

## main

| Script | Shows | Reads |
|---|---|---|
| `latent_state_development.py` | How the latent state develops through each model | latent states and translators in `data/` |
| `covariance_similarity.py` | Similarity of the latent covariance matrices across model depth (paper mode only) | `results/paper/covariance_matrices/` |
| `relative_mse_all_variables.py` | Model loss of the direct and tuned decodings, and the relative MSE of all variables | `data/model_loss/`, `results/<mode>/relative_mse/` |
| `encoder_decoder.py` | Input difference, encoder–decoder increment and full-model increment for geopotential at 500 hPa and 2 m temperature | decodings and ERA5 in `data/` |
| `graphcast_increments_spatial_maps.py` | Maps of selected GraphCast increments | `results/<mode>/increments/graphcast/` |
| `increment_spectra_geopot_u_10m.py` | Increment spectra and spectral centroids | `results/<mode>/spectra/` |
| `physical_balances.py` | Vertical profiles of r_geo and r_div for the final forecasts and ERA5 | `results/<mode>/balances_full_forecast_summary/` |
| `physical_balances_500hPa.py` | r_geo and r_div of all intermediate decodings against normalised depth, for large and small scales | `results/<mode>/balances_intermediate_decodings/`, `results/<mode>/balances_era5_scale_separated/` |

## supplement

| Script | Shows | Reads |
|---|---|---|
| `increments_all_models.py` | Increment maps of all three models for geopotential and 10 m zonal wind | `results/<mode>/increments/` |
| `increment_spectra_all_variables.py` | Increment spectra of all variables | `results/<mode>/spectra/` |
| `translator_training_loss.py` | Training loss of the translators | training logs in `data/translators/` |
| `no_skip_translator_training_loss_aurora.py`, `…_arches.py` | Training loss of the no-skip translators | training logs in `data/translators/` |
| `tuned_decoding_comparison_aurora.py`, `…_arches.py` | Tuned decoding compared with the no-skip decoding | decodings in `data/` |
| `latent_channel_aurora.py`, `…_arches.py` | One latent channel, untranslated and translated | latent states and translators in `data/` |

The latent-channel scripts show channel 100 at one level; to show a different channel or level, change
`paper_channels` and `level_indices` at the top of the script. These figures are saved at 150 dpi
with a 256-colour palette to keep the files small.

## tables

| Script | Table | Reads |
|---|---|---|
| `encoder_decoder_pattern_correlation.py` | Pattern correlation between the encoder–decoder increment and the full-model increment for all variables, mean and standard deviation over all dates. Saved as `results/<mode>/encoder_decoder_similarity/summary_<mean_48_dates or date>.csv`. | `results/<mode>/encoder_decoder_similarity/` |
| `translator_convergence.py` | Number of epochs until the translators converge. Saved to `results/<mode>/translator_convergence/`. | training logs in `data/translators/` |
