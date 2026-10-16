# Code cleanup (October 2026)

This file documents a readability cleanup of the scripts. The goal was code that is easier to
read, without changing any calculation or figure. The calculations use the same functions and
operations as before; unused code was removed, and long plot scripts were restructured.

How this was checked is described at the end ([Verification](#verification)).

## Overview

- The plot scripts are about half as long (about 4,450 to 2,150 lines, plus 225 lines in three
  new helper modules). Repeated code (six almost identical heatmap calls, the same panel code for
  every model, the same log-reading functions in four scripts) was replaced by loops and small
  functions.
- Unused code was removed: options that were never used, branches for a GraphCast version that
  is no longer computed, unused parameters and variables, and one obsolete calculation script.
- Every script starts with a docstring that says what it does, followed by the imports in the
  usual order.

## All scripts: header

Every script used to find the `scripts/` folder with a loop:

```python
import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
```

This is now one line with the folder depth of the script, after the other imports:

```python
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
```

The description of each script, which was a comment block somewhere after the imports or a string
after the imports (training scripts), is now the module docstring at the top.

## Removed files

| File | Why |
|---|---|
| `calculations/4_spectral_filtering/3_spectral_power_of_states_and_input.py` | Its power spectra (`11_4_…pkl`) were not used by any figure. The one output that was needed, the spectral coefficients of the input state X_t, is now computed directly in `5_increment_spectra/compute_increment_spectra_geopot_u10m.py` (same transform). |
| `calculations/1_increments/aurora_input_difference.py` | Its output was not used: Aurora gets the same ERA5 inputs as GraphCast, and the increment maps use the GraphCast input difference for Aurora. |

(Both were removed shortly before this cleanup; they are listed here for completeness.)

**Bug fix:** the removed `3_spectral_power_of_states_and_input.py` created the output folder
`results/<mode>/spectra/<date>/`. Without it, `compute_increment_spectra_geopot_u10m.py` failed
on a fresh results folder ("Permission denied" when writing its output). It now creates the
folder itself. This affected the version of commit `c884400`.

## New helper modules (`scripts/utils/`)

| Module | Content | Used by |
|---|---|---|
| `translator_logs.py` | Reading the translator training logs (`load_losses`, `load_initial_losses`, `load_run`) and plotting loss curves (`plot_runs`, `plot_runs_with_zoom`). These functions were copied in four scripts before. | `translator_training_loss.py`, `no_skip_translator_training_loss_*.py`, `translator_convergence.py` |
| `increment_spectra.py` | Loading, normalizing and drawing the increment spectra (`load_mean_spectra`, `normalize`, `centroid`, `layer_ticks`, `draw_spectra`). These were copied in two scripts before. | `increment_spectra_geopot_u_10m.py`, `increment_spectra_all_variables.py` |
| `latent_channel_figure.py` | The figure of one latent channel for all blocks of a model (`plot_latent_channel`). The two scripts had almost identical plot functions. | `latent_channel_aurora.py`, `latent_channel_arches.py` |

## Plot scripts (`scripts/plots/`)

| Script | Lines before / after | Changes |
|---|---|---|
| `main/latent_state_development.py` | 564 / 272 | Only the 9 GraphCast steps that are shown are translated (before: all 16, then every second one selected; each step is independent, so the values are the same). The ArchesWeather latent states are reshaped with numpy directly instead of via torch tensors and an xarray dataset (pure reshapes, same values). Repeated panel code moved into `image_panel`, `label_panel` and `section_title`. The translations themselves are unchanged (GraphCast in bfloat16 with jax, ArchesWeather with `np.dot`, Aurora with `x @ W + b`). torch is no longer imported. |
| `main/encoder_decoder.py` | 501 / 145 | Only the three fields per model that are shown are loaded (input difference, first and last decoded state), not all intermediate decodings of all models. The unused GraphCast versions `s`/`op`, the unused `var_levels`, the print of all keys and the land-sea tint option (always off) were removed. The style dictionary was replaced by a few constants. |
| `main/relative_mse_all_variables.py` | 372 / 197 | The 49 Aurora state names are generated instead of written out. The three loss panels are drawn in one loop. Unused settings (e.g. the figure title size of a title that was not shown) removed. The axes are still created in the same order, so that the layout is the same. |
| `main/covariance_similarity.py` | 327 / 119 | The six heatmap calls are one loop over the panels. The option `remove_top_k` (remove leading principal components; always 0) and its function were removed. The two Aurora loaders were merged. |
| `main/physical_balances_500hPa.py` | 183 / 129 | Written for one level (500 hPa) instead of a dictionary of level ranges with one entry. |
| `main/physical_balances.py` | 121 / 88 | Small cleanup (date strings, constants). |
| `main/increment_spectra_geopot_u_10m.py` | 146 / 63 | Uses `utils/increment_spectra.py`. |
| `main/graphcast_increments_spatial_maps.py` | 113 / 71 | The four file names are generated from the step numbers; small cleanup. |
| `supplement/translator_training_loss.py` | 341 / 127 | Uses `utils/translator_logs.py`; the three plot functions share the panel code. |
| `supplement/no_skip_translator_training_loss_arches.py` | 195 / 46 | Uses `utils/translator_logs.py`. |
| `supplement/no_skip_translator_training_loss_aurora.py` | 167 / 40 | Uses `utils/translator_logs.py`. |
| `supplement/latent_channel_aurora.py` | 298 / 135 | Written for the one channel and level that is shown (`CHANNEL`, `LEVEL`) instead of lists of channels and levels with one entry each, which needed four levels of nested dictionaries. Uses `utils/latent_channel_figure.py`. The upscaling parameters are derived from the stage grid instead of a separate table. |
| `supplement/latent_channel_arches.py` | 274 / 121 | As for Aurora. |
| `supplement/tuned_decoding_comparison_aurora.py` | 192 / 103 | One loading function and one relative-MSE function for both decodings; blocks in network order from the start. |
| `supplement/tuned_decoding_comparison_arches.py` | 144 / 91 | As for Aurora. |
| `supplement/increment_spectra_all_variables.py` | 150 / 74 | Uses `utils/increment_spectra.py`. |
| `supplement/increments_all_models.py` | 131 / 97 | Small cleanup; the style overrides moved into `utils/plot_spatial_maps_in_raster.py`. |
| `tables/translator_convergence.py` | 205 / 133 | Uses `utils/translator_logs.py`; one function for a table row. |
| `tables/encoder_decoder_pattern_correlation.py` | 41 / 36 | Small cleanup. |

`utils/plot_spatial_maps_in_raster.py` (292 / 102 lines): the plot function is only used by
`increments_all_models.py`, always with one colorbar per panel and the same style. The unused
variant with one shared colorbar, the unused `cbar_levels` parameter and the style options that
were never read were removed; the style values that were used are now written in the function.

## Calculation scripts (`scripts/calculations/`)

The calculations are unchanged; only unused code was removed.

| Script | Changes |
|---|---|
| `2_relative_mse/compute_relative_mse.py` (340 / 191 lines) | The branches for the GraphCast version `s` (1 degree) and the 1-degree truth and persistence fields that only they used were removed (only `std` is computed). The six Aurora loading loops are one loop. The six conversion functions for the pickle are two (`mean_over_dates`, `to_floats`). The relative MSE is computed with the same steps in one helper function. The pickle has the same structure and key order. |
| `1_increments/*.py` | The list of the 48 dates is taken from `config.DATES_48` (identical list) instead of being written out in every script. |
| `6_physical_balances/1_create_spectrally_scaled_era5.py` | The unused return values of the `reconstruct` calls (`recon_*`) are no longer stored. |
| `6_physical_balances/*.py` | `lat_cut=None` is no longer passed (see `physical_consistency_functions.py`). |
| all | Header and docstring as described above. |

## Helper modules (`scripts/utils/`)

| Module | Changes |
|---|---|
| `load_data.py` (103 / 27 lines) | Reduced to what is used: all callers pass only a start and an optional end date. Removed: the merge of a constants file that does not exist (it was never merged), the unused parameters `variables`, `chunks`, `include_constants` and `preprocess`, and the environment variable `GRAPHCAST_INIT_DIR`. The `open_mfdataset` call is the same. |
| `physical_consistency_functions.py` (171 / 92 lines) | `calculate_divergence_vorticity_ratio_from_fields` had a branch for the extratropics (`lat_cut`), which was never used (all callers passed `lat_cut=None`); removed. The unused `lat` parameter of `calculate_geostrophic_wind` was removed. The returned dictionaries are the same. |
| `spectral_filter_for_xarray_fields_using_cdo.py` (219 / 134 lines) | The unused parameters `remapmethod` (of `field_to_spectral` and `field_to_spectral_clean`) and `l_min` (of `truncate_spectral_file`) were removed; the cdo calls are written with a small `run` function. The cdo commands are the same. |
| `encoder_decoder_similarity.py` | An unused argument in a path template removed. |

## Training and covariance scripts (`scripts/methods_without_data/`)

These scripts need the latent states of the whole year 2020, which are not published, so they
could not be run. They were therefore changed as little as possible:

- header and docstring as described above,
- f-strings without placeholders are normal strings,
- a debugging remark was removed from a log message
  ("takes longer or shorter then loading? check if this is the bottleneck").

The training logic is unchanged. The six Aurora scripts `train_translators_{encoder,decoder}_stage{0,1,2}.py`
differ only in the stage, the target state and the number of channels and could be merged into
one script with a table of stages. This was not done, because it is a larger change to code that
cannot be tested here.

## Not changed

Things that were noticed but deliberately left as they are, because changing them would change a
figure or a result:

- `translator_training_loss.py`: the lower ArchesWeather panel is titled "Layer 1 & 4", but also
  shows layer 0.
- `relative_mse_all_variables.py`: panel f) draws an empty legend box, as in the published figure.
- `encoder_decoder.py`: the input difference for GraphCast and Aurora is the difference of the
  second and first time step in the ERA5 file of the valid date (00 and 06 UTC). This equals
  X(t-6h) - X(t-12h) for the example forecast valid at 12 UTC, but not for other valid times.
- The GraphCast "version" loops over the single version `std` in the spectral filtering and
  balance scripts, since the version is part of the output file names.

## Verification

All three pipelines were run with the old and the new scripts and the outputs compared:

1. **Figures (pipelines 1 and 3).** `run_paper_figures.sh` and `run_single_run_figures.sh` were run
   with the scripts before the cleanup (baseline) and after it. All figures are identical pixel by
   pixel: 30 of 30 in paper mode, 29 of 29 in single-run mode. All tables (the encoder-decoder
   pattern correlation and the three translator convergence tables, in both modes) are identical
   byte by byte.
2. **Calculations (pipeline 2).** `run_single_run_calculations.sh` was run with the new scripts
   into an empty results folder and every output file (netCDF, pickle) was compared value by value
   with `results/single_run/`, computed with the old scripts: increments (101 files), relative MSE,
   encoder-decoder similarity, spectral filtering (500 files), spectra (6), spectrally scaled ERA5
   (11) and the balances (10). All 631 files are identical (largest difference 0), including the
   key order of the pickles. This run found the bug described above.
3. **Pipeline 3 on the new results.** `run_single_run_figures.sh` on the results of step 2 gives the
   same 29 figures and the same tables as the baseline, again pixel by pixel and byte by byte.
4. **Training and covariance scripts** (`methods_without_data/`, cannot be run without the
   latent states of 2020): all scripts compile, and a comparison of their syntax trees with the
   old versions shows that apart from the header, the docstring position and the changed strings
   the code is the same.
