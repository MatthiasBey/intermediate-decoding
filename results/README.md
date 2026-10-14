# results

Computed diagnostics. Figures are saved separately in `figures/`.

```
results/
├── paper/        diagnostics of all 48 forecasts of the paper (download)
└── single_run/   diagnostics of the example forecast, written by run_single_run_calculations.sh
```

## paper

The diagnostics behind the figures and tables of the paper, published together with `data/` on
Zenodo in three parts, [10.5281/zenodo.23345097](https://doi.org/10.5281/zenodo.23345097), [10.5281/zenodo.23345099](https://doi.org/10.5281/zenodo.23345099) and [10.5281/zenodo.23345101](https://doi.org/10.5281/zenodo.23345101). `bash download_data.sh results` downloads them into this folder.
`run_paper_figures.sh` plots the paper figures from these files and saves its tables here as well.

The diagnostics cover the 48 forecasts of the paper (the 10th of every month in 2020 at 00, 06, 12
and 18 UTC). Most folders hold one file per forecast, with the same layout as in `single_run/`.

| Folder | Made by (`scripts/`) | Used by (`scripts/`) |
|---|---|---|
| `relative_mse/` | `calculations/2_relative_mse/compute_relative_mse.py` | `plots/main/relative_mse_all_variables.py` |
| `encoder_decoder_similarity/` | `calculations/3_encoder_decoder_similarity/compute_encoder_decoder_similarity.py` | `plots/tables/encoder_decoder_pattern_correlation.py` |
| `spectra/` | `calculations/5_increment_spectra/` | `plots/main/increment_spectra_geopot_u_10m.py`, `plots/supplement/increment_spectra_all_variables.py` |
| `balances_full_forecast_summary/` | `calculations/6_physical_balances/3_summarize_balances_full_forecast.py` | `plots/main/physical_balances.py` |
| `balances_intermediate_decodings/` | `calculations/6_physical_balances/compute_balances_intermediate_decodings.py` | `plots/main/physical_balances_500hPa.py` |
| `balances_era5_scale_separated/` | `calculations/6_physical_balances/compute_balances_era5_scale_separated.py` | `plots/main/physical_balances_500hPa.py` |
| `increments/` | `calculations/1_increments/` | `plots/main/graphcast_increments_spatial_maps.py`, `plots/supplement/increments_all_models.py` |
| `covariance_matrices/` | `methods_without_data/covariance_matrices/` | `plots/main/covariance_similarity.py` |

Two folders differ from the rest:

- `increments/` covers only the example forecast, and only the two fields shown in the increment
  maps: geopotential at 500 hPa and 10 m zonal wind. The full files contain all variables and levels
  and take up 28 GB; `scripts/utils/slim_increments.py` reduces them to these two fields.
- `covariance_matrices/` holds the covariance matrices of the latent states, averaged over all
  forecasts of 2020, both raw and after translation (`withtranslator_*`). They cannot be computed from
  a single forecast, so the covariance figure is only made by `run_paper_figures.sh`.

Total size: about 1.5 GB (`spectra/` 0.6 GB, `covariance_matrices/` 0.6 GB, `increments/` 0.3 GB,
the rest less than 20 MB).

## single_run

Written by `run_single_run_calculations.sh` and read by `run_single_run_figures.sh`, which also saves
its tables here. It takes up about 120 GB, mostly the spectrally filtered fields in
`spectral_filtering/` and the increments in `increments/`. `tmp/` holds temporary files of the
spectral filtering. To put this folder somewhere else, for example on a scratch file system, set the
environment variable `ID_RESULTS_DIR`.
