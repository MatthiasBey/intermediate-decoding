# Interpretability of AI weather models via intermediate decoding

This repository contains the analysis and plotting code for the paper *Interpretability of AI weather
models via intermediate decoding* and its supplement.

We take the intermediate latent states of GraphCast, Aurora and ArchesWeather and map them back to
physical space with each model's own decoder, either directly or after an affine *translator*
(tuned decoding). The decoded fields are then analysed for forecast skill, for how the spatial scales
develop through the model, and for physical balance.

## Getting started

Create the conda environment, which also contains the `cdo` command line tool:

```bash
bash install_env.sh
conda activate intermediate_decoding
```

Then download the data (see [Data](#data)):

```bash
bash download_data.sh results           # 4.5 GB, enough for most figures of the paper
bash download_data.sh all               # 112 GB, everything
```

and run one of the three pipelines:

```bash
bash run_paper_figures.sh               # 1. all figures and tables of the paper
bash run_single_run_calculations.sh     # 2. recompute all diagnostics for the example forecast
bash run_single_run_figures.sh          # 3. plot the results of step 2
```

Each pipeline is split into steps, which can also be run on their own, for example
`bash run_single_run_calculations.sh spectral_filtering spectra`. The available steps are listed at
the bottom of each script. On a SLURM cluster, submit a pipeline with
`sbatch run_pipeline.sub run_paper_figures.sh` (adapt the resources in `run_pipeline.sub` first).

## Data

The paper evaluates 48 forecasts: the 10th of every month in 2020 at 00, 06, 12 and 18 UTC. The
translators and covariance matrices were computed from the latent states of every 6-hourly forecast
in 2020. All of this together is several tens of terabytes, too much to publish. Instead, two smaller
packages are published on Zenodo in three parts, [10.5281/zenodo.23345097](https://doi.org/10.5281/zenodo.23345097), [10.5281/zenodo.23345099](https://doi.org/10.5281/zenodo.23345099) and [10.5281/zenodo.23345101](https://doi.org/10.5281/zenodo.23345101):

- **`data/` (about 110 GB)**: everything needed to redo the analysis for one example forecast, valid
  2020-03-10 12 UTC. This includes the intermediate decodings, latent states and ERA5 fields of this
  forecast, the trained translators with their training logs, and the model losses of all 48
  forecasts. See `data/README.md`.
- **`results/paper/` (about 1.5 GB)**: the diagnostics behind the paper figures, computed for all 48
  forecasts, together with the covariance matrices of 2020. See `results/README.md`.

`download_data.sh` downloads and extracts them. With `results` it fetches `results/paper/` and the
small parts of `data/` (model losses, translators, land–sea mask), 4.5 GB in total. That is enough
for all figures averaged over the 48 forecasts, the covariance figure, the increment maps, the
translator training-loss figures and both tables. The remaining example-forecast figures
(`latent_state_development`, `encoder_decoder`, `tuned_decoding_comparison_*`, `latent_channel_*`)
and the single-run pipelines need everything, which you get with `all`. Run the script without
arguments to see all options.

## The three pipelines

**1. `run_paper_figures.sh`** recreates every figure and table of the paper. It does not compute
anything new: the figures averaged over 48 forecasts are plotted from `results/paper/`, and the
figures of the example forecast from `results/paper/` and `data/`. Figures are written to
`figures/paper/`, tables to `results/paper/`.

**2. `run_single_run_calculations.sh`** recomputes all diagnostics of the example forecast from the
raw data in `data/` and saves them to `results/single_run/`. This takes about 120 GB of disk space.

**3. `run_single_run_figures.sh`** makes the same figures and tables from the results of pipeline 2.
They show the example forecast only, so where the paper shows a mean over 48 forecasts, these
figures show a single date. Figures are written to `figures/single_run/`, tables to
`results/single_run/`. The covariance figure is left out, because it needs a full year of latent
states.

Pipelines 2 and 3 are meant to show that the published diagnostics can be reproduced from the raw
data. The files in `results/single_run/` have the same layout as those in `results/paper/`, so you can
compare them directly for 2020-03-10 12 UTC.

Most steps need 16–40 GB of memory.

## Figures and tables

Every figure and table has its own plot script in `scripts/plots/`, named after the figure. The
table below also lists which calculations in `scripts/calculations/` the figure depends on.

| Figure or table | Plot script | Calculations |
|---|---|---|
| **Main text** | | |
| `latent_state_development` | `main/latent_state_development.py` | – |
| `covariance_similarity` | `main/covariance_similarity.py` | see `scripts/methods_without_data/` |
| `relative_mse_all_variables` | `main/relative_mse_all_variables.py` | `2_relative_mse` |
| `encoder_decoder` | `main/encoder_decoder.py` | – |
| `graphcast_increments_spatial_maps` | `main/graphcast_increments_spatial_maps.py` | `1_increments` |
| `increment_spectra_geopot_u_10m` | `main/increment_spectra_geopot_u_10m.py` | `4_spectral_filtering`, `5_increment_spectra` |
| `physical_balances` | `main/physical_balances.py` | `4_spectral_filtering`, `6_physical_balances` |
| `physical_balances_500hPa` | `main/physical_balances_500hPa.py` | `4_spectral_filtering`, `6_physical_balances` |
| Table: encoder–decoder pattern correlation | `tables/encoder_decoder_pattern_correlation.py` | `3_encoder_decoder_similarity` |
| Table: translator convergence | `tables/translator_convergence.py` | – |
| **Supplement** | | |
| `{graphcast,aurora,arches}_increments_{geopot,u_10m}` | `supplement/increments_all_models.py` | `1_increments` |
| `increment_spectra_all_variables_{model}` | `supplement/increment_spectra_all_variables.py` | `4_spectral_filtering`, `5_increment_spectra` |
| `translator_training_loss_{graphcast,aurora,arches}` | `supplement/translator_training_loss.py` | – |
| `no_skip_translator_training_loss_{aurora,arches}` | `supplement/no_skip_translator_training_loss_*.py` | – |
| `tuned_decoding_comparison_{aurora,arches}` | `supplement/tuned_decoding_comparison_*.py` | – |
| `latent_channel_{aurora,arches}_*` | `supplement/latent_channel_*.py` | – |

Figures that show a single forecast end in its valid time (`2020-03-10-12-00`). Figures that average
over the 48 forecasts end in `mean_48_dates` in pipeline 1; in pipeline 3 they show the example
forecast only and end in its date as well. The training-loss figures and the covariance figure do
not depend on a forecast and have no date. The figures as they
appear in the paper are in `figures/published/`.

## Repository layout

```
├── install_env.sh, environment.yml   conda environment
├── download_data.sh                  downloads data/ and results/paper/
├── run_paper_figures.sh              pipeline 1
├── run_single_run_calculations.sh    pipeline 2
├── run_single_run_figures.sh         pipeline 3
├── run_pipeline.sub                  SLURM job script for the pipelines
├── scripts/
│   ├── config.py                     paths, dates and mode, used by all scripts
│   ├── calculations/                 diagnostics of a single forecast
│   ├── plots/                        one script per figure or table
│   ├── methods_without_data/         translator training and covariance matrices
│   └── utils/                        shared helper modules
├── data/                             input data of the example forecast (download)
├── results/
│   ├── paper/                        diagnostics of all 48 forecasts (download)
│   └── single_run/                   written by pipeline 2
└── figures/
    ├── published/                    figures as published in the paper
    ├── paper/                        written by pipeline 1
    └── single_run/                   written by pipeline 3
```

The folders `scripts/`, `data/` and `results/` and the subfolders of `scripts/` each have their own
README with more detail.

## What is not included

- **Running the models.** The code that runs GraphCast, Aurora and ArchesWeather and extracts their
  latent states is not part of this repository. For Aurora and ArchesWeather, the latent states can be
  read out with PyTorch forward hooks on the backbone blocks; for GraphCast, by returning the
  intermediate outputs of the processor. Passing a (translated) latent state to the model's decoder
  gives the intermediate decoding.
- **Latent states of the whole year 2020.** They are needed to train the translators and to compute
  the covariance matrices, but are about 14 TB in total. The scripts for both are in
  `scripts/methods_without_data/`, and their results are part of the published data.

## Configuration

All paths and dates are set in `scripts/config.py`. The data folders can also be moved elsewhere with
the environment variables `ID_DATA_DIR` and `ID_RESULTS_DIR`. Every script can be run on its own from
any directory, e.g.

```bash
python scripts/plots/supplement/increments_all_models.py 2020-03-10-12-00
```

By default, scripts run in single-run mode. Set `ID_MODE=paper` to work with the 48 forecasts of
the paper instead.

## A note on file names

Files are named after the time at which the forecast is valid (`2020-03-10-12-00`). The exception
are the Aurora data files and the diagnostics computed from them, which are named after the initial
time of the forecast (`2020-03-10-06-00-00`). All figures use the valid time.

## License

The code in this repository is released under the MIT License, see `LICENSE`. The data published
with the paper have their own license, given in the data repository. ERA5 data are provided by the
Copernicus Climate Change Service under the Copernicus licence.
