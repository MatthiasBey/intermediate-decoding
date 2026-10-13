# scripts

All code of the repository. Each subfolder has its own README.

| Folder or file | Content |
|---|---|
| `config.py` | Paths, dates and mode. Every script imports it. |
| `calculations/` | Diagnostics of a single forecast, written to `results/<mode>/` |
| `plots/` | One script per figure or table of the paper, written to `figures/<mode>/` |
| `methods_without_data/` | Translator training and covariance matrices. These need the latent states of the whole year 2020, which are not published. |
| `utils/` | Shared helper modules for loading ERA5, spectral filtering with cdo, physical balances, encoder–decoder similarity and map plotting. Also `pack_data.sh`, which packs `data/` and `results/paper/` into the archives published with the paper. |

## Modes

The scripts run in one of two modes, set with the environment variable `ID_MODE`:

- `paper` works with the 48 forecasts of the paper and reads the published diagnostics in
  `results/paper/`. Pipeline 1 (`run_paper_figures.sh`) uses this mode.
- `single_run` (the default) works with the example forecast only and reads and writes
  `results/single_run/`. Pipelines 2 and 3 use this mode.

Each script can also be run on its own, from any directory:

```bash
python scripts/plots/supplement/increments_all_models.py 2020-03-10-12-00
ID_MODE=paper python scripts/plots/supplement/increments_all_models.py 2020-03-10-12-00
```
