# data

Input data for the example forecast, valid 2020-03-10 12 UTC. At about 110 GB the data are too large
for git. They are published together with the diagnostics in `results/paper/` at
**[add DOI / link]**. Download them with `bash download_data.sh all` from the repository folder. To
keep them somewhere else, set the environment variable `ID_DATA_DIR` to that location before
downloading and running the scripts.

Everything here belongs to the example forecast, except `model_loss/`, which covers all 48 forecasts
of the paper.

```
data/
├── era5/
│   ├── era4aim_2020-03-{08,09,10}.nc     ERA5 at 0.25°, the model input variables
│   └── constants/era4aim_const.nc        land–sea mask and orography
├── intermediate_decodings/
│   ├── tuned/                            decodings of the translated latent states
│   │   ├── graphcast/                    processor steps 0–16 (16 is the full forecast)
│   │   ├── aurora/                       encoder output and all encoder and decoder blocks
│   │   └── archesweather/                all layers and blocks (layer 4, block 3 is the full forecast)
│   └── no_skip/                          decodings with the no-skip translators (supplement)
│       ├── aurora/
│       └── archesweather/
├── latent_states/<model>/                latent states of all layers
├── translators/                          translator weights (W and b) and training logs
│   ├── graphcast/{weights,logs}/
│   ├── aurora/{skip,no_skip}/{weights,logs}/
│   └── archesweather/{skip,no_skip}/{weights,logs}/
└── model_loss/<model>/{direct,tuned}/    model loss of each latent state, all 48 forecasts
```

ERA5 is included for three days because ArchesWeather uses inputs 24 and 48 hours before the valid
time.

| Folder | Size |
|---|---|
| `era5/` | 17 GB |
| `intermediate_decodings/tuned/` | 61 GB (GraphCast 32, Aurora 28, ArchesWeather 0.6) |
| `intermediate_decodings/no_skip/` | 21 GB |
| `latent_states/` | 8.4 GB |
| `translators/` | 3.0 GB |
| `model_loss/` | < 0.1 GB |

## File names

Files are named after the time at which the forecast is valid (`2020-03-10-12-00`), except for Aurora
files, which are named after the initial time (`2020-03-10-06-00-00`).

The Aurora translators are named by stage and block,
`translator_matrix_<enc|dec>_<stage>_<block>_gnn.npz`, plus `translator_matrix_encoder_output_gnn.npz`
for the encoder output.

## Translators

The *skip* translators map a latent state in one of the downsampled stages to the last latent state
of the same resolution. *Stage translators* then replace the remaining stages, so that the model's
skip connections are still used:

- Aurora: `stage_translator_decoder_stage1_gnn.npz` and `stage_translator_decoder_stage2_gnn.npz`
- ArchesWeather: `Arches_stage_translator_layer4.npz`

The *no-skip* translators map every latent state directly to the final latent state at full
resolution. They are only used in the supplement.

More on the translators is in `scripts/methods_without_data/README.md`.

## Sources

- ERA5 reanalysis from the Copernicus Climate Change Service (C3S), regridded to the input format of
  the models.
- Forecasts, latent states and decodings were produced with GraphCast, Aurora (0.25° fine-tuned
  version) and ArchesWeather.
