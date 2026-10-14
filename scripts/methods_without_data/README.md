# methods_without_data

These scripts show how the translators and the covariance matrices of the paper were made. Both need
the latent states of every 6-hourly forecast in 2020, about 15 TB in total (12 TB of them from
Aurora), which we could not publish. None of the pipelines runs these scripts, but their results
are part of the published data: the translators are in `data/translators/` and the covariance
matrices in `results/paper/covariance_matrices/`.

To run the scripts yourself, extract the latent states (see the main README) and save them to
`data/latent_states/<model>/` as `all_latents_<date>.npz`.

## translator_training

A translator is an affine map, `W x + b`, applied at every grid point or mesh node. It maps an
intermediate latent state to a later latent state of the same model and is used for the tuned
decoding.

- **GraphCast:** every processor step is translated directly to the final latent state.
- **Aurora and ArchesWeather, skip translators** (used in the main paper): these models work on
  several resolutions. A latent state in one of the downsampled stages is translated to the last
  latent state of the same resolution. The remaining stages are then replaced by *stage translators*,
  so that the model's skip connections are still used.
- **Aurora and ArchesWeather, no-skip translators** (supplement only): every latent state is
  translated directly to the final latent state, upsampling where needed.

Each call trains one translator. The argument selects which one.

| Script | Translates | Argument |
|---|---|---|
| `graphcast/train_translators.py` | processor step *n* to step 16 | `<n> std`, n = 0–15 |
| `aurora/train_translator_encoder_output.py` | encoder output to the final latent state (x₄₇ with skip x₅) | – |
| `aurora/train_translators_encoder_stage0.py` | encoder stage 0, x₀–x₅, to the final latent state (x₄₇ with skip x₅) | 0–5 |
| `aurora/train_translators_encoder_stage1.py` | encoder stage 1, x₆–x₁₅, to x₄₁ (last block of decoder stage 1) | 6–15 |
| `aurora/train_translators_encoder_stage2.py` | encoder stage 2, x₁₆–x₂₃, to x₃₁ (last block of decoder stage 0) | 16–23 |
| `aurora/train_translators_decoder_stage0.py` | decoder stage 0, x₂₄–x₃₀, to x₃₁ | 24–30 |
| `aurora/train_translators_decoder_stage1.py` | decoder stage 1, x₃₂–x₄₀, to x₄₁ | 32–40 |
| `aurora/train_translators_decoder_stage2.py` | decoder stage 2, x₄₂–x₄₆, to x₄₇ (final) | 42–46 |
| `aurora/train_stage_translator_decoder_stage1.py` | stage translator for decoder stage 1 (stage input to x₄₁) | `0` |
| `aurora/train_stage_translator_decoder_stage2.py` | stage translator for decoder stage 2 (stage input to x₄₇) | `1` |
| `aurora/train_no_skip_translators.py` | no-skip translators of the downsampled stages | 0–35 |
| `archesweather/train_translators_layer0_and_1.py` | blocks of layers 0 and 1 to the final latent state | 0–4 |
| `archesweather/train_translators_layer2_and_3.py` | blocks of layers 2 and 3 to the last block of layer 3 | 0–22 |
| `archesweather/train_translators_layer4.py` | layer 4, blocks 0–2, to block 3 (final) | 0–2 |
| `archesweather/train_stage_translator_layer4.py` | stage translator for layer 4 (stage input to layer 4, block 3) | – |
| `archesweather/train_no_skip_translators_layer2_and_3.py` | no-skip translators of layers 2 and 3 to the final latent state | 0–23 |

The stage translators also need the inputs of the stage they replace. These are extracted in the same
way as the latent states: for Aurora into `latent_states/aurora/block_inputs/`, for ArchesWeather as
`layer4_block0_input_<date>.npz`.

The weights and training logs are written to `data/translators/`, next to the published ones. The
training-loss figures and the convergence table of the supplement are made from these logs.

The Aurora scripts number the translators consecutively (`translator_matrix_<id>_gnn.npz`). In the
published data the files are named by stage and block instead
(`translator_matrix_<enc|dec>_<stage>_<block>_gnn.npz`).

## covariance_matrices

These scripts compute the covariance matrix of each latent state, averaged over all grid points and
all forecasts of 2020. `plots/main/covariance_similarity.py` compares them across the depth of the
model using the cosine similarity of the matrices.

| Script | Computes |
|---|---|
| `<model>/covariance_untranslated.py` | Covariance matrices of the raw latent states. Also saves a few diagnostic plots to `figures/<mode>/covariance_diagnostics/`. |
| `<model>/covariance_translated.py` | The same after applying the translators. The Aurora script takes the block index (0–48) as argument. |

The results are written to `results/paper/covariance_matrices/<model>/`.
