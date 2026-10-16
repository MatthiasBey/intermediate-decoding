"""One latent channel of all Aurora blocks (supplement), in three figures: untranslated, with
the skip translators and with the no_skip translators.

skip: per-token affine channel map (translator_matrix_<part>_<stage>_<block>_gnn.npz) to the
    last decoder state on the same grid (x_31, x_41 or the final state x_47).
no_skip: for enc2, enc3, dec1 and dec2 the translator also upsamples to the full-resolution grid
    (translator_matrix_upscale_<stage>_<block>_gnn.npz, trained by
    methods_without_data/translator_training/aurora/train_no_skip_translators.py). For enc1 and
    dec3 it is the same as skip.
The skip training scripts save files named by translator id (translator_matrix_<0..46>_gnn.npz);
the published weights use the names above.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
from utils.latent_channel_figure import plot_latent_channel

CHANNEL = 100  # channel and latent level shown in the supplement
LEVEL = 1
FIGURE_NAMES = {"raw": "untranslated", "skip": "translated_within_stage", "no_skip": "translated_to_final"}

# (latent levels, lat, lon) of the three resolutions of the Swin3D backbone. Encoder stages go
# from fine to coarse, decoder stages from coarse to fine.
GRIDS = [(4, 180, 360), (4, 90, 180), (4, 45, 90)]
STAGES = [  # (part, stage, row label, grid)
    ("enc", 0, "enc1", GRIDS[0]),
    ("enc", 1, "enc2", GRIDS[1]),
    ("enc", 2, "enc3", GRIDS[2]),
    ("dec", 0, "dec1", GRIDS[2]),
    ("dec", 1, "dec2", GRIDS[1]),
    ("dec", 2, "dec3", GRIDS[0]),
]
UPSCALE_STAGES = {("enc", 1): "enc_1", ("enc", 2): "enc_2", ("dec", 0): "dec_0", ("dec", 1): "dec_1"}
UPSCALE_OUT_DIM = 1024  # output channels of the upscale translators

# dec1 block 7 (x_31) and dec2 block 9 (x_41) are targets of skip translators and have none
# themselves; the raw field is shown instead
NO_SKIP_TRANSLATOR = {("dec", 0, 7), ("dec", 1, 9)}

analysis_datetime = config.analysis_date()  # forecast valid time, optional argument
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")
init_str = config.aurora_init_time(analysis_datetime).strftime("%Y-%m-%d-%H-%M-%S")
skip_dir = f"{config.TRANSLATOR_DIR}/aurora/skip/weights"
no_skip_dir = f"{config.TRANSLATOR_DIR}/aurora/no_skip/weights"


def load_translator(path):
    with np.load(path) as tr:
        return tr["W"].astype(np.float32), tr["b"].astype(np.float32)


def upscale(x, W, b, grid):
    """Upscale translator as UpscaleTranslator.forward() in train_no_skip_translators.py (without
    batch axis); W is stored as (out_dim, in_dim)."""
    c, h, w = grid
    factor = GRIDS[0][1] // h
    x = x @ W.T + b
    x = x.reshape(c, h, w, factor, factor, UPSCALE_OUT_DIM)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(c * h * factor * w * factor, UPSCALE_OUT_DIM)


def channel_field(x, grid):
    """(lat, lon) field of CHANNEL at LEVEL from tokens x (C*H*W, D), or None if x has fewer
    channels."""
    if CHANNEL >= x.shape[-1]:
        return None
    c, h, w = grid
    field = x[:, CHANNEL].reshape(c, h, w)
    field = np.flip(field, axis=1)               # latitude orientation
    field = np.roll(field, shift=w // 2, axis=2)  # longitude phase
    return field[LEVEL]


# %% Raw, skip and no_skip field of every block. The npz file (about 6 GB, 48 float16 blocks) is
# read block by block.
fields = {kind: {} for kind in FIGURE_NAMES}  # fields[kind][row label] = [field per block]
with np.load(f"{config.LATENT_DIR}/aurora/all_latents_aurora-ft_{init_str}.npz", allow_pickle=False) as npz:
    record_index = {(str(part), int(stage), int(block)): i
                    for i, (part, stage, block) in enumerate(zip(npz["part"], npz["stage"], npz["block"]))}

    for part, stage, label, grid in STAGES:
        stage_blocks = sorted(block for p, s, block in record_index if (p, s) == (part, stage))
        for kind in fields:
            fields[kind][label] = []
        for block in stage_blocks:
            i = record_index[(part, stage, block)]
            x = npz[f"x_{i}"][0].astype(np.float32)  # (C*H*W, D)
            raw = channel_field(x, grid)
            is_final_state = part == "dec" and stage == 2 and block == stage_blocks[-1]

            # skip translator, on the block's own grid
            skip_path = os.path.join(skip_dir, f"translator_matrix_{part}_{stage}_{block}_gnn.npz")
            if is_final_state or (part, stage, block) in NO_SKIP_TRANSLATOR:
                skip = raw
            elif not os.path.exists(skip_path):
                print(f"{label} block {block} (x_{i}): no skip translator, panel left blank")
                skip = None
            else:
                W, b = load_translator(skip_path)
                skip = channel_field(x @ W + b, grid)  # W is stored as (in_dim, out_dim)

            # no_skip translator, to the full-resolution grid
            if is_final_state:
                no_skip = raw
            elif (part, stage) in UPSCALE_STAGES:
                path = os.path.join(no_skip_dir, f"translator_matrix_upscale_{UPSCALE_STAGES[(part, stage)]}_{block}_gnn.npz")
                if os.path.exists(path):
                    no_skip = channel_field(upscale(x, *load_translator(path), grid), GRIDS[0])
                else:
                    print(f"{label} block {block} (x_{i}): no upscale translator, panel left blank")
                    no_skip = None
            else:
                no_skip = skip

            for kind, field in (("raw", raw), ("skip", skip), ("no_skip", no_skip)):
                fields[kind][label].append(field)
            print(f"{label} block {block} (x_{i}): loaded")

# %% One figure per kind
for kind, figure_name in FIGURE_NAMES.items():
    rows = []
    for part, stage, label, grid in STAGES:
        on_full_grid = kind == "no_skip" and (part, stage) in UPSCALE_STAGES
        rows.append((label, fields[kind][label], (GRIDS[0] if on_full_grid else grid)[1:]))
    plot_latent_channel(
        rows, f"Aurora latent channel {CHANNEL} (level {LEVEL}) — {kind}",
        os.path.join(config.SUPPLEMENT_DIR, f"latent_channel_aurora_ch{CHANNEL}_lv{LEVEL}_{figure_name}_{time_str}.png"),
        origin="lower")
