"""One latent channel of all ArchesWeather blocks (supplement), in three figures: untranslated,
with the skip translators and with the no_skip translators.

skip: per-token affine channel map on the layer's own grid, to the final latent state (layer 4
    block 3) for layers 0, 1 and 4 and to layer 3 block 11 for the coarse layers 2 and 3.
no_skip: for layers 2 and 3 the translator also learns the 2x2 lat/lon upsampling to the final
    (60, 120) grid. For layers 0, 1 and 4 (already at full resolution) it is the same as skip.
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
from utils.latent_channel_figure import plot_latent_channel

CHANNEL = 100  # channel and latent level shown in the supplement
LEVEL = 2
FIGURE_NAMES = {"raw": "untranslated", "skip": "translated_within_stage", "no_skip": "translated_to_final"}

N_LEVELS = 8  # latent pressure levels
FULL_GRID = (60, 120)
LAYERS = [  # (layer, number of blocks, (lat, lon) grid); layers 2 and 3 are downsampled
    ("layer0", 1, FULL_GRID),
    ("layer1", 4, FULL_GRID),
    ("layer2", 12, (30, 60)),
    ("layer3", 12, (30, 60)),
    ("layer4", 4, FULL_GRID),
]
EXPAND_LAYERS = {"layer2", "layer3"}  # layers where skip and no_skip differ
EXPAND_OUT_DIM = 384

# layer3 block 11 is the target of the skip translators of layers 2 and 3 and has none itself;
# the raw field is shown instead
NO_SKIP_TRANSLATOR = {("layer3", 11)}

time_str = config.analysis_date().strftime("%Y-%m-%d-%H-%M")  # forecast valid time, optional argument
skip_dir = f"{config.TRANSLATOR_DIR}/archesweather/skip/weights/"
no_skip_dir = f"{config.TRANSLATOR_DIR}/archesweather/no_skip/weights/"


def load_translator(path):
    """W is stored as (out_features, in_features), as in torch.nn.Linear."""
    with np.load(path) as tr:
        return tr["W"].astype(np.float32), tr["b"].astype(np.float32)


def expand(x, W, b):
    """Affine map followed by the 2x2 lat/lon upsampling, as ExpandTranslator.forward() in
    train_no_skip_translators_layer2_and_3.py (without batch axis)."""
    lat, lon = 30, 60
    x = x @ W.T + b  # (14400, 1536)
    x = x.reshape(N_LEVELS, lat, lon, 2, 2, EXPAND_OUT_DIM)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(N_LEVELS * lat * 2 * lon * 2, EXPAND_OUT_DIM)  # (57600, 384)


def channel_field(x, grid):
    """(lat, lon) field of CHANNEL at LEVEL from tokens x (levels*lat*lon, D), or None if x has
    fewer channels (the raw latents of layers 0 and 1 have 192 channels, all others 384)."""
    if CHANNEL >= x.shape[-1]:
        return None
    return x[:, CHANNEL].reshape(N_LEVELS, *grid)[LEVEL]


# %% Raw, skip and no_skip field of every block
fields = {kind: {} for kind in FIGURE_NAMES}  # fields[kind][layer] = [field per block]
with np.load(f"{config.LATENT_DIR}/archesweather/all_latents_{time_str}.npz", allow_pickle=False) as npz:
    # keys like 'layer1.blocks.0' become 'layer1_block0'
    latents = {re.sub(r"layer(\d+)\.blocks\.(\d+)", r"layer\1_block\2", key): npz[key] for key in npz.files}

for layer, n_blocks, grid in LAYERS:
    for kind in fields:
        fields[kind][layer] = []
    for block in range(n_blocks):
        x = np.asarray(latents[f"{layer}_block{block}"], dtype=np.float32)
        if x.ndim == 3:  # drop batch axis
            x = x[0]
        raw = channel_field(x, grid)
        is_final_state = layer == "layer4" and block == n_blocks - 1

        # skip translator, on the layer's own grid
        skip_path = os.path.join(skip_dir, f"Arches_translator_matrix_{layer}_block{block}.npz")
        if is_final_state or (layer, block) in NO_SKIP_TRANSLATOR:
            skip = raw
        elif not os.path.exists(skip_path):
            print(f"{layer} block {block}: no skip translator, panel left blank")
            skip = None
        else:
            W, b = load_translator(skip_path)
            skip = channel_field(x @ W.T + b, grid)

        # no_skip translator, to the full grid
        if is_final_state:
            no_skip = raw
        elif layer in EXPAND_LAYERS:
            path = os.path.join(no_skip_dir, f"Arches_translator_rep_stab_matrix_{layer}_block{block}.npz")
            if os.path.exists(path):
                no_skip = channel_field(expand(x, *load_translator(path)), FULL_GRID)
            else:
                print(f"{layer} block {block}: no expand translator, panel left blank")
                no_skip = None
        else:
            no_skip = skip

        for kind, field in (("raw", raw), ("skip", skip), ("no_skip", no_skip)):
            fields[kind][layer].append(field)
        print(f"{layer} block {block}: loaded")

# %% One figure per kind
for kind, figure_name in FIGURE_NAMES.items():
    rows = [(layer, fields[kind][layer], FULL_GRID if kind == "no_skip" and layer in EXPAND_LAYERS else grid)
            for layer, _, grid in LAYERS]
    plot_latent_channel(
        rows, f"ArchesWeather latent channel {CHANNEL} (level {LEVEL}) — {kind}",
        os.path.join(config.SUPPLEMENT_DIR, f"latent_channel_arches_ch{CHANNEL}_lv{LEVEL}_{figure_name}_{time_str}.png"),
        origin="upper")
