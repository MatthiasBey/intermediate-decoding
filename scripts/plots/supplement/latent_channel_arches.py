import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% IMPORTS
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

from utils.reduce_png import reduce_png

# %% CONFIG

data_dir = f"{config.LATENT_DIR}/archesweather/"
timestamp = config.analysis_date().strftime("%Y-%m-%d-%H-%M")  # forecast valid time, optional argument
npz_path = os.path.join(data_dir, f"all_latents_{timestamp}.npz")

# Two translator families:
#   "skip": per-token affine channel map on the layer's own grid, to the final latent state
#     (layer4 block3) for layer0/1/4 and to layer3 block11 for the coarse layer2/3.
#   "no_skip": for layer2/3 the translator also learns the 2x2 lat/lon upsampling to the final
#     (60, 120) grid. For layer0/1/4 (already at full resolution) it is identical to "skip".
translator_dir_skip = f"{config.TRANSLATOR_DIR}/archesweather/skip/weights/"
translator_dir_no_skip = f"{config.TRANSLATOR_DIR}/archesweather/no_skip/weights/"

KINDS = ("raw", "skip", "no_skip")
KIND_NAMES = {"raw": "untranslated", "skip": "translated_within_stage", "no_skip": "translated_to_final"}  # names in the paper

# Translated latents have 384 channels; raw layer0/1 latents have only 192, so raw panels
# for channels >= 192 are shown as "missing".
paper_channels = [100]  # channel shown in the supplement
level_indices = [2]  # latent level shown in the supplement
cmap = "RdBu_r"
cmap_missing = mpl.colormaps[cmap].copy()
cmap_missing.set_bad(color="lightgray")

# (lat, lon) latent grid per stage; layer2/3 are downsampled by a factor of 2
STAGE_GRID = {
    "layer0": (60, 120),
    "layer1": (60, 120),
    "layer2": (30, 60),
    "layer3": (30, 60),
    "layer4": (60, 120),
}
TARGET_GRID = (60, 120)  # grid of the "no_skip" output

# (layer, n_blocks) in forward-pass / network-depth order
STAGES = [
    ("layer0", 1),
    ("layer1", 4),
    ("layer2", 12),
    ("layer3", 12),
    ("layer4", 4),
]
EXPAND_LAYERS = {"layer2", "layer3"}  # layers where "skip" and "no_skip" differ

# geometry of the no_skip (expand) translator, see
# methods_without_data/translator_training/archesweather/train_no_skip_translators_layer2_and_3.py
PL, COARSE_LAT, COARSE_LON, EXPAND_OUT_DIM = 8, 30, 60, 384

# layer3 block11 is the skip translator target of layer2/3 and has no translator; the raw
# field is shown instead
SKIP_MISSING_BY_DESIGN = {("layer3", 11)}


def skip_translator_path(layer, block):
    return os.path.join(translator_dir_skip, f"Arches_translator_matrix_{layer}_block{block}.npz")


def no_skip_translator_path(layer, block):
    return os.path.join(translator_dir_no_skip, f"Arches_translator_rep_stab_matrix_{layer}_block{block}.npz")


def apply_affine(x_np, W, b):
    # W is stored as (out_features, in_features), as in torch.nn.Linear
    return x_np @ W.T + b


def apply_expand(x_np, W, b):
    # affine map followed by the 2x2 lat/lon upsampling, as ExpandTranslator.forward() in
    # train_no_skip_translators_layer2_and_3.py (without batch axis)
    x = x_np @ W.T + b  # (14400, 1536)
    x = x.reshape(PL, COARSE_LAT, COARSE_LON, 2, 2, EXPAND_OUT_DIM)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    x = x.reshape(PL * COARSE_LAT * 2 * COARSE_LON * 2, EXPAND_OUT_DIM)  # (57600, 384)
    return x


def reshape_to_field(x_np, lat, lon):
    """Reshape tokens of shape (N_tokens, D) to a field of shape (D, PL, lat, lon)."""
    D = x_np.shape[-1]
    return x_np.T.reshape(D, PL, lat, lon)

# %% LOAD THE SELECTED CHANNELS AND LEVELS FOR EVERY BLOCK
# raw, skip and no_skip fields are extracted block by block

missing_blocks = {"skip": set(), "no_skip": set()}  # (layer, block)

with np.load(npz_path, allow_pickle=False) as npz:
    latents_by_key = {}
    for k in npz.files:
        # rename e.g. 'layer1.blocks.0' to 'layer1_block0'
        new_key = re.sub(r"layer(\d+)\.blocks\.(\d+)", r"layer\1_block\2", k)
        latents_by_key[new_key] = npz[k]

    # panels_by_level[kind][level_idx][channel][layer]: list of (lat, lon) fields, one per block
    # (None for a channel out of range or a missing translator)
    panels_by_level = {
        kind: {lvl: {ch: {} for ch in paper_channels} for lvl in level_indices}
        for kind in KINDS
    }

    for layer, n_blocks in STAGES:
        lat, lon = STAGE_GRID[layer]
        is_expand_layer = layer in EXPAND_LAYERS

        fields = {
            kind: {lvl: {ch: [] for ch in paper_channels} for lvl in level_indices}
            for kind in KINDS
        }

        def extract(x_f32, D_avail, kind, lat, lon):
            field_full = reshape_to_field(x_f32, lat, lon) if D_avail else None
            for ch in paper_channels:
                if ch >= D_avail:
                    for lvl in level_indices:
                        fields[kind][lvl][ch].append(None)
                    continue
                for lvl in level_indices:
                    fields[kind][lvl][ch].append(field_full[ch, lvl])

        # the last block of layer4 is the final latent state and needs no translator
        last_layer4_block = n_blocks - 1 if layer == "layer4" else None

        for block in range(n_blocks):
            key = f"{layer}_block{block}"
            is_self_target = (layer == "layer4" and block == last_layer4_block)
            is_missing_by_design = (layer, block) in SKIP_MISSING_BY_DESIGN

            x_f32 = np.asarray(latents_by_key[key], dtype=np.float32)
            if x_f32.ndim == 3:  # drop batch axis
                x_f32 = x_f32[0]
            D_in = x_f32.shape[-1]

            extract(x_f32, D_in, "raw", lat, lon)

            # skip translator (on the layer's own grid)
            skip_path = skip_translator_path(layer, block)
            if is_self_target or is_missing_by_design:
                for lvl in level_indices:
                    for ch in paper_channels:
                        fields["skip"][lvl][ch].append(fields["raw"][lvl][ch][-1])
            elif not os.path.exists(skip_path):
                missing_blocks["skip"].add((layer, block))
                print(f"{layer} block {block}: no skip translator, panel left blank")
                for lvl in level_indices:
                    for ch in paper_channels:
                        fields["skip"][lvl][ch].append(None)
            else:
                with np.load(skip_path) as tr:
                    W_mat = tr["W"].astype(np.float32)
                    b_vec = tr["b"].astype(np.float32)
                x_skip = apply_affine(x_f32, W_mat, b_vec)
                extract(x_skip, x_skip.shape[-1], "skip", lat, lon)

            # no_skip translator (to the final (60, 120) grid)
            if is_self_target:
                for lvl in level_indices:
                    for ch in paper_channels:
                        fields["no_skip"][lvl][ch].append(fields["raw"][lvl][ch][-1])
            elif is_expand_layer:
                expand_path = no_skip_translator_path(layer, block)
                if not os.path.exists(expand_path):
                    missing_blocks["no_skip"].add((layer, block))
                    print(f"{layer} block {block}: no no_skip (expand) translator, panel left blank")
                    for lvl in level_indices:
                        for ch in paper_channels:
                            fields["no_skip"][lvl][ch].append(None)
                else:
                    with np.load(expand_path) as tr:
                        W_mat = tr["W"].astype(np.float32)
                        b_vec = tr["b"].astype(np.float32)
                    x_translated = apply_expand(x_f32, W_mat, b_vec)
                    extract(x_translated, x_translated.shape[-1], "no_skip", *TARGET_GRID)
            else:
                # layer0/1/4: identical to skip
                for lvl in level_indices:
                    for ch in paper_channels:
                        fields["no_skip"][lvl][ch].append(fields["skip"][lvl][ch][-1])

            print(f"{layer} block {block}: loaded {len(paper_channels)} channels x {len(level_indices)} levels (raw + skip + no_skip)")

        for kind in KINDS:
            for lvl in level_indices:
                for ch in paper_channels:
                    panels_by_level[kind][lvl][ch][layer] = fields[kind][lvl][ch]

for kind, missing in missing_blocks.items():
    if missing:
        print(f"Missing {kind} translators (panels will render blank): {sorted(missing)}")

# %% PLOT one figure each for raw, skip and no_skip per level and channel

n_rows = len(STAGES)
n_cols = max(n_blocks for _, n_blocks in STAGES)


def plot_one(kind, level_idx, channel):
    panels = panels_by_level[kind][level_idx][channel]

    fig = plt.figure(figsize=(2.5 * n_cols, 2.5 * n_rows))
    # one row per layer, one column per block; rows with fewer blocks end earlier
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        left=0.04, right=0.97, top=0.94, bottom=0.04,
        hspace=0.15, wspace=0.05,
    )

    for row, (layer, n_blocks) in enumerate(STAGES):
        row_fields = panels[layer]
        valid_fields = [f for f in row_fields if f is not None]
        # symmetric colour range per row (99th percentile of |value|)
        if valid_fields:
            row_values = np.concatenate([f.ravel() for f in valid_fields])
            row_vmax = np.nanpercentile(np.abs(row_values), 99)
            row_vmin = -row_vmax
        else:
            row_vmin, row_vmax = -1, 1

        # no_skip panels of layer2/3 are on TARGET_GRID, all others on the layer's own grid
        is_expand_layer = layer in EXPAND_LAYERS
        H, W = TARGET_GRID if (kind == "no_skip" and is_expand_layer) else STAGE_GRID[layer]

        row_axes = []
        im = None
        for col, field in enumerate(row_fields):
            ax = fig.add_subplot(gs[row, col])
            if field is None:
                ax.imshow(np.full((H, W), np.nan), cmap=cmap_missing, origin="upper",
                          vmin=row_vmin, vmax=row_vmax, aspect="auto")
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9,
                        transform=ax.transAxes, color="dimgray")
            else:
                im = ax.imshow(field, cmap=cmap, origin="upper", vmin=row_vmin, vmax=row_vmax, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"b{col}", fontsize=14)
            if col == 0:
                ax.set_ylabel(layer, fontsize=20)
            row_axes.append(ax)

        if im is not None:
            cbar = fig.colorbar(im, ax=row_axes, shrink=0.9, location="right", pad=0.01)
            cbar.ax.tick_params(labelsize=12)

    fig.suptitle(f"ArchesWeather latent channel {channel} (level {level_idx}) — {kind}", fontsize=26)

    save_path = os.path.join(config.SUPPLEMENT_DIR, f"latent_channel_arches_ch{channel}_lv{level_idx}_{KIND_NAMES[kind]}_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    reduce_png(save_path)  # 150 dpi, 256 colours
    plt.close(fig)
    print("Saved:", save_path)


for level_idx in level_indices:
    for channel in paper_channels:
        for kind in KINDS:
            plot_one(kind, level_idx, channel)
