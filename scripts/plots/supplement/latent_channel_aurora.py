import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% IMPORTS
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

from utils.reduce_png import reduce_png

# %% CONFIG

data_dir = f"{config.LATENT_DIR}/aurora/"
analysis_datetime = config.analysis_date()  # forecast valid time, optional argument
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")
timestamp = config.aurora_init_time(analysis_datetime).strftime("%Y-%m-%d-%H-%M-%S")
npz_path = os.path.join(data_dir, f"all_latents_aurora-ft_{timestamp}.npz")

# Two translator families (as in latent_channel_arches.py):
#   "skip": per-token affine channel map (translator_matrix_{part}_{stage}_{block}_gnn.npz) to
#     the last decoder state on the same grid (x_31, x_41 or the final state x_47).
#   "no_skip": for enc2/enc3/dec1/dec2 the translator also upsamples to the full-resolution
#     grid (translator_matrix_upscale_{stage}_{block}_gnn.npz, trained by
#     methods_without_data/translator_training/aurora/train_no_skip_translators.py).
#     For enc1/dec3 it is identical to "skip".
# The skip training scripts save files indexed by model_id (translator_matrix_{0..46}_gnn.npz);
# the provided weights use the names above.
translator_dir = f"{config.TRANSLATOR_DIR}/aurora/skip/weights"
no_skip_translator_dir = f"{config.TRANSLATOR_DIR}/aurora/no_skip/weights"

KINDS = ("raw", "skip", "no_skip")
KIND_NAMES = {"raw": "untranslated", "skip": "translated_within_stage", "no_skip": "translated_to_final"}  # names in the paper

paper_channels = [100]  # channel shown in the supplement
level_indices = [1]  # latent level shown in the supplement
cmap = "RdBu_r"
cmap_missing = mpl.colormaps[cmap].copy()
cmap_missing.set_bad(color="lightgray")

# (latent levels, H, W) per resolution stage of the Swin3D backbone; H and W halve at each
# of the two downsampling steps
STAGE_GRID = {
    0: (4, 180, 360),
    1: (4, 90, 180),
    2: (4, 45, 90),
}


def grid_for(part, stage):
    # encoder stages go from fine (0) to coarse (2), decoder stages from coarse (0) to fine (2)
    return STAGE_GRID[stage] if part == "enc" else STAGE_GRID[2 - stage]

# (part, stage index, display label) in forward-pass / network-depth order
STAGES = [
    ("enc", 0, "enc1"),
    ("enc", 1, "enc2"),
    ("enc", 2, "enc3"),
    ("dec", 0, "dec1"),
    ("dec", 1, "dec2"),
    ("dec", 2, "dec3"),
]

# full-resolution grid of the "no_skip" output of enc2/enc3/dec1/dec2
TARGET_GRID = STAGE_GRID[0]  # (4, 180, 360)
UPSCALE_OUT_DIM = 1024  # output channels of the upscale translators

UPSCALE_STAGE_PARAMS = {
    # (part, stage): (stage_name_in_filename, c_lat, h_src, w_src, block_factor)
    ("enc", 1): ("enc_1", 4, 90, 180, 2),
    ("enc", 2): ("enc_2", 4, 45, 90, 4),
    ("dec", 0): ("dec_0", 4, 45, 90, 4),
    ("dec", 1): ("dec_1", 4, 90, 180, 2),
}

# dec1 block7 (x_31) and dec2 block9 (x_41) are skip translator targets themselves and have
# no translator; the raw field is shown instead
SKIP_MISSING_BY_DESIGN = {("dec", 0, 7), ("dec", 1, 9)}


def skip_translator_path(part, stage, block):
    return os.path.join(translator_dir, f"translator_matrix_{part}_{stage}_{block}_gnn.npz")


def no_skip_translator_path(part, stage, block):
    stage_name = UPSCALE_STAGE_PARAMS[(part, stage)][0]
    return os.path.join(no_skip_translator_dir, f"translator_matrix_upscale_{stage_name}_{block}_gnn.npz")


def apply_plain(x_np, W, b):
    # W is stored as (in_dim, out_dim) by the skip training scripts
    return x_np @ W + b


def apply_upscale(x_np, W, b, c_lat, h_src, w_src, bf, out_dim):
    # W is stored as (out_dim, in_dim) by train_no_skip_translators.py; the reshape/transpose
    # matches UpscaleTranslator.forward() there (without batch axis)
    x = x_np @ W.T + b
    x = x.reshape(c_lat, h_src, w_src, bf, bf, out_dim)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    x = x.reshape(c_lat * h_src * bf * w_src * bf, out_dim)
    return x

# %% LOAD THE SELECTED CHANNELS AND LEVELS FOR EVERY BLOCK
# The npz file (~6 GB, 48 float16 blocks) is read block by block and the raw, skip and
# no_skip fields are extracted from each block. Channels beyond a block's channel count
# (e.g. >= 512 for raw enc1/dec3) and missing translators give a blank "missing" panel.

missing_blocks = {"skip": set(), "no_skip": set()}  # (part, stage, block)
stage_grid_by_label = {label: grid_for(part, stage) for part, stage, label in STAGES}

with np.load(npz_path, allow_pickle=False) as npz:
    parts = npz["part"]
    stages = npz["stage"]
    blocks = npz["block"]

    # record index in the npz per (part, stage) and block
    records_by_stage = {}
    for i in range(len(parts)):
        key = (str(parts[i]), int(stages[i]))
        records_by_stage.setdefault(key, {})[int(blocks[i])] = i

    # panels_by_level[kind][level_idx][channel][label]: list of (H, W) fields, one per block
    # (None for a channel out of range or a missing translator)
    panels_by_level = {
        kind: {lvl: {ch: {} for ch in paper_channels} for lvl in level_indices}
        for kind in KINDS
    }

    for part, stage, label in STAGES:
        C, H, W = grid_for(part, stage)
        is_upscale_stage = (part, stage) in UPSCALE_STAGE_PARAMS
        block_to_record = records_by_stage[(part, stage)]

        fields = {
            kind: {lvl: {ch: [] for ch in paper_channels} for lvl in level_indices}
            for kind in KINDS
        }

        def extract(x_f32, D_avail, kind, c, h, w):
            for ch in paper_channels:
                if ch >= D_avail:
                    for lvl in level_indices:
                        fields[kind][lvl][ch].append(None)
                    continue
                field = x_f32[:, ch].reshape(c, h, w)
                field = np.flip(field, axis=1)              # fix latitude orientation
                field = np.roll(field, shift=w // 2, axis=2)  # fix longitude phase
                for lvl in level_indices:
                    fields[kind][lvl][ch].append(field[lvl])

        # the last block of decoder stage 2 (x_47) is the final latent state and needs no translator
        last_dec_block = max(block_to_record) if (part == "dec" and stage == 2) else None

        for block in sorted(block_to_record):
            i = block_to_record[block]
            is_self_target = (part == "dec" and block == last_dec_block)
            is_missing_by_design = (part, stage, block) in SKIP_MISSING_BY_DESIGN

            x = npz[f"x_{i}"]  # (1, C*H*W, D_in), float16
            x_f32 = x[0].astype(np.float32)
            D_in = x_f32.shape[-1]

            extract(x_f32, D_in, "raw", C, H, W)

            # skip translator (on the block's own grid)
            skip_path = skip_translator_path(part, stage, block)
            if is_self_target or is_missing_by_design:
                for lvl in level_indices:
                    for ch in paper_channels:
                        fields["skip"][lvl][ch].append(fields["raw"][lvl][ch][-1])
            elif not os.path.exists(skip_path):
                missing_blocks["skip"].add((part, stage, block))
                print(f"{label} block {block}: (x_{i}) no skip translator, panel left blank")
                for lvl in level_indices:
                    for ch in paper_channels:
                        fields["skip"][lvl][ch].append(None)
            else:
                with np.load(skip_path) as tr:
                    W_mat = tr["W"].astype(np.float32)
                    b_vec = tr["b"].astype(np.float32)
                x_skip = apply_plain(x_f32, W_mat, b_vec)
                extract(x_skip, x_skip.shape[-1], "skip", C, H, W)

            # no_skip translator (to the full-resolution grid)
            if is_self_target:
                for lvl in level_indices:
                    for ch in paper_channels:
                        fields["no_skip"][lvl][ch].append(fields["raw"][lvl][ch][-1])
            elif is_upscale_stage:
                upscale_path = no_skip_translator_path(part, stage, block)
                if not os.path.exists(upscale_path):
                    missing_blocks["no_skip"].add((part, stage, block))
                    print(f"{label} block {block}: (x_{i}) no no_skip (upscale) translator, panel left blank")
                    for lvl in level_indices:
                        for ch in paper_channels:
                            fields["no_skip"][lvl][ch].append(None)
                else:
                    _, c_lat, h_src, w_src, bf = UPSCALE_STAGE_PARAMS[(part, stage)]
                    with np.load(upscale_path) as tr:
                        W_mat = tr["W"].astype(np.float32)
                        b_vec = tr["b"].astype(np.float32)
                    x_translated = apply_upscale(x_f32, W_mat, b_vec, c_lat, h_src, w_src, bf, UPSCALE_OUT_DIM)
                    extract(x_translated, UPSCALE_OUT_DIM, "no_skip", *TARGET_GRID)
            else:
                # enc1/dec3: identical to skip
                for lvl in level_indices:
                    for ch in paper_channels:
                        fields["no_skip"][lvl][ch].append(fields["skip"][lvl][ch][-1])

            print(f"{label} block {block}: (x_{i}) loaded {len(paper_channels)} channels x {len(level_indices)} levels (raw + skip + no_skip)")

        for kind in KINDS:
            for lvl in level_indices:
                for ch in paper_channels:
                    panels_by_level[kind][lvl][ch][label] = fields[kind][lvl][ch]

for kind, missing in missing_blocks.items():
    if missing:
        print(f"Missing {kind} translators (panels will render blank): {sorted(missing)}")

# %% PLOT one figure each for raw, skip and no_skip per level and channel

n_rows = len(STAGES)
n_cols = max(
    len(panels_by_level[KINDS[0]][level_indices[0]][paper_channels[0]][label]) for _, _, label in STAGES
)


def plot_one(kind, level_idx, channel):
    panels = panels_by_level[kind][level_idx][channel]

    fig = plt.figure(figsize=(2.5 * n_cols, 2.5 * n_rows))
    # one row per stage, one column per block; rows with fewer blocks end earlier
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        left=0.04, right=0.97, top=0.94, bottom=0.04,
        hspace=0.15, wspace=0.05,
    )

    for row, (part, stage, label) in enumerate(STAGES):
        row_fields = panels[label]
        valid_fields = [f for f in row_fields if f is not None]
        # symmetric colour range per row (99th percentile of |value|)
        if valid_fields:
            row_values = np.concatenate([f.ravel() for f in valid_fields])
            row_vmax = np.nanpercentile(np.abs(row_values), 99)
            row_vmin = -row_vmax
        else:
            row_vmin, row_vmax = -1, 1

        # no_skip panels of the upscale stages are on TARGET_GRID, all others on the stage's own grid
        is_upscale_stage = (part, stage) in UPSCALE_STAGE_PARAMS
        if kind == "no_skip" and is_upscale_stage:
            _, H, W = TARGET_GRID
        else:
            _, H, W = stage_grid_by_label[label]

        row_axes = []
        im = None
        for col, field in enumerate(row_fields):
            ax = fig.add_subplot(gs[row, col])
            if field is None:
                ax.imshow(np.full((H, W), np.nan), cmap=cmap_missing, origin="lower",
                          vmin=row_vmin, vmax=row_vmax, aspect="auto")
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9,
                        transform=ax.transAxes, color="dimgray")
            else:
                im = ax.imshow(field, cmap=cmap, origin="lower", vmin=row_vmin, vmax=row_vmax, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"b{col}", fontsize=14)
            if col == 0:
                ax.set_ylabel(label, fontsize=20)
            row_axes.append(ax)

        if im is not None:
            cbar = fig.colorbar(im, ax=row_axes, shrink=0.9, location="right", pad=0.01)
            cbar.ax.tick_params(labelsize=12)

    fig.suptitle(f"Aurora latent channel {channel} (level {level_idx}) — {kind}", fontsize=26)

    save_path = os.path.join(config.SUPPLEMENT_DIR, f"latent_channel_aurora_ch{channel}_lv{level_idx}_{KIND_NAMES[kind]}_{time_str}.png")
    plt.savefig(save_path, dpi=300)
    reduce_png(save_path)  # 150 dpi, 256 colours
    plt.close(fig)
    print("Saved:", save_path)


for level_idx in level_indices:
    for channel in paper_channels:
        for kind in KINDS:
            plot_one(kind, level_idx, channel)
