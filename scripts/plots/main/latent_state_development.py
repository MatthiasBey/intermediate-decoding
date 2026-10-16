"""One latent channel through the network of GraphCast, Aurora and ArchesWeather, raw and after
the translator, for one forecast. One section per model with a row of raw and a row of
translated latent states; one symmetric colour range per model (99th percentile of the raw
|values|, since the latent channels are heavy-tailed)."""
import os
import re
import sys

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from graphcast import icosahedral_mesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config

FONT_SIZE = 30  # all labels, colorbar ticks and section titles
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
})
CMAP = "RdBu_r"
CBAR_PERCENTILE = 99

GC_CHANNEL = 95
GC_STEPS = [0, 2, 4, 6, 8, 10, 12, 14, 16]

AURORA_CHANNEL = 60
AURORA_LEVEL = 1  # 0-3 (Aurora has 4 latent levels, ArchesWeather 8)
AURORA_GRID = (4, 180, 360)  # (levels, lat, lon) of the first encoder and the last decoder stage
# encoder output and blocks of the first encoder stage (enc1) and the last decoder stage (dec3),
# the two stages connected by Aurora's encoder-decoder skip connection
AURORA_BLOCKS = ["encoder_output"] + [f"enc1_block{b}" for b in (0, 2, 4, 5)] + [f"dec3_block{b}" for b in (0, 2, 4, 5)]
AURORA_LAST_STATE = "dec3_block5"  # needs no translator

ARCHES_CHANNEL = 243
ARCHES_LEVEL = 4
# layers 0, 1 and 4; the last state (layer 4, block 3) needs no translator
ARCHES_BLOCKS = ["layer0_block0", "layer1_block0", "layer1_block1", "layer1_block2", "layer1_block3",
                 "layer4_block0", "layer4_block1", "layer4_block2", "layer4_block3"]
ARCHES_LAST_STATE = "layer4_block3"

analysis_datetime = config.analysis_date()  # forecast valid time, optional argument
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")
print(f"Analysis date: {time_str}")


def state_label(index):
    return rf"$\tilde{{X}}_{{{index}}}$"


def symmetric_limit(fields):
    """CBAR_PERCENTILE-th percentile of |value| over all fields."""
    values = np.abs(np.concatenate([np.asarray(f, dtype=np.float64).ravel() for f in fields]))
    return float(np.nanpercentile(values, CBAR_PERCENTILE))


# %% GraphCast: latent states on the icosahedral mesh (finest level, 40962 nodes), translated
# in bfloat16 as in the model
with np.load(f"{config.LATENT_DIR}/graphcast/all_latents_{time_str}.npz") as data:
    gc_latents = dict(data)

gc_raw, gc_translated = [], []
for step in GC_STEPS:
    latent = gc_latents[f"step_{step}"]  # (nodes, 1, channels)
    if step == 16:  # last state
        translated = latent
    else:
        with np.load(f"{config.TRANSLATOR_DIR}/graphcast/weights/translator_matrix_{step}_gnn.npz") as tr:
            W = jnp.array(tr["W"], dtype=jnp.bfloat16)  # (in_dim, out_dim)
            b = jnp.array(tr["b"], dtype=jnp.bfloat16)
        translated = jnp.einsum("nbd,df->nbf", jnp.array(latent, dtype=jnp.bfloat16), W) + b
    gc_raw.append(np.asarray(latent).squeeze(1)[:, GC_CHANNEL])
    gc_translated.append(np.asarray(translated).squeeze(1)[:, GC_CHANNEL])

xyz = icosahedral_mesh.get_last_triangular_mesh_for_sphere(splits=6).vertices
mesh_lat = np.degrees(np.arcsin(xyz[:, 2]))
mesh_lon = ((np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) + 180) % 360) - 180

# %% Aurora
init_str = config.aurora_init_time(analysis_datetime).strftime("%Y-%m-%d-%H-%M-%S")
aurora_translator_dir = f"{config.TRANSLATOR_DIR}/aurora/skip/weights"


def aurora_field(x):
    """(lat, lon) field of AURORA_CHANNEL at AURORA_LEVEL from tokens x (C*H*W, D), or None if
    x has fewer channels."""
    if AURORA_CHANNEL >= x.shape[-1]:
        return None
    c, h, w = AURORA_GRID
    field = x[:, AURORA_CHANNEL].reshape(c, h, w)
    field = np.flip(field, axis=1)               # latitude orientation
    field = np.roll(field, shift=w // 2, axis=2)  # longitude phase
    return field[AURORA_LEVEL]


aurora_raw, aurora_translated = [], []
with np.load(f"{config.LATENT_DIR}/aurora/all_latents_aurora-ft_{init_str}.npz", allow_pickle=False) as npz:
    record_index = {(str(part), int(stage), int(block)): i
                    for i, (part, stage, block) in enumerate(zip(npz["part"], npz["stage"], npz["block"]))}
    for key in AURORA_BLOCKS:
        if key == "encoder_output":
            with np.load(f"{config.LATENT_DIR}/aurora/encoder_output/all_latents_aurora-ft_{init_str}.npz",
                         allow_pickle=False) as enc:
                x = enc["encoder_x"][0].astype(np.float32)
            translator_path = os.path.join(aurora_translator_dir, "translator_matrix_encoder_output_gnn.npz")
        else:
            stage_name, block = key.split("_block")
            part, stage = ("enc", 0) if stage_name == "enc1" else ("dec", 2)
            x = npz[f"x_{record_index[(part, stage, int(block))]}"][0].astype(np.float32)
            translator_path = os.path.join(aurora_translator_dir, f"translator_matrix_{part}_{stage}_{block}_gnn.npz")

        aurora_raw.append(aurora_field(x))
        if key == AURORA_LAST_STATE and not os.path.exists(translator_path):
            aurora_translated.append(aurora_raw[-1])
        elif not os.path.exists(translator_path):
            print(f"Aurora {key}: no translator, panel left blank")
            aurora_translated.append(None)
        else:
            with np.load(translator_path) as tr:
                W = tr["W"].astype(np.float32)  # (in_dim, out_dim), unlike the ArchesWeather translators
                b = tr["b"].astype(np.float32)
            aurora_translated.append(aurora_field(x @ W + b))


def aurora_label(key):
    """Position of the state in the whole network: encoder_output 0, then enc1 1-6, enc2 7-16,
    enc3 17-24, dec1 25-32, dec2 33-42, dec3 43-48."""
    if key == "encoder_output":
        return state_label(0)
    stage_name, block = key.split("_block")
    offsets = {"enc1": 1, "enc2": 7, "enc3": 17, "dec1": 25, "dec2": 33, "dec3": 43}
    return state_label(offsets[stage_name] + int(block))


# %% ArchesWeather: tokens (levels * lat * lon, channels) on the 60 x 120 grid of layers 0, 1 and 4
arches_latents = {}
with np.load(f"{config.LATENT_DIR}/archesweather/all_latents_{time_str}.npz") as data:
    for key, value in data.items():
        # 'layer1.blocks.0' becomes 'layer1_block0'
        arches_latents[re.sub(r'layer(\d+)\.blocks\.(\d+)', r'layer\1_block\2', key)] = value


def arches_field(latent, channel):
    x = np.asarray(latent).astype(np.float32)
    if x.ndim == 3:  # drop batch axis
        x = x[0]
    return x.T.reshape(x.shape[1], 8, 60, 120)[channel, ARCHES_LEVEL]


arches_raw, arches_translated = [], []
for col, key in enumerate(ARCHES_BLOCKS):
    latent = arches_latents[key]
    # the raw latents of layers 0 and 1 have only 192 channels
    arches_raw.append(arches_field(latent, ARCHES_CHANNEL - 192 if col < 5 else ARCHES_CHANNEL))
    if key == ARCHES_LAST_STATE:
        translated = latent
    else:
        with np.load(f"{config.TRANSLATOR_DIR}/archesweather/skip/weights/Arches_translator_matrix_{key}.npz") as tr:
            W = tr["W"].astype(np.float32).T  # stored as (out_features, in_features)
            b = tr["b"].astype(np.float32)
        translated = np.dot(latent, W) + b
    arches_translated.append(arches_field(translated, ARCHES_CHANNEL))

# %% Figure
gc_vmax = symmetric_limit(gc_raw)
aurora_vmax = symmetric_limit([f for f in aurora_raw if f is not None])
arches_vmax = symmetric_limit(arches_raw)
print(f"Colour limits ({CBAR_PERCENTILE}th pct of raw |value|): GraphCast +-{gc_vmax:.3f}, "
      f"Aurora +-{aurora_vmax:.3f}, Arches +-{arches_vmax:.3f}")

cmap_missing = mpl.colormaps[CMAP].copy()
cmap_missing.set_bad(color="lightgray")

n_cols = max(len(GC_STEPS), len(AURORA_BLOCKS), len(ARCHES_BLOCKS))
# 9 rows: title, GraphCast (raw, translated), title, Aurora (raw, translated), title, ArchesWeather
fig = plt.figure(figsize=(3 * n_cols, 2.5 * 9))
gs = gridspec.GridSpec(9, n_cols, figure=fig, height_ratios=[0.3, 1, 1, 0.55, 1, 1, 0.55, 1, 1],
                       hspace=0.05, wspace=0.05)


def section_title(row, text):
    """Title at the bottom of an invisible axis spanning the spacer row."""
    ax = fig.add_subplot(gs[row, :])
    ax.axis('off')
    ax.annotate(text, xy=(0.5, 0), xycoords='axes fraction', xytext=(0, 0), textcoords='offset points',
                ha='center', va='bottom', fontsize=FONT_SIZE)


def image_panel(ax, field, vmax, origin):
    """imshow of a field, or a grey panel saying "missing"; returns the image or None."""
    if field is None:
        ax.imshow(np.full(AURORA_GRID[1:], np.nan), cmap=cmap_missing, origin=origin, aspect="auto")
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=FONT_SIZE, transform=ax.transAxes,
                color="dimgray")
        return None
    return ax.imshow(field, cmap=CMAP, origin=origin, aspect="auto", vmin=-vmax, vmax=vmax)


def label_panel(ax, col, row_label, xlabel=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if col == 0:
        ax.set_ylabel(row_label, fontsize=FONT_SIZE)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE)


def add_colorbar(image, axes):
    return fig.colorbar(image, ax=axes, shrink=0.9, location='right', pad=0.01, fraction=0.02, extend='both')


# GraphCast (rows 1 and 2)
section_title(0, f"GraphCast latent channel {GC_CHANNEL}")
axes_gc = []
for col, step in enumerate(GC_STEPS):
    ax = fig.add_subplot(gs[1, col])
    im_gc = ax.scatter(mesh_lon, mesh_lat, c=gc_raw[col], s=4, cmap=CMAP, vmin=-gc_vmax, vmax=gc_vmax)
    label_panel(ax, col, "raw")
    ax2 = fig.add_subplot(gs[2, col])
    ax2.scatter(mesh_lon, mesh_lat, c=gc_translated[col], s=4, cmap=CMAP, vmin=-gc_vmax, vmax=gc_vmax)
    label_panel(ax2, col, "translated", state_label(step))
    axes_gc += [ax, ax2]

# Aurora (rows 4 and 5)
section_title(3, f"Aurora latent channel {AURORA_CHANNEL}")
axes_aurora = []
im_aurora = None
for col, key in enumerate(AURORA_BLOCKS):
    ax = fig.add_subplot(gs[4, col])
    image = image_panel(ax, aurora_raw[col], aurora_vmax, "lower")
    if image is not None:
        im_aurora = image
    label_panel(ax, col, "raw")
    ax2 = fig.add_subplot(gs[5, col])
    image_panel(ax2, aurora_translated[col], aurora_vmax, "lower")
    label_panel(ax2, col, "translated", aurora_label(key))
    axes_aurora += [ax, ax2]

# ArchesWeather (rows 7 and 8); the states are numbered over all blocks (layer 4 starts at 29)
section_title(6, f"ArchesWeather latent channel {ARCHES_CHANNEL}")
axes_arches = []
for col in range(len(ARCHES_BLOCKS)):
    ax = fig.add_subplot(gs[7, col])
    im_arches = image_panel(ax, arches_raw[col], arches_vmax, "upper")
    label_panel(ax, col, "raw")
    ax2 = fig.add_subplot(gs[8, col])
    image_panel(ax2, arches_translated[col], arches_vmax, "upper")
    label_panel(ax2, col, "translated", state_label(col if col < 5 else col + 24))
    axes_arches += [ax, ax2]

# colorbars with ticks at -vmax, 0, +vmax; the arrows mark the values beyond the limit
colorbars = [(add_colorbar(im_gc, axes_gc), gc_vmax)]
if im_aurora is not None:
    colorbars.append((add_colorbar(im_aurora, axes_aurora), aurora_vmax))
colorbars.append((add_colorbar(im_arches, axes_arches), arches_vmax))
for cbar, vmax in colorbars:
    cbar.set_ticks([-vmax, 0, vmax], labels=[f"\N{MINUS SIGN}{vmax:.2f}", "0", f"{vmax:.2f}"])
    cbar.ax.tick_params(labelsize=FONT_SIZE)

plt.tight_layout(rect=[0, 0, 0.99, 0.96])
save_path = os.path.join(config.FIGURES_DIR, f"latent_state_development_{time_str}.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print("Saved:", save_path)
