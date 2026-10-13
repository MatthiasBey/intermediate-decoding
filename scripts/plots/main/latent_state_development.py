import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% Imports
import re
import torch
import jax.numpy as jnp

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from graphcast import icosahedral_mesh

# =============================================================================
# Figure style (all font sizes are set here)
# =============================================================================
FONT_FAMILY = "sans-serif"             # e.g. "sans-serif", "serif", "monospace"
FONT_NAME = ["DejaVu Sans"]            # preferred fonts in order, e.g. ["Arial", "Helvetica", "DejaVu Sans"]

FONT_SIZE_LABEL = 30                  # all labels
FONT_SIZE_ROW_LABEL = FONT_SIZE_LABEL       # "raw" / "translated" left of each row
FONT_SIZE_H_LABEL = FONT_SIZE_LABEL         # latent state label under each panel
FONT_SIZE_CBAR_TICK = FONT_SIZE_LABEL       # colorbar tick numbers
FONT_SIZE_MISSING = FONT_SIZE_LABEL         # "missing" text in blank panels
FONT_SIZE_TITLE = 30                  # section titles (one per model)
TITLE_PAD = 0                          # gap between section title and the plot row below it (points)

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    f"font.{FONT_FAMILY}": FONT_NAME,
    "font.size": FONT_SIZE_LABEL,
    "axes.labelsize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_CBAR_TICK,
    "ytick.labelsize": FONT_SIZE_CBAR_TICK,
})


# %% Paths
gc_version = "std"
path_gc_latent_data = f'{config.LATENT_DIR}/graphcast/'

patch_arches = f"{config.LATENT_DIR}/archesweather/"

path_plots = config.FIGURES_DIR

# translators
path_translators_gc = f'{config.TRANSLATOR_DIR}/graphcast/weights/'

path_translators_arches = f"{config.TRANSLATOR_DIR}/archesweather/skip/weights/"

# %% Analysis date
analysis_datetime = config.analysis_date()  # forecast valid time, optional argument
time_str = analysis_datetime.strftime("%Y-%m-%d-%H-%M")
print(f"Analysis date: {time_str}")

# %% Load latent states and translators of GraphCast and ArchesWeather

npz_path = os.path.join(path_gc_latent_data, f"all_latents_{time_str}.npz")
with np.load(npz_path) as data:
    latent_data_gc = dict(data)

latent_data_arches = {}
npz_path_arches = os.path.join(patch_arches, f"all_latents_{time_str}.npz")
with np.load(npz_path_arches) as data:
    for k, v in data.items():
        # 'layer1.blocks.0' becomes 'layer1_block0'; 'layer0_block0' stays unchanged
        new_key = re.sub(r'layer(\d+)\.blocks\.(\d+)', r'layer\1_block\2', k)
        latent_data_arches[new_key] = v
    print(f"Loaded Arches latent data with keys: {list(latent_data_arches.keys())}")

# translators for GraphCast (all 16 steps) and ArchesWeather
translator_gc = {}
for step in range(16):
    translator_path = os.path.join(path_translators_gc, f"translator_matrix_{step}_gnn.npz")
    with np.load(translator_path) as data:
        translator_gc[f"gnn_{step}"] = dict(data)

# ArchesWeather: only layers 0, 1 and 4 are plotted
translator_arches = {}
for layer in [0, 1, 4]:
    for block in range({0:1, 1:4, 4:3}[layer]):
        translator_path = os.path.join(
            path_translators_arches,
            f"Arches_translator_matrix_layer{layer}_block{block}.npz"
        )
        with np.load(translator_path) as data:
            translator_arches[f"layer{layer}_block{block}"] = dict(data)

# %% Translate the ArchesWeather latent states (layers 0, 1 and 4)
translated_latents_arches = {}

for layer in [0, 1, 4]:
    for block in range({0:1, 1:4, 4:3}[layer]):
        latent_key = f"layer{layer}_block{block}"
        print(f"🔹 Translating Arches latents for {latent_key}...")
        translator = translator_arches[latent_key]
        W = torch.tensor(translator["W"], dtype=torch.float32).T  # transpose to match (n_channels, n_out)
        b = torch.tensor(translator["b"], dtype=torch.float32)
        translated_latents_arches[latent_key] = np.dot(latent_data_arches[latent_key], W.numpy()) + b.numpy()

# the last state (layer 4, block 3) needs no translator
translated_latents_arches['layer4_block3'] = latent_data_arches["layer4_block3"]

# %% Select the same blocks of the raw latent states

latent_data_arches_selected = {}
for layer in [0, 1, 4]:
    for block in range({0:1, 1:4, 4:3}[layer]):
        latent_key = f"layer{layer}_block{block}"
        latent_data_arches_selected[latent_key] = latent_data_arches[latent_key]

latent_data_arches_selected['layer4_block3'] = latent_data_arches["layer4_block3"]

# %% Translate the GraphCast latent states (all 16 GNN steps)
translated_latents_gc = {}
torch.set_default_device("cpu")
for step in range(16):
    latent_key = f"step_{step}"
    print(f"🔹 Translating GraphCast latents for {gc_version} {latent_key}...")
    translator = translator_gc[f"gnn_{step}"]
    W = jnp.array(translator["W"], dtype=jnp.bfloat16)  # shape: [in_dim, out_dim]
    b = jnp.array(translator["b"], dtype=jnp.bfloat16)  # shape: [out_dim]
    latent = jnp.array(latent_data_gc[f"step_{step}"], dtype=jnp.bfloat16)  # shape: [n_samples, n_nodes, in_dim]
    translated = jnp.einsum("nbd,df->nbf", latent, W) + b
    translated_latents_gc[latent_key] = translated
    print("GraphCast translation:", type(translated), getattr(translated, 'shape', 'no shape'))
    sys.stdout.flush()

# the last state (step 16) needs no translator
translated_latents_gc['step_16'] = latent_data_gc['step_16']

# %% Select every second GraphCast step
selected_steps = [0, 2, 4, 6, 8, 10, 12, 14, 16]
translated_latents_gc_selected = {
    f"step_{step}": translated_latents_gc[f"step_{step}"]
    for step in selected_steps
}

latents_gc_selected = {
    f"step_{step}": latent_data_gc[f"step_{step}"]
    for step in selected_steps
}

# %% Prepare the data for plotting

def transform_input(output, lon_resolution=240, lat_resolution=120, patch_size=(2, 2, 2), middle_layers=False):
    """Reshape ArchesWeather tokens (batch, tokens, channels) to (batch, channels, 8, lat, lon).

    A missing batch dimension is added. `middle_layers=True` uses the coarser (30, 60)
    grid of layers 2 and 3.
    """

    if middle_layers:
        patch_size = (2, 4, 4)

    if not isinstance(output, torch.Tensor):
        output = torch.tensor(output)

    if output.ndim == 2:
        output = output.unsqueeze(0)

    layer1_shape = (
        lat_resolution // patch_size[1],
        lon_resolution // patch_size[2]
    )

    if output.ndim < 3:
        raise ValueError(f"Expected at least 3D tensor, got {output.shape}")

    output = output.transpose(1, 2).reshape(output.shape[0], -1, 8, *layer1_shape)

    return output


def to_numpy_latent(latent):
    """Convert a torch, jax or numpy array to a float32 numpy array."""
    if isinstance(latent, torch.Tensor):
        arr = latent.detach().cpu().numpy()
    elif hasattr(latent, "numpy"):  # jax or numpy
        arr = np.asarray(latent)
    else:
        arr = np.array(latent)
    return arr.astype(np.float32)


# GraphCast icosahedral mesh (finest level, 40962 vertices)
mesh = icosahedral_mesh.get_last_triangular_mesh_for_sphere(splits=6)
vertices = mesh.vertices   # (40962, 3)

# mesh vertex coordinates (x, y, z) to lat/lon

def xyz_to_latlon(xyz):

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))

    lon = ((lon + 180) % 360) - 180

    return lat, lon


lat, lon = xyz_to_latlon(vertices)

# %% ArchesWeather: translated and raw latent states as DataArrays (block, channel, plevel, lat, lon)

arches_blocks = [
    "layer0_block0",
    "layer1_block0",
    "layer1_block1",
    "layer1_block2",
    "layer1_block3",
    "layer4_block0",
    "layer4_block1",
    "layer4_block2",
    "layer4_block3"
]

def build_arches_dataset(latent_dict, arches_blocks, middle_layers=False):
    data_list = []

    for block_id, key in enumerate(arches_blocks):

        print(f"Processing {key}")

        latent = latent_dict[key]
        latent_np = to_numpy_latent(latent)

        transformed = transform_input(latent_np, middle_layers=middle_layers)

        latent_np_final = (
            transformed
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            if isinstance(transformed, torch.Tensor)
            else np.asarray(transformed).astype(np.float32)
        )

        da_block = xr.DataArray(
            latent_np_final,
            dims=["channel", "plevel", "lat", "lon"],
            coords={
                "block": block_id,
                "channel": np.arange(latent_np_final.shape[0]),
                "plevel": np.arange(8),
                "lat": np.linspace(-90, 90, latent_np_final.shape[2]),
                "lon": np.linspace(0, 360, latent_np_final.shape[3], endpoint=False),
            }
        )

        data_list.append(da_block)

    return xr.concat(data_list, dim="block")

arches_dataset = build_arches_dataset(translated_latents_arches, arches_blocks)
arches_dataset_untranslated = build_arches_dataset(latent_data_arches_selected, arches_blocks)

# %% Settings and colour limits for GraphCast and ArchesWeather
gc_channel = 95
gc_steps = [0, 2, 4, 6, 8, 10, 12, 14, 16]

arch_channel = 243
plevel_idx = 4
n_arch = arches_dataset.sizes["block"]

# One symmetric colour scale per model for all its panels: the CBAR_PERCENTILE-th
# percentile of |value| over the raw panels. A percentile is used because the latent
# channels are heavy-tailed.
CBAR_PERCENTILE = 99


def symmetric_percentile_limit(arrays, q=CBAR_PERCENTILE):
    a = np.abs(np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in arrays]))
    return float(np.nanpercentile(a, q))


def label_limit_colorbar(cbar, vmax):
    """Ticks at -vmax, 0, +vmax, the ends labelled with the actual (percentile) limit."""
    cbar.set_ticks([-vmax, 0, vmax], labels=[f"\N{MINUS SIGN}{vmax:.2f}", "0", f"{vmax:.2f}"])
    cbar.ax.tick_params(labelsize=FONT_SIZE_CBAR_TICK)


def arch_panel(dataset, col, untranslated):
    """(lat, lon) field of one ArchesWeather panel, same selection as in the figure."""
    channel = arch_channel - 192 if (untranslated and col in [0, 1, 2, 3, 4]) else arch_channel
    return dataset.isel(block=col, channel=channel, plevel=plevel_idx).values


gc_raw_panels = [np.asarray(latents_gc_selected[f"step_{step}"]).squeeze(1)[:, gc_channel] for step in gc_steps]
arch_raw_panels = [arch_panel(arches_dataset_untranslated, col, True) for col in range(n_arch)]

gc_vmax = symmetric_percentile_limit(gc_raw_panels)
arch_vmax = symmetric_percentile_limit(arch_raw_panels)
gc_vmin, arch_vmin = -gc_vmax, -arch_vmax
print(f"Colour limits ({CBAR_PERCENTILE}th pct of raw |value|): GraphCast +-{gc_vmax:.3f}, Arches +-{arch_vmax:.3f}")

cmap_gc = "RdBu_r"
cmap_arch = "RdBu_r"

# %% Aurora: encoder output and blocks of the first encoder stage (enc1) and the last
# decoder stage (dec3), the two stages connected by Aurora's encoder-decoder skip connection
aurora_data_dir = f"{config.LATENT_DIR}/aurora/"
aurora_encoder_dir = f"{config.LATENT_DIR}/aurora/encoder_output"
aurora_translator_dir = f"{config.TRANSLATOR_DIR}/aurora/skip/weights"
aurora_timestamp = config.aurora_init_time(analysis_datetime).strftime("%Y-%m-%d-%H-%M-%S")
aurora_npz_path = os.path.join(aurora_data_dir, f"all_latents_aurora-ft_{aurora_timestamp}.npz")
aurora_encoder_path = os.path.join(aurora_encoder_dir, f"all_latents_aurora-ft_{aurora_timestamp}.npz")

aurora_channel = 60
aurora_plevel_idx = 1  # 0-3 (Aurora has 4 latent levels, ArchesWeather 8)

# enc1 and dec3 are on Aurora's finest latent grid (C, H, W) = (4, 180, 360)
AURORA_C, AURORA_H, AURORA_W = 4, 180, 360

aurora_blocks = (
    ["encoder_output"] +
    [f"enc1_block{b}" for b in (0, 2, 4, 5)] +
    [f"dec3_block{b}" for b in (0, 2, 4, 5)]
)
n_aurora = len(aurora_blocks)
aurora_missing_translators = set()

# last latent state of the model, which needs no translator
AURORA_LAST_STATE_KEY = "dec3_block5"


def aurora_h_label(key):
    """Label with the position of the state in the whole network: encoder_output 0,
    then enc1 1-6, enc2 7-16, enc3 17-24, dec1 25-32, dec2 33-42, dec3 43-48."""
    if key == "encoder_output":
        return r"$\tilde{X}_{0}$"
    stage_name, block_str = key.split("_block")
    offsets = {"enc1": 1, "enc2": 7, "enc3": 17, "dec1": 25, "dec2": 33, "dec3": 43}
    return rf"$\tilde{{X}}_{{{offsets[stage_name] + int(block_str)}}}$"


def _aurora_extract(x_f32, D_avail, channel, C, H, W, level_idx):
    """x_f32: (C*H*W, D_avail). Returns the (H, W) field at `channel` and `level_idx`,
    or None if the channel does not exist."""
    if channel >= D_avail:
        return None
    field = x_f32[:, channel].reshape(C, H, W)
    field = np.flip(field, axis=1)               # fix latitude orientation
    field = np.roll(field, shift=W // 2, axis=2)  # fix longitude phase
    return field[level_idx]


aurora_untranslated = {}
aurora_translated = {}

with np.load(aurora_npz_path, allow_pickle=False) as anpz:
    a_parts = anpz["part"]
    a_stages = anpz["stage"]
    a_blocks = anpz["block"]
    a_record_index = {}
    for i in range(len(a_parts)):
        a_record_index[(str(a_parts[i]), int(a_stages[i]), int(a_blocks[i]))] = i

    for key in aurora_blocks:
        if key == "encoder_output":
            with np.load(aurora_encoder_path, allow_pickle=False) as aenc:
                x_f32 = aenc["encoder_x"][0].astype(np.float32)
            translator_path = os.path.join(aurora_translator_dir, "translator_matrix_encoder_output_gnn.npz")
        else:
            stage_name, block_str = key.split("_block")
            part, stage = ("enc", 0) if stage_name == "enc1" else ("dec", 2)
            block = int(block_str)
            i = a_record_index[(part, stage, block)]
            x_f32 = anpz[f"x_{i}"][0].astype(np.float32)
            translator_path = os.path.join(aurora_translator_dir, f"translator_matrix_{part}_{stage}_{block}_gnn.npz")

        D_in = x_f32.shape[-1]
        aurora_untranslated[key] = _aurora_extract(
            x_f32, D_in, aurora_channel, AURORA_C, AURORA_H, AURORA_W, aurora_plevel_idx
        )

        if not os.path.exists(translator_path):
            if key == AURORA_LAST_STATE_KEY:
                aurora_translated[key] = aurora_untranslated[key]
            else:
                aurora_missing_translators.add(key)
                aurora_translated[key] = None
            continue

        with np.load(translator_path) as tr:
            # W is stored as (in_dim, out_dim), unlike the ArchesWeather translators
            W_mat = tr["W"].astype(np.float32)
            b_vec = tr["b"].astype(np.float32)
        x_translated = x_f32 @ W_mat + b_vec
        D_out = x_translated.shape[-1]
        aurora_translated[key] = _aurora_extract(
            x_translated, D_out, aurora_channel, AURORA_C, AURORA_H, AURORA_W, aurora_plevel_idx
        )
        if aurora_translated[key] is None:
            aurora_missing_translators.add(key)

if aurora_missing_translators:
    print(f"Aurora: missing or out-of-range translated channel for {sorted(aurora_missing_translators)} (will render blank)")

# same percentile-based colour limit as for GraphCast and ArchesWeather
aurora_vmax = symmetric_percentile_limit(
    [aurora_untranslated[key] for key in aurora_blocks if aurora_untranslated[key] is not None]
)
aurora_vmin = -aurora_vmax
print(f"Colour limit ({CBAR_PERCENTILE}th pct of raw |value|): Aurora +-{aurora_vmax:.3f}")

# %% Figure: raw and translated latent channel for GraphCast, Aurora and ArchesWeather

cmap_aurora_missing = mpl.colormaps[cmap_arch].copy()
cmap_aurora_missing.set_bad(color="lightgray")

fig_width_per_col = 3
fig_height_per_row = 2.5
n_cols = max(len(gc_steps), n_arch, n_aurora)


def _section_title(gs_row, text):
    """Section title in an invisible axis spanning the spacer row `gs_row`."""
    ax = fig.add_subplot(gs[gs_row, :])
    ax.axis('off')
    # bottom of the title at the bottom of the spacer row, lifted by TITLE_PAD points
    ax.annotate(text, xy=(0.5, 0), xycoords='axes fraction', xytext=(0, TITLE_PAD),
                textcoords='offset points', ha='center', va='bottom', fontsize=FONT_SIZE_TITLE)


# 9 rows: title, GraphCast (2 rows), title, Aurora (2 rows), title, ArchesWeather (2 rows)
fig = plt.figure(figsize=(fig_width_per_col*n_cols, fig_height_per_row*9))
gs = gridspec.GridSpec(9, n_cols, figure=fig, height_ratios=[0.3,1,1,0.55,1,1,0.55,1,1], hspace=0.05, wspace=0.05)

axes_gc = []
axes_arch = []
axes_aurora = []

# row 0: GraphCast title
_section_title(0, f"GraphCast latent channel {gc_channel}")

# rows 1 and 2: GraphCast
im_gc = None
for col, step in enumerate(gc_steps):
    latent_un = np.asarray(latents_gc_selected[f"step_{step}"]).squeeze(1)
    values_un = latent_un[:, gc_channel]

    ax = fig.add_subplot(gs[1, col])
    im_gc = ax.scatter(lon, lat, c=values_un, s=4, cmap=cmap_gc, vmin=gc_vmin, vmax=gc_vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("raw", fontsize=FONT_SIZE_ROW_LABEL)
    axes_gc.append(ax)

    latent_tr = np.asarray(translated_latents_gc_selected[f"step_{step}"]).squeeze(1)
    values_tr = latent_tr[:, gc_channel]

    ax = fig.add_subplot(gs[2, col])
    ax.scatter(lon, lat, c=values_tr, s=4, cmap=cmap_gc, vmin=gc_vmin, vmax=gc_vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("translated", fontsize=FONT_SIZE_ROW_LABEL)
    ax.set_xlabel(rf"$\tilde{{X}}_{{{step}}}$", fontsize=FONT_SIZE_H_LABEL)
    axes_gc.append(ax)

# row 3: Aurora title
_section_title(3, f"Aurora latent channel {aurora_channel}")

# rows 4 and 5: Aurora
im_aurora = None
for col, key in enumerate(aurora_blocks):
    data_un = aurora_untranslated[key]
    ax = fig.add_subplot(gs[4, col])
    if data_un is None:
        ax.imshow(np.full((AURORA_H, AURORA_W), np.nan), cmap=cmap_aurora_missing, origin="lower", aspect="auto")
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=FONT_SIZE_MISSING, transform=ax.transAxes, color="dimgray")
    else:
        img_obj = ax.imshow(data_un, cmap=cmap_arch, origin="lower", aspect="auto",
                             vmin=aurora_vmin, vmax=aurora_vmax)
        im_aurora = img_obj
    ax.set_xticks([])
    ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("raw", fontsize=FONT_SIZE_ROW_LABEL)
    axes_aurora.append(ax)

    data_tr = aurora_translated[key]
    ax = fig.add_subplot(gs[5, col])
    if data_tr is None:
        ax.imshow(np.full((AURORA_H, AURORA_W), np.nan), cmap=cmap_aurora_missing, origin="lower", aspect="auto")
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=FONT_SIZE_MISSING, transform=ax.transAxes, color="dimgray")
    else:
        ax.imshow(data_tr, cmap=cmap_arch, origin="lower", aspect="auto",
                  vmin=aurora_vmin, vmax=aurora_vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("translated", fontsize=FONT_SIZE_ROW_LABEL)
    ax.set_xlabel(aurora_h_label(key), fontsize=FONT_SIZE_H_LABEL)
    axes_aurora.append(ax)

# row 6: ArchesWeather title
_section_title(6, f"ArchesWeather latent channel {arch_channel}")

# rows 7 and 8: ArchesWeather
im_arch = None
for col in range(n_arch):
    data_un = arches_dataset_untranslated.isel(
        block=col,
        channel=arch_channel - 192 if col in [0,1,2,3,4] else arch_channel,
        plevel=plevel_idx
    ).values

    ax = fig.add_subplot(gs[7, col])
    im_arch = ax.imshow(data_un, cmap=cmap_arch, origin="upper", aspect="auto",
                        vmin=arch_vmin, vmax=arch_vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("raw", fontsize=FONT_SIZE_ROW_LABEL)
    axes_arch.append(ax)

    data_tr = arches_dataset.isel(
        block=col,
        channel=arch_channel,
        plevel=plevel_idx
    ).values

    ax = fig.add_subplot(gs[8, col])
    ax.imshow(data_tr, cmap=cmap_arch, origin="upper", aspect="auto",
              vmin=arch_vmin, vmax=arch_vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if col == 0:
        ax.set_ylabel("translated", fontsize=FONT_SIZE_ROW_LABEL)
    ax.set_xlabel(rf"$\tilde{{X}}_{{{col if col < 5 else col+24}}}$", fontsize=FONT_SIZE_H_LABEL)
    axes_arch.append(ax)

# colorbars; the arrows mark the values beyond the percentile limit
cbar_gc = fig.colorbar(im_gc, ax=axes_gc, shrink=0.9, location='right', pad=0.01, fraction=0.02, extend='both')
if im_aurora is not None:
    cbar_aurora = fig.colorbar(im_aurora, ax=axes_aurora, shrink=0.9, location='right', pad=0.01, fraction=0.02, extend='both')
    label_limit_colorbar(cbar_aurora, aurora_vmax)
cbar_arch = fig.colorbar(im_arch, ax=axes_arch, shrink=0.9, location='right', pad=0.01, fraction=0.02, extend='both')

label_limit_colorbar(cbar_gc, gc_vmax)
label_limit_colorbar(cbar_arch, arch_vmax)

plt.tight_layout(rect=[0,0,0.99,0.96])

save_path = os.path.join(path_plots, f"latent_state_development_{time_str}.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print("Saved:", save_path)

