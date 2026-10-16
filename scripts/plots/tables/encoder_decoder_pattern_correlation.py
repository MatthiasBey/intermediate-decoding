"""Table of the encoder-decoder pattern correlations (main text): mean and standard deviation over
config.DATES of the per-date results of
calculations/3_encoder_decoder_similarity/compute_encoder_decoder_similarity.py.
Output: results/<mode>/encoder_decoder_similarity/summary_<config.DATES_TAG>.csv
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
from utils.encoder_decoder_similarity import MODEL_VARS, PAIRS

MODEL_LABELS = {"graphcast": "GraphCast", "aurora": "Aurora", "arches": "ArchesWeather"}
KEYS = ([f"corr_{a}_{b}" for a, b in PAIRS] + ["partial_corr_encdec_full_given_input"]
        + [f"amp_{a}_{b}" for a, b in PAIRS])

out_dir = f"{config.RESULTS_DIR}/encoder_decoder_similarity"
per_date = []
for date in config.DATES:
    with open(os.path.join(out_dir, f"{date:%Y-%m-%d-%H-%M}.pkl"), "rb") as f:
        per_date.append(pickle.load(f))

lines = ["model,variable," + ",".join(f"{key}_mean,{key}_std" for key in KEYS)]
for model, label in MODEL_LABELS.items():
    for var in MODEL_VARS[model]:
        values = {key: np.array([result[model][var][key] for result in per_date]) for key in KEYS}
        lines.append(f"{label},{var}," + ",".join(f"{values[key].mean():.4f},{values[key].std():.4f}" for key in KEYS))

summary_path = os.path.join(out_dir, f"summary_{config.DATES_TAG}.csv")
with open(summary_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nSaved {summary_path}")
