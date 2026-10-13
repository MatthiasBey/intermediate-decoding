import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% IMPORTS
# Table of the encoder-decoder pattern correlations (main text): mean and std over config.DATES
# of the per-date results of calculations/3_encoder_decoder_similarity/compute_encoder_decoder_similarity.py.
# Output: results/<mode>/encoder_decoder_similarity/summary_<config.DATES_TAG>.csv
import pickle

import numpy as np

import utils.encoder_decoder_similarity as sim
from utils.encoder_decoder_similarity import MODEL_VARS

out_dir = f"{config.RESULTS_DIR}/encoder_decoder_similarity"


def load_date(analysis_datetime):
    with open(os.path.join(out_dir, f"{analysis_datetime:%Y-%m-%d-%H-%M}.pkl"), "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    per_date = [load_date(d) for d in config.DATES]

    model_labels = {"graphcast": "GraphCast", "aurora": "Aurora", "arches": "ArchesWeather"}
    keys = [f"corr_{a}_{b}" for a, b in sim.PAIRS] + ["partial_corr_encdec_full_given_input"] + [f"amp_{a}_{b}" for a, b in sim.PAIRS]

    lines = ["model,variable," + ",".join(f"{k}_mean,{k}_std" for k in keys)]
    for model, label in model_labels.items():
        for var in MODEL_VARS[model]:
            vals = {k: np.array([r[model][var][k] for r in per_date]) for k in keys}
            lines.append(f"{label},{var}," + ",".join(f"{vals[k].mean():.4f},{vals[k].std():.4f}" for k in keys))
    summary_path = os.path.join(out_dir, f"summary_{config.DATES_TAG}.csv")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {summary_path}")
