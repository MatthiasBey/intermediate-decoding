#!/bin/bash
# Creates the conda environment "intermediate_decoding" for all scripts in this repository.
#
# Usage:
#   bash install_env.sh                 # creates the environment "intermediate_decoding"
#   bash install_env.sh /path/to/env    # creates the environment in the given folder
# Then activate it with "conda activate intermediate_decoding" (or "conda activate /path/to/env").
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

if [ $# -ge 1 ]; then TARGET=(-p "$1"); else TARGET=(-n intermediate_decoding); fi

# Allow pip to install into the conda environment, even where PIP_REQUIRE_VIRTUALENV is set.
export PIP_REQUIRE_VIRTUALENV=false

conda env create "${TARGET[@]}" -f environment.yml

# From GraphCast, only the icosahedral mesh is needed, to place the latent states on the mesh.
# It is installed without the dependencies of the full model, at release v0.2.
conda run "${TARGET[@]}" pip install --no-deps \
    "graphcast @ https://github.com/google-deepmind/graphcast/archive/refs/tags/v0.2.zip"

# Check that everything can be imported.
conda run "${TARGET[@]}" python -c "import numpy, scipy, xarray, dask, netCDF4, matplotlib, seaborn, pandas, psutil, torch, jax; from graphcast import icosahedral_mesh; print('environment OK')"
conda run "${TARGET[@]}" cdo --version | head -1
