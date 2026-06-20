#!/usr/bin/env bash

# Exist immediately if:
#   - A command exits with non-zero status (-e)
#   - An undefined variable is being used (-u)
#   - A pipeline fails anywhere (-o pipefail)
set -euo pipefail

#   Name of the virtual environment & Python Version
VENV=".venv"
PYTHON="${1:-python3.11}"

echo "Activating the virtual environment"
"$PYTHON" -m venv "$VENV"


#   Activate venv
#   Shellcheck disable=SC1090
source "$VENV/bin/activate"



# -------------------------
#   Core Tooling
# -------------------------

echo "Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel


# -------------------------
#   Install Tensorflow
# -------------------------

echo "Installing dependencies..."

pip install \
    "tensorflow[and-cuda]==2.20.*" \
    pandas \
    numpy \
    scipy \
    pyarrow \
    numba \
    matplotlib \
    seaborn \
    tqdm \
    orjson \
    einops \
    h5py \
    plotly \
    tensorboard

pip install git+https://github.com/NVlabs/sionna.git@main

pip freeze > requirements-lock.txt



