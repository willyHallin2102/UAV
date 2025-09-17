#!/bin/bash
set -e

VENV="venv"
PYTHON="python3.10"


command_exists() {
    command -v "$1" &>/dev/null
}

if ! command_exists "$PYTHON"; then
    echo "${PYTHON} not found, please install it first."
    exit 1
fi

# Check if virtualenv already exists and offer to remove it
if [ -d "$VENV" ]; then
    echo "Virtual environment '$VENV' already exists."
    read -p "Do you want to remove it? (y/n): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV"
    else
        echo "Skipping virtual environment removal."
    fi
fi

echo "Creating virtual environment..."
$PYTHON -m venv "$VENV"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV/bin/activate"

# Ensure pip exists and upgrade
echo "Ensuring pip is up to date..."
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel

# Install required packages
echo "Installing necessary packages..."
pip install pandas seaborn scikit-learn tqdm orjson

# Install from GitHub repository (Sionna)
echo "Installing Sionna from GitHub..."
pip install git+https://github.com/NVlabs/sionna.git@main

# Check if requirements.txt exists and ask if you want to overwrite it
if [ -f "requirements.txt" ]; then
    echo "requirements.txt already exists."
    read -p "Do you want to overwrite it? (y/n): " overwrite
    if [[ "$overwrite" =~ ^[Yy]$ ]]; then
        echo "Overwriting requirements.txt..."
        pip freeze > requirements.txt
    else
        echo "Skipping overwriting requirements.txt."
    fi
else
    echo "Generating requirements.txt..."
    pip freeze > requirements.txt
fi

echo "Setup completed successfully!"

# Deactivate the virtual environment
deactivate