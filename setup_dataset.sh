#!/bin/bash

# Constants
ZIP_NAME="cafa-5-protein-function-prediction.zip"
ZIP_PATH="dataset/$ZIP_NAME"
TARGET_DIR="dataset/"

# Kaggle competition name
KAGGLE_COMPETITION="cafa-5-protein-function-prediction"

# Ensure dataset directory exists
mkdir -p dataset

# Step 1: Download if not already downloaded
if [ ! -f "$ZIP_PATH" ]; then
    echo "⬇️  Dataset not found. Downloading from Kaggle..."

    # Check if Kaggle CLI is installed
    if ! command -v kaggle &> /dev/null; then
        echo "❌ Kaggle CLI is not installed. Please install it first: https://github.com/Kaggle/kaggle-api"
        exit 1
    fi

    # Download dataset
    kaggle competitions download -c "$KAGGLE_COMPETITION" -p dataset

    if [ $? -ne 0 ]; then
        echo "❌ Kaggle download failed."
        exit 1
    fi

    echo "✅ Download complete!"
else
    echo "📁 Dataset zip already exists at $ZIP_PATH"
fi

# Step 2: Extract dataset
echo "📦 Extracting $ZIP_PATH to $TARGET_DIR ..."
unzip -o "$ZIP_PATH" -d "$TARGET_DIR"

echo "✅ Extraction complete!"