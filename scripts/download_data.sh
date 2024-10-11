#!/bin/bash

# Check if competition slug is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <competition-slug> [data-directory]"
  exit 1
fi

# Set the competition slug from the first argument
COMPETITION_SLUG=$1

# If a data directory is provided, use it; otherwise, default to the current directory
if [ -n "$2" ]; then
  DATA_DIR=$2
else
  DATA_DIR=$COMPETITION_SLUG
fi

# Check if Kaggle is installed and configured
if ! command -v kaggle &> /dev/null; then
  echo "Kaggle CLI is not installed. Please install it using 'pip install kaggle' and configure it."
  exit 1
fi

# Create the data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Download the competition dataset as a .zip file
echo "Downloading dataset for competition: $COMPETITION_SLUG"
kaggle competitions download -c "$COMPETITION_SLUG" -p "$DATA_DIR"

# Extract the downloaded .zip file into the data directory
echo "Extracting dataset..."
unzip "$DATA_DIR"/*.zip -d "$DATA_DIR"

# Remove the .zip file after extraction
echo "Cleaning up..."
rm "$DATA_DIR"/*.zip

echo "Dataset downloaded and extracted to $DATA_DIR/"
