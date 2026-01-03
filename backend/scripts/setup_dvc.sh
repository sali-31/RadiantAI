#!/bin/bash

set -e

echo "=== DVC Setup for LesionRec (Acne Detection) ==="
echo ""

# Check if DVC is installed
if ! command -v dvc &> /dev/null; then
    echo "DVC is not installed. Installing DVC..."
    pip install dvc dvc-gdrive
else
    echo "✓ DVC is already installed"
fi

# Initialize DVC (if not already initialized)
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
    echo "✓ DVC initialized"
else
    echo "✓ DVC already initialized"
fi

# Configure DVC
echo ""
echo "Configuring DVC settings..."
dvc config core.autostage true
dvc config core.analytics false

echo ""
echo "=== Google Drive Setup ==="
echo "To set up Google Drive as your remote storage for acne datasets:"
echo ""
echo "1. Create a folder in your Google Drive: 'LesionRec_Acne_Data'"
echo "2. Get the folder ID from the URL:"
echo "   https://drive.google.com/drive/folders/YOUR_FOLDER_ID"
echo ""
echo "3. Run this command with your folder ID:"
echo "   dvc remote add -d gdrive gdrive://YOUR_FOLDER_ID"
echo ""
echo "4. Authenticate with Google Drive:"
echo "   dvc push"
echo ""
echo "The first time you push, you'll be prompted to authenticate."
echo ""
echo "=== Setup Complete ==="
echo "DVC is configured and ready to use for acne datasets!"
echo ""
echo "Next steps:"
echo "1. Download acne datasets: python scripts/download_acne_datasets.py --all"
echo "2. Add datasets to DVC: dvc add data/raw/acne_primary"
echo "3. Commit the .dvc file: git add data/raw/acne_primary.dvc .gitignore"
echo "4. Push to Google Drive: dvc push"
echo ""
echo "See data/README.md for detailed instructions."
