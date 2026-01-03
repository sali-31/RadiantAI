# RadiantAI - Acne Detection

This directory contains all acne datasets, processed data, and models for the RadiantAI project.

## Directory Structure

```
data/
├── raw/              # Original, immutable datasets (tracked with DVC)
│   ├── acne_primary/     # Main Acne Dataset (~1,800 images)
│   ├── acne_spots/       # Acne-Wrinkles-Spots dataset
│   ├── skin_disease/     # Skin Disease Dataset (filtered for acne)
│   └── fitzpatrick17k/   # FitzPatrick17k (diversity testing)
├── processed/        # Cleaned and preprocessed data (tracked with DVC)
│   ├── acne_unified/     # Combined dataset
│   ├── train/
│   ├── val/
│   └── test/
└── samples/          # Small sample images (tracked with Git LFS)
    └── acne_sample_*.jpg  # Representative acne images for quick testing
```

## Getting Started

### 1. Install Dependencies

```bash
# Install DVC with Google Drive support
pip install dvc dvc-gdrive

# Install Kaggle API (for dataset downloads)
pip install kaggle

# Install Git LFS (if not already installed)
# macOS:
brew install git-lfs

# Ubuntu/Debian:
apt-get install git-lfs
```

### 2. Set Up Kaggle API

Most acne datasets are on Kaggle. Set up API access:

```bash
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New Token" under API section
# 3. This downloads kaggle.json

# 4. Move to ~/.kaggle/
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 5. Test it
kaggle datasets list
```

### 3. Initialize DVC

```bash
# Run the setup script
bash scripts/setup_dvc.sh

# Or manually:
dvc init
dvc config core.autostage true
```

### 4. Connect to Google Drive

**First Time Setup (One person):**

1. Create a folder in Google Drive for this project: "LesionRec_Acne_Data"
2. Get the folder ID from the URL: `https://drive.google.com/drive/folders/FOLDER_ID`
3. Add the remote:
   ```bash
   dvc remote add -d gdrive gdrive://YOUR_FOLDER_ID
   ```
4. Commit the config:
   ```bash
   git add .dvc/config
   git commit -m "Configure DVC remote storage for acne datasets"
   git push
   ```

**Team Members:**

After cloning the repo, just pull the data:
```bash
dvc pull
```

On first pull, you'll authenticate with Google Drive once.

## Downloading Acne Datasets

### Option 1: Use the Download Script (Recommended)

```bash
# Download all acne datasets
python scripts/download_acne_datasets.py --all

# Download specific dataset
python scripts/download_acne_datasets.py --dataset acne_primary
python scripts/download_acne_datasets.py --dataset acne_spots

# Filter broad datasets for acne only
python scripts/download_acne_datasets.py --filter-acne

# Create unified dataset
python scripts/download_acne_datasets.py --create-unified

# Create sample images for Git LFS
python scripts/download_acne_datasets.py --create-samples
```

### Option 2: Manual Download

**Acne Dataset (Primary)**:
1. Visit: https://www.kaggle.com/datasets/nayanchaure/acne-dataset
2. Download and extract
3. Place in `data/raw/acne_primary/`

**Acne-Wrinkles-Spots Classification**:
1. Visit: https://www.kaggle.com/datasets/ranvijaybalbir/acne-wrinkles-spots-classification
2. Download and extract
3. Place in `data/raw/acne_spots/`

**Skin Disease Dataset (Filter for Acne)**:
1. Visit: https://www.kaggle.com/datasets/pacificrm/skindiseasedataset
2. Download and extract
3. Place in `data/raw/skin_disease/`
4. Run filter script to extract acne-only images

**FitzPatrick17k (Diversity Testing)**:
1. Visit: https://github.com/mattgroh/fitzpatrick17k
2. Clone or download
3. Place in `data/raw/fitzpatrick17k/`
4. Filter for acne cases only

## Tracking Data with DVC

### Adding New Datasets

```bash
# Add a dataset to DVC tracking
dvc add data/raw/acne_primary

# This creates data/raw/acne_primary.dvc
# The actual data is NOT tracked by git

# Commit the .dvc file
git add data/raw/acne_primary.dvc data/raw/.gitignore
git commit -m "Add Acne Primary dataset"

# Push data to Google Drive
dvc push

# Push git changes
git push
```

### Pulling Data (Team Members)

```bash
# Pull all datasets
dvc pull

# Pull specific dataset
dvc pull data/raw/acne_primary.dvc
```

### Updating Datasets

```bash
# Modify the data
# Then update DVC tracking
dvc add data/raw/acne_primary

# Commit changes
git add data/raw/acne_primary.dvc
git commit -m "Update Acne Primary dataset with new annotations"

# Push to Google Drive
dvc push
git push
```

## Sample Images (Git LFS)

Small sample images are stored in `data/samples/` using Git LFS.

**Why?** This allows team members to:
- Test code without downloading full datasets (~2GB+)
- Have quick access to example acne images
- Keep total repo size reasonable

**Usage:**

```bash
# Git LFS is automatically initialized
# Sample images are automatically pulled with git clone

# To manually pull LFS files:
git lfs pull
```

**Adding Samples:**

```bash
# Create samples automatically from downloaded datasets
python scripts/download_acne_datasets.py --create-samples

# Or manually add (keep under 100MB total)
cp acne_image.jpg data/samples/acne_sample_11.jpg
git add data/samples/acne_sample_11.jpg
git commit -m "Add new acne sample image"
git push
```

## Data Preprocessing

### Filtering for Acne

For broad dermatology datasets, filter to keep only acne:

```python
import pandas as pd
from pathlib import Path

# Load metadata
df = pd.read_csv('data/raw/skin_disease/metadata.csv')

# Filter for acne-related labels
acne_labels = ['acne', 'rosacea', 'Acne and Rosacea Photos', 'comedone', 'papule', 'pustule']
mask = df['label'].str.contains('|'.join(acne_labels), case=False, na=False)
acne_df = df[mask]

# Save filtered metadata
acne_df.to_csv('data/raw/skin_disease/acne_filtered.csv', index=False)

print(f"Filtered {len(acne_df)} acne images from {len(df)} total")
```

### Creating Unified Dataset

Combine all acne datasets into one unified format:

```bash
python scripts/download_acne_datasets.py --create-unified
```

This creates:
```
data/processed/acne_unified/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── metadata.json
```

## Dataset Information

### Acne Primary Dataset
- **Source**: Kaggle
- **Size**: ~200MB
- **Images**: ~1,800
- **Format**: JPG images with labels
- **Classes**: Various acne types
- **Use**: Primary training dataset

### Acne-Wrinkles-Spots
- **Source**: Kaggle
- **Size**: ~150MB
- **Images**: ~500 (acne-labeled)
- **Format**: Multi-label classification
- **Use**: Supplementary training

### Skin Disease Dataset (Filtered)
- **Source**: Kaggle
- **Size**: ~500MB (full), ~50MB (acne only)
- **Images**: ~500 acne cases (from 5,000 total)
- **Format**: Classification labels
- **Use**: Additional training data

### FitzPatrick17k (Filtered)
- **Source**: GitHub
- **Size**: ~2GB (full), ~200MB (acne only)
- **Images**: ~100 acne cases
- **Format**: Images with skin tone labels (I-VI)
- **Use**: **Diversity/bias testing ONLY**
- **Purpose**: Ensure model fairness across skin tones

## Best Practices

### DO:
- ✅ Always use DVC for datasets (> 10MB)
- ✅ Use Git LFS only for small sample files (< 100MB total)
- ✅ Keep `data/raw/` immutable - never modify original data
- ✅ Put processed data in `data/processed/`
- ✅ Commit `.dvc` files to git
- ✅ Run `dvc push` after adding/updating datasets
- ✅ Document data sources and filtering steps
- ✅ Test on FitzPatrick17k for fairness

### DON'T:
- ❌ Never commit large datasets directly to git
- ❌ Don't modify raw data - create processed versions instead
- ❌ Don't forget to `dvc push` before git pushing
- ❌ Don't share Google Drive credentials in code/commits
- ❌ Don't use FitzPatrick17k for training (testing only!)

## Workflow Examples

### Starting Fresh on a New Machine

```bash
# Clone the repository
git clone <repo-url>
cd LesionRec

# Install dependencies
pip install dvc dvc-gdrive kaggle

# Set up Kaggle API
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Pull data from Google Drive
dvc pull

# Pull sample images (automatic with git clone)
git lfs pull

# Ready to work!
python src/ensemble_detector.py
```

### Adding a New Acne Dataset

```bash
# Download and place in data/raw/new_acne_dataset/

# Track with DVC
dvc add data/raw/new_acne_dataset

# Commit
git add data/raw/new_acne_dataset.dvc data/raw/.gitignore
git commit -m "Add new acne dataset"

# Push to Google Drive and git
dvc push
git push

# Teammates can now: dvc pull
```

### Testing for Bias Across Skin Tones

```bash
# Filter FitzPatrick17k for acne cases
python scripts/download_acne_datasets.py --dataset fitzpatrick --filter-acne

# Test ensemble detector on each skin type
from src.ensemble_detector import AcneEnsembleDetector
detector = AcneEnsembleDetector(api_key="YOUR_KEY")

results_by_skin_type = {}
for skin_type in ['I', 'II', 'III', 'IV', 'V', 'VI']:
    # Load images for this skin type
    images = load_fitzpatrick_images(skin_type, condition='acne')

    # Test detection
    for img in images:
        result = detector.detect(img)
        results_by_skin_type[skin_type].append(result.count)

# Analyze variance
variances = {st: np.std(counts) for st, counts in results_by_skin_type.items()}
print(f"Accuracy variance: {variances}")

# Flag if variance > 10%
if max(variances.values()) - min(variances.values()) > 0.10:
    print("⚠️ WARNING: Model shows bias across skin tones")
```

## Troubleshooting

### DVC Pull Fails

```bash
# Check remote configuration
dvc remote list

# Verify Google Drive authentication
dvc pull -v

# Re-authenticate if needed
rm .dvc/tmp/gdrive-user-credentials.json
dvc pull
```

### Kaggle Download Fails

```bash
# Check API is set up
ls ~/.kaggle/kaggle.json

# Test connection
kaggle datasets list

# If permission denied:
chmod 600 ~/.kaggle/kaggle.json
```

### Git LFS Quota Issues

If you hit GitHub LFS bandwidth limits:
- Use DVC for those files instead
- Consider self-hosted Git LFS
- Move to different hosting (GitLab has higher LFS limits)

### Cache Issues

```bash
# Clear DVC cache
dvc cache dir
rm -rf .dvc/cache

# Re-pull from remote
dvc pull -f
```

## Data Statistics

| Dataset | Size | Images | Acne Types | Split | Use |
|---------|------|--------|------------|-------|-----|
| Acne Primary | ~200MB | 1,800 | Various | 70/15/15 | Training |
| Acne-Spots | ~150MB | 500 | Multi-label | 70/15/15 | Training |
| Skin Disease (filtered) | ~50MB | 500 | Acne/Rosacea | 70/15/15 | Training |
| FitzPatrick17k (filtered) | ~200MB | 100 | Various | - | Testing Only |

**Total Training Data**: ~2,800 acne images
**Testing Data**: ~100 images (diverse skin tones)

## Resources

- [DVC Documentation](https://dvc.org/doc)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [Kaggle API Guide](https://www.kaggle.com/docs/api)
- [Acne Dataset on Kaggle](https://www.kaggle.com/datasets/nayanchaure/acne-dataset)
- [FitzPatrick17k Paper](https://arxiv.org/abs/2104.09957)
