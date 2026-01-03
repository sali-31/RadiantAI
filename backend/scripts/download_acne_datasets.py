#!/usr/bin/env python3
"""

Downloads only acne-relevant datasets for the project pivot.

Usage:
    python scripts/download_acne_datasets.py --all
    python scripts/download_acne_datasets.py --dataset acne_primary
    python scripts/download_acne_datasets.py --dataset fitzpatrick --filter-acne
"""

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional
import subprocess
import sys

# Check for required packages
try:
    from tqdm import tqdm
    import pandas as pd
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "pandas"])
    from tqdm import tqdm
    import pandas as pd


DATASETS = {
    'acne_primary': {
        'name': 'Acne Dataset (Primary)',
        'source': 'kaggle',
        'kaggle_dataset': 'nayanchaure/acne-dataset',
        'size': '~200MB',
        'images': 1800,
        'description': 'Core acne dataset with ~1,800 annotated images',
        'priority': 'HIGH',
        'output_dir': 'acne_primary'
    },
    'acne_spots': {
        'name': 'Acne-Wrinkles-Spots Classification',
        'source': 'kaggle',
        'kaggle_dataset': 'ranvijaybalbir/acne-wrinkles-spots-classification',
        'size': '~150MB',
        'images': 500,
        'description': 'Multi-label classification including acne',
        'priority': 'MEDIUM',
        'output_dir': 'acne_spots',
        'filter_labels': ['acne']
    },
    'skin_disease': {
        'name': 'Skin Disease Dataset (Filtered)',
        'source': 'kaggle',
        'kaggle_dataset': 'pacificrm/skindiseasedataset',
        'size': '~500MB',
        'images': 5000,
        'description': 'Broad skin disease dataset - will filter for acne only',
        'priority': 'MEDIUM',
        'output_dir': 'skin_disease',
        'filter_labels': ['acne', 'rosacea', 'Acne and Rosacea Photos']
    },
    'fitzpatrick': {
        'name': 'FitzPatrick17k (Diversity Testing)',
        'source': 'github',
        'github_repo': 'mattgroh/fitzpatrick17k',
        'size': '~2GB',
        'images': 100,
        'description': 'Skin tone diversity dataset - for bias testing',
        'priority': 'LOW',
        'output_dir': 'fitzpatrick17k',
        'filter_labels': ['acne'],
        'use_case': 'testing_only'
    }
}


def check_kaggle_setup():
    """Check if Kaggle API is set up"""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'

    if not kaggle_json.exists():
        print("=" * 70)
        print("⚠️  Kaggle API not configured")
        print("=" * 70)
        print("\nTo download Kaggle datasets, you need to:")
        print("\n1. Go to https://www.kaggle.com/settings/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token'")
        print("4. This downloads kaggle.json")
        print("5. Move it to ~/.kaggle/kaggle.json")
        print("\nCommands:")
        print("  mkdir -p ~/.kaggle")
        print("  mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("  chmod 600 ~/.kaggle/kaggle.json")
        print("=" * 70)
        return False

    # Install kaggle package if needed
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        import kaggle

    return True


def download_from_kaggle(dataset_id: str, output_dir: Path):
    """Download dataset from Kaggle"""
    import kaggle

    print(f"\n📥 Downloading from Kaggle: {dataset_id}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        kaggle.api.dataset_download_files(
            dataset_id,
            path=output_dir,
            unzip=True,
            quiet=False
        )
        print(f"✓ Downloaded to {output_dir}")
        return True
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        return False


def download_from_github(repo: str, output_dir: Path):
    """Download dataset from GitHub"""
    print(f"\n📥 Downloading from GitHub: {repo}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clone repository
    repo_url = f"https://github.com/{repo}.git"

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(output_dir)],
            check=True,
            capture_output=True
        )
        print(f"✓ Cloned to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to clone: {e}")
        print("\nAlternative: Download manually from:")
        print(f"  https://github.com/{repo}")
        return False


def filter_dataset_for_acne(dataset_dir: Path, filter_labels: list):
    """
    Filter a broad dataset to keep only acne-related images

    Args:
        dataset_dir: Path to dataset
        filter_labels: Labels to keep (e.g., ['acne', 'rosacea'])
    """
    print(f"\n🔍 Filtering dataset for labels: {filter_labels}")

    # Look for metadata files
    metadata_files = list(dataset_dir.glob("*.csv")) + list(dataset_dir.glob("**/metadata.csv"))

    if not metadata_files:
        print("⚠️  No metadata CSV found - cannot auto-filter")
        print("   You'll need to manually filter this dataset")
        return

    metadata_file = metadata_files[0]
    print(f"📄 Found metadata: {metadata_file.name}")

    # Read metadata
    df = pd.read_csv(metadata_file)
    print(f"   Total images: {len(df)}")

    # Try to find label column
    label_cols = [col for col in df.columns if 'label' in col.lower() or 'class' in col.lower()]

    if not label_cols:
        print("⚠️  No label column found in metadata")
        print(f"   Columns: {df.columns.tolist()}")
        return

    label_col = label_cols[0]
    print(f"   Using label column: {label_col}")

    # Filter for acne-related labels
    mask = df[label_col].str.contains('|'.join(filter_labels), case=False, na=False)
    filtered_df = df[mask]

    print(f"   Acne-related images: {len(filtered_df)}")

    if len(filtered_df) == 0:
        print("⚠️  No acne-related images found")
        return

    # Save filtered metadata
    output_file = dataset_dir / 'acne_filtered_metadata.csv'
    filtered_df.to_csv(output_file, index=False)
    print(f"✓ Saved filtered metadata to {output_file.name}")

    # Optionally move non-acne images to separate folder
    print("\n📁 Organizing images...")
    acne_dir = dataset_dir / 'acne_only'
    other_dir = dataset_dir / 'other_conditions'

    acne_dir.mkdir(exist_ok=True)
    other_dir.mkdir(exist_ok=True)

    # This is a placeholder - actual implementation depends on dataset structure
    print("   Note: Manual organization recommended - dataset structures vary")


def create_unified_dataset(data_dir: Path, output_dir: Path):
    """
    Combine all downloaded acne datasets into unified format

    Creates:
        acne_unified/
            train/
            val/
            test/
            metadata.json
    """
    print("\n🔗 Creating unified acne dataset...")

    unified_dir = output_dir
    unified_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        (unified_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (unified_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Find all acne datasets
    raw_dir = data_dir / 'raw'
    acne_datasets = [d for d in raw_dir.iterdir() if d.is_dir() and 'acne' in d.name.lower()]

    print(f"📦 Found {len(acne_datasets)} acne datasets")

    # Metadata for unified dataset
    unified_metadata = {
        'datasets_included': [],
        'total_images': 0,
        'split_ratio': {'train': 0.7, 'val': 0.15, 'test': 0.15},
        'classes': []
    }

    for dataset_dir in acne_datasets:
        print(f"\n   Processing: {dataset_dir.name}")

        # Count images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(dataset_dir.rglob(ext))

        if image_files:
            print(f"      Found {len(image_files)} images")
            unified_metadata['datasets_included'].append({
                'name': dataset_dir.name,
                'images': len(image_files),
                'path': str(dataset_dir.relative_to(data_dir))
            })
            unified_metadata['total_images'] += len(image_files)

    # Save metadata
    metadata_path = unified_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(unified_metadata, f, indent=2)

    print(f"\n✓ Unified dataset metadata saved to {metadata_path}")
    print(f"   Total images available: {unified_metadata['total_images']}")
    print("\n⚠️  Note: Actual data organization requires manual review")
    print("   Different datasets have different annotation formats")
    print("   Recommended: Use Roboflow to unify annotations")


def create_samples_for_git_lfs(data_dir: Path, num_samples: int = 10):
    """Create sample images for Git LFS"""
    print(f"\n🖼️  Creating {num_samples} sample images for Git LFS...")

    samples_dir = data_dir / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Find acne images
    raw_dir = data_dir / 'raw'
    image_files = []

    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(raw_dir.rglob(ext))

    if not image_files:
        print("⚠️  No images found - download datasets first")
        return

    # Copy random samples
    import random
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    for i, img_path in enumerate(samples, 1):
        dest = samples_dir / f"acne_sample_{i}{img_path.suffix}"

        if dest.exists():
            print(f"   ✓ {dest.name} already exists")
            continue

        shutil.copy2(img_path, dest)
        print(f"   ✓ Created {dest.name}")

    total_size = sum(f.stat().st_size for f in samples_dir.glob("*"))
    print(f"\n✓ Created {len(list(samples_dir.glob('*')))} samples ({total_size / 1024 / 1024:.2f} MB)")

    if total_size > 100 * 1024 * 1024:
        print("⚠️  WARNING: Samples exceed 100MB - consider reducing")


def print_dataset_info():
    """Print information about available datasets"""
    print("\n" + "=" * 70)
    print("ACNE DETECTION DATASETS")
    print("=" * 70)

    for key, info in DATASETS.items():
        print(f"\n{info['priority']} - {info['name']}")
        print(f"  Dataset ID: {key}")
        print(f"  Source: {info['source']}")
        print(f"  Size: {info['size']}")
        print(f"  Images: ~{info['images']}")
        print(f"  Description: {info['description']}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Download acne-specific datasets for LesionRec",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--dataset',
        choices=list(DATASETS.keys()),
        help='Specific dataset to download'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all acne datasets'
    )

    parser.add_argument(
        '--info',
        action='store_true',
        help='Show information about available datasets'
    )

    parser.add_argument(
        '--filter-acne',
        action='store_true',
        help='Filter broad datasets for acne-only images'
    )

    parser.add_argument(
        '--create-unified',
        action='store_true',
        help='Create unified dataset from all sources'
    )

    parser.add_argument(
        '--create-samples',
        action='store_true',
        help='Create sample images for Git LFS'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'data',
        help='Data directory (default: ./data)'
    )

    args = parser.parse_args()

    if args.info:
        print_dataset_info()
        return

    print("=" * 70)
    print("LESIONREC - ACNE DATASET DOWNLOADER")
    print("=" * 70)

    # Check setup
    if not check_kaggle_setup():
        print("\n⚠️  Set up Kaggle API first (see instructions above)")
        return

    # Determine which datasets to download
    datasets_to_download = []

    if args.all:
        datasets_to_download = list(DATASETS.keys())
    elif args.dataset:
        datasets_to_download = [args.dataset]

    # Download datasets
    for dataset_key in datasets_to_download:
        info = DATASETS[dataset_key]
        output_dir = args.data_dir / 'raw' / info['output_dir']

        print(f"\n{'=' * 70}")
        print(f"📦 {info['name']}")
        print(f"{'=' * 70}")

        # Check if already downloaded
        if output_dir.exists() and any(output_dir.iterdir()):
            print(f"✓ Already downloaded to {output_dir}")
            if args.filter_acne and 'filter_labels' in info:
                filter_dataset_for_acne(output_dir, info['filter_labels'])
            continue

        # Download based on source
        if info['source'] == 'kaggle':
            success = download_from_kaggle(info['kaggle_dataset'], output_dir)
        elif info['source'] == 'github':
            success = download_from_github(info['github_repo'], output_dir)
        else:
            print(f"⚠️  Unknown source: {info['source']}")
            continue

        # Filter if requested
        if success and args.filter_acne and 'filter_labels' in info:
            filter_dataset_for_acne(output_dir, info['filter_labels'])

    # Create unified dataset
    if args.create_unified:
        unified_dir = args.data_dir / 'processed' / 'acne_unified'
        create_unified_dataset(args.data_dir, unified_dir)

    # Create samples
    if args.create_samples:
        create_samples_for_git_lfs(args.data_dir)

    print("\n" + "=" * 70)
    print("✓ DOWNLOAD COMPLETE")
    print("=" * 70)

    if datasets_to_download:
        print("\nNext steps:")
        print("1. Review downloaded datasets in data/raw/")
        print("2. Filter for acne: python scripts/download_acne_datasets.py --filter-acne")
        print("3. Create unified dataset: python scripts/download_acne_datasets.py --create-unified")
        print("4. Create samples: python scripts/download_acne_datasets.py --create-samples")
        print("5. Track with DVC: dvc add data/raw/acne_*")


if __name__ == '__main__':
    main()
