#!/usr/bin/env python3
"""
Download nuScenes dataset.

This script helps download the nuScenes dataset from the official source.
You need to register at https://www.nuscenes.org/nuscenes first.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm


# nuScenes download URLs (you need to be logged in to access these)
NUSCENES_URLS = {
    'mini': {
        'metadata': 'https://www.nuscenes.org/data/v1.0-mini.tgz',
        'samples': 'https://motional-nuscenes.s3.amazonaws.com/v1.0-mini.tgz',
        'size': '11GB'
    },
    'trainval': {
        'metadata': 'https://www.nuscenes.org/data/v1.0-trainval_meta.tgz',
        'samples': [
            'https://motional-nuscenes.s3.amazonaws.com/v1.0-trainval01_blobs.tgz',
            'https://motional-nuscenes.s3.amazonaws.com/v1.0-trainval02_blobs.tgz',
            'https://motional-nuscenes.s3.amazonaws.com/v1.0-trainval03_blobs.tgz',
            'https://motional-nuscenes.s3.amazonaws.com/v1.0-trainval04_blobs.tgz',
            'https://motional-nuscenes.s3.amazonaws.com/v1.0-trainval05_blobs.tgz',
            'https://motional-nuscenes.s3.amazonaws.com/v1.0-trainval06_blobs.tgz',
            'https://motional-nuscenes.s3.amazonaws.com/v1.0-trainval07_blobs.tgz',
            'https://motional-nuscenes.s3.amazonaws.com/v1.0-trainval08_blobs.tgz',
            'https://motional-nuscenes.s3.amazonaws.com/v1.0-trainval09_blobs.tgz',
            'https://motional-nuscenes.s3.amazonaws.com/v1.0-trainval10_blobs.tgz',
        ],
        'size': '350GB'
    }
}


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        chunk_size: Size of chunks to download
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading: {url}")
    print(f"Saving to: {output_path}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓ Downloaded: {output_path.name}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {url}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def extract_archive(archive_path: Path, output_dir: Path):
    """
    Extract a .tgz archive.
    
    Args:
        archive_path: Path to the archive
        output_dir: Directory to extract to
    """
    print(f"\nExtracting: {archive_path.name}")
    
    try:
        subprocess.run(
            ['tar', '-xzf', str(archive_path), '-C', str(output_dir)],
            check=True,
            capture_output=True
        )
        print(f"✓ Extracted: {archive_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error extracting {archive_path}: {e}")
        return False
    except FileNotFoundError:
        print("✗ Error: 'tar' command not found. Please install tar.")
        return False


def download_nuscenes(split: str, output_dir: Path, extract: bool = True, cleanup: bool = True):
    """
    Download nuScenes dataset.
    
    Args:
        split: 'mini' or 'trainval'
        output_dir: Output directory
        extract: Whether to extract archives
        cleanup: Whether to delete archives after extraction
    """
    if split not in NUSCENES_URLS:
        print(f"✗ Invalid split: {split}. Choose from: {list(NUSCENES_URLS.keys())}")
        return False
    
    urls = NUSCENES_URLS[split]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"nuScenes Dataset Download - {split.upper()}")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Expected size: {urls['size']}")
    print("=" * 80)
    
    # Download metadata
    print("\n[1/2] Downloading metadata...")
    metadata_url = urls['metadata']
    metadata_file = output_dir / Path(metadata_url).name
    
    if not download_file(metadata_url, metadata_file):
        return False
    
    if extract:
        if not extract_archive(metadata_file, output_dir):
            return False
        if cleanup:
            print(f"Removing: {metadata_file.name}")
            metadata_file.unlink()
    
    # Download samples
    print("\n[2/2] Downloading samples...")
    samples_urls = urls['samples']
    if isinstance(samples_urls, str):
        samples_urls = [samples_urls]
    
    for i, url in enumerate(samples_urls, 1):
        print(f"\n--- Sample {i}/{len(samples_urls)} ---")
        sample_file = output_dir / Path(url).name
        
        if not download_file(url, sample_file):
            print(f"Warning: Failed to download {url}")
            continue
        
        if extract:
            if not extract_archive(sample_file, output_dir):
                print(f"Warning: Failed to extract {sample_file}")
                continue
            if cleanup:
                print(f"Removing: {sample_file.name}")
                sample_file.unlink()
    
    print("\n" + "=" * 80)
    print("✓ Download complete!")
    print("=" * 80)
    print(f"\nDataset location: {output_dir}")
    print("\nNext steps:")
    print("1. Verify installation:")
    print(f"   python -c \"from nuscenes.nuscenes import NuScenes; nusc = NuScenes(version='v1.0-{split}', dataroot='{output_dir}')\"")
    print("\n2. Preprocess for training:")
    print(f"   python scripts/prepare_dataset.py --dataset nuscenes --data-root {output_dir}")
    print("=" * 80)
    
    return True


def download_via_wget(split: str, output_dir: Path):
    """
    Alternative download method using wget (faster for large files).
    
    Args:
        split: 'mini' or 'trainval'
        output_dir: Output directory
    """
    if split not in NUSCENES_URLS:
        print(f"✗ Invalid split: {split}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    urls = NUSCENES_URLS[split]
    
    print("\n" + "=" * 80)
    print("Using wget for download (faster for large files)")
    print("=" * 80)
    
    # Check if wget is available
    try:
        subprocess.run(['wget', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Error: wget not found. Please install wget or use --method requests")
        return False
    
    all_urls = [urls['metadata']]
    if isinstance(urls['samples'], str):
        all_urls.append(urls['samples'])
    else:
        all_urls.extend(urls['samples'])
    
    for url in all_urls:
        filename = Path(url).name
        output_file = output_dir / filename
        
        print(f"\nDownloading: {filename}")
        
        cmd = [
            'wget',
            '--continue',  # Resume partial downloads
            '--progress=bar:force',
            '--output-document', str(output_file),
            url
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Downloaded: {filename}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error downloading {filename}: {e}")
            return False
    
    print("\n✓ All files downloaded!")
    print("\nTo extract, run:")
    print(f"cd {output_dir} && for f in *.tgz; do tar -xzf $f; done")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download nuScenes dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download mini split (11GB, recommended for testing)
  python scripts/download_nuscenes.py --split mini --output data/nuscenes

  # Download full trainval split (350GB)
  python scripts/download_nuscenes.py --split trainval --output data/nuscenes

  # Download using wget (faster for large files)
  python scripts/download_nuscenes.py --split mini --output data/nuscenes --method wget

  # Download without extraction
  python scripts/download_nuscenes.py --split mini --output data/nuscenes --no-extract

Note:
  You may need to register at https://www.nuscenes.org/nuscenes first.
  Some downloads may require authentication.
        """
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['mini', 'trainval'],
        default='mini',
        help='Dataset split to download (default: mini)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/nuscenes',
        help='Output directory (default: data/nuscenes)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['requests', 'wget'],
        default='requests',
        help='Download method (default: requests)'
    )
    
    parser.add_argument(
        '--no-extract',
        action='store_true',
        help='Do not extract archives after download'
    )
    
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Keep archives after extraction'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    # Show warning for trainval
    if args.split == 'trainval':
        print("\n⚠️  WARNING: You are about to download the FULL dataset (350GB)!")
        print("This will take several hours and requires significant disk space.")
        response = input("Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Download cancelled.")
            return
    
    if args.method == 'wget':
        success = download_via_wget(args.split, output_dir)
    else:
        success = download_nuscenes(
            split=args.split,
            output_dir=output_dir,
            extract=not args.no_extract,
            cleanup=not args.no_cleanup
        )
    
    if success:
        print("\n✓ Success!")
        return 0
    else:
        print("\n✗ Download failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())