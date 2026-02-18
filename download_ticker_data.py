#!/usr/bin/env python3
"""
Download ticker data from remote dataminer server.
Organizes data into data/SYMBOL/ directory structure.
"""
import subprocess
import os
from datetime import datetime, timedelta
from pathlib import Path
import gzip
import json

REMOTE_HOST = "ubuntu@13.251.79.76"
SSH_KEY = "~/.ssh/id_ed25519_remote"
REMOTE_BASE = "~/dataminer/data/archive/raw"
LOCAL_BASE = "data"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]

def get_2025_dates():
    """Get all dates in 2025 that have data (May 11 - Aug 10)."""
    start = datetime(2025, 5, 11)
    end = datetime(2025, 8, 10)
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates

def check_remote_file_exists(date, hour, symbol):
    """Check if a file exists on remote server."""
    remote_path = f"{REMOTE_BASE}/dt={date}/hr={hour:02d}/exchange=bybit/source=rest/market=linear/stream=ticker/symbol={symbol}/data.jsonl.gz"
    cmd = f"ssh -i {SSH_KEY} {REMOTE_HOST} 'test -f {remote_path} && echo exists'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip() == "exists"

def download_file(date, hour, symbol, local_dir):
    """Download a single ticker file from remote server."""
    remote_path = f"{REMOTE_BASE}/dt={date}/hr={hour:02d}/exchange=bybit/source=rest/market=linear/stream=ticker/symbol={symbol}/data.jsonl.gz"
    local_file = local_dir / f"ticker_{date}_hr{hour:02d}.jsonl.gz"
    
    # Skip if already downloaded
    if local_file.exists():
        print(f"  ✓ Already exists: {local_file.name}")
        return True
    
    # Download using scp
    cmd = f"scp -i {SSH_KEY} {REMOTE_HOST}:{remote_path} {local_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✓ Downloaded: {local_file.name}")
        return True
    else:
        print(f"  ✗ Failed: {local_file.name} - {result.stderr.strip()}")
        return False

def verify_file(filepath):
    """Verify downloaded file is valid gzipped JSON."""
    try:
        with gzip.open(filepath, 'rt') as f:
            # Try to read first line
            first_line = f.readline()
            json.loads(first_line)
        return True
    except Exception as e:
        print(f"  ✗ Verification failed: {filepath.name} - {e}")
        return False

def download_symbol_data(symbol):
    """Download all 2025 data for a symbol."""
    print(f"\n{'='*70}")
    print(f"Downloading {symbol}")
    print(f"{'='*70}")
    
    local_dir = Path(LOCAL_BASE) / symbol
    local_dir.mkdir(parents=True, exist_ok=True)
    
    dates = get_2025_dates()
    total_files = 0
    downloaded_files = 0
    failed_files = 0
    
    for date in dates:
        print(f"\n[{date}] Processing...")
        date_files = 0
        
        for hour in range(24):
            # Check if file exists on remote
            if not check_remote_file_exists(date, hour, symbol):
                continue
            
            total_files += 1
            if download_file(date, hour, symbol, local_dir):
                # Verify the downloaded file
                local_file = local_dir / f"ticker_{date}_hr{hour:02d}.jsonl.gz"
                if verify_file(local_file):
                    downloaded_files += 1
                    date_files += 1
                else:
                    failed_files += 1
                    local_file.unlink()  # Remove corrupted file
            else:
                failed_files += 1
        
        if date_files > 0:
            print(f"  → Downloaded {date_files} files for {date}")
    
    print(f"\n{'='*70}")
    print(f"Summary for {symbol}:")
    print(f"  Total files found: {total_files}")
    print(f"  Successfully downloaded: {downloaded_files}")
    print(f"  Failed: {failed_files}")
    print(f"{'='*70}")
    
    return downloaded_files, failed_files

def main():
    print("="*70)
    print("TICKER DATA DOWNLOAD FROM REMOTE DATAMINER SERVER")
    print("="*70)
    print(f"Remote: {REMOTE_HOST}")
    print(f"Period: 2025-05-11 to 2025-08-10 (92 days)")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Local directory: {LOCAL_BASE}/")
    print("="*70)
    
    total_downloaded = 0
    total_failed = 0
    
    for symbol in SYMBOLS:
        downloaded, failed = download_symbol_data(symbol)
        total_downloaded += downloaded
        total_failed += failed
    
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"Total files downloaded: {total_downloaded}")
    print(f"Total files failed: {total_failed}")
    print(f"Success rate: {100*total_downloaded/(total_downloaded+total_failed):.1f}%")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
