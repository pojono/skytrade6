#!/usr/bin/env python3
"""
Download ticker data from remote dataminer server with progress tracking and ETA.
Organizes data into data_bybit/SYMBOL/ticker/ directory structure.
"""
import subprocess
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import gzip
import json
import time

REMOTE_HOST = "ubuntu@13.251.79.76"
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_remote")
REMOTE_BASE = "~/dataminer/data/archive/raw"
LOCAL_BASE = "data_bybit"

def get_2025_dates():
    """Get all dates in 2025 (Jan 1 - Dec 31)."""
    start = datetime(2025, 1, 1)
    end = datetime(2025, 12, 31)
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates

def check_remote_files_batch(date, symbol):
    """Check which hour files exist on remote server for a given date (batch check)."""
    # Build a single SSH command to check all 24 hours at once
    checks = []
    for hour in range(24):
        remote_path = f"{REMOTE_BASE}/dt={date}/hr={hour:02d}/exchange=bybit/source=rest/market=linear/stream=ticker/symbol={symbol}/data.jsonl.gz"
        checks.append(f"test -f {remote_path} && echo {hour}")
    
    cmd = f"ssh -i {SSH_KEY} {REMOTE_HOST} '{'; '.join(checks)}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    
    # Parse which hours exist
    existing_hours = set()
    if result.returncode == 0 and result.stdout.strip():
        for line in result.stdout.strip().split('\n'):
            if line.strip().isdigit():
                existing_hours.add(int(line.strip()))
    
    return existing_hours

def download_file(date, hour, symbol, local_dir):
    """Download a single ticker file from remote server."""
    remote_path = f"{REMOTE_BASE}/dt={date}/hr={hour:02d}/exchange=bybit/source=rest/market=linear/stream=ticker/symbol={symbol}/data.jsonl.gz"
    local_file = local_dir / f"ticker_{date}_hr{hour:02d}.jsonl.gz"
    
    # Skip if already downloaded and valid
    if local_file.exists():
        if verify_file(local_file):
            return 'exists', local_file
        else:
            local_file.unlink()  # Remove corrupted file
    
    # Download using scp with compression
    cmd = f"scp -C -i {SSH_KEY} {REMOTE_HOST}:{remote_path} {local_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    
    if result.returncode == 0:
        if verify_file(local_file):
            return 'success', local_file
        else:
            local_file.unlink()
            return 'corrupt', None
    else:
        return 'failed', None

def verify_file(filepath):
    """Verify downloaded file is valid gzipped JSON."""
    try:
        with gzip.open(filepath, 'rt') as f:
            first_line = f.readline()
            if not first_line:
                return False
            json.loads(first_line)
        return True
    except Exception:
        return False

def format_eta(seconds):
    """Format seconds into human-readable ETA."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m {int(seconds%60)}s"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"

def download_symbol_data(symbol, start_date=None, end_date=None):
    """Download all 2025 data for a symbol with progress tracking."""
    print(f"\n{'='*80}")
    print(f"📥 Downloading {symbol} ticker data from Bybit")
    print(f"{'='*80}")
    
    # Setup local directory structure
    local_dir = Path(LOCAL_BASE) / symbol / "ticker"
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Get date range
    all_dates = get_2025_dates()
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]
    
    print(f"📅 Date range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")
    print(f"📁 Local directory: {local_dir}")
    print(f"{'='*80}\n")
    
    # Statistics
    stats = {
        'total_files': 0,
        'already_exists': 0,
        'downloaded': 0,
        'failed': 0,
        'corrupt': 0,
        'total_bytes': 0
    }
    
    start_time = time.time()
    files_processed = 0
    
    # Process each date
    for date_idx, date in enumerate(all_dates, 1):
        # Batch check which hours exist on remote
        existing_hours = check_remote_files_batch(date, symbol)
        
        if not existing_hours:
            print(f"[{date_idx:3d}/{len(all_dates)}] {date}: No data available")
            continue
        
        stats['total_files'] += len(existing_hours)
        date_downloaded = 0
        date_exists = 0
        date_failed = 0
        
        # Download each hour file
        for hour in sorted(existing_hours):
            status, local_file = download_file(date, hour, symbol, local_dir)
            files_processed += 1
            
            if status == 'exists':
                stats['already_exists'] += 1
                date_exists += 1
                if local_file:
                    stats['total_bytes'] += local_file.stat().st_size
            elif status == 'success':
                stats['downloaded'] += 1
                date_downloaded += 1
                if local_file:
                    stats['total_bytes'] += local_file.stat().st_size
            elif status == 'corrupt':
                stats['corrupt'] += 1
                date_failed += 1
            else:
                stats['failed'] += 1
                date_failed += 1
            
            # Calculate ETA
            elapsed = time.time() - start_time
            if files_processed > 0:
                avg_time_per_file = elapsed / files_processed
                remaining_files = stats['total_files'] - files_processed
                eta_seconds = avg_time_per_file * remaining_files
                eta_str = format_eta(eta_seconds)
            else:
                eta_str = "calculating..."
            
            # Progress bar
            progress = files_processed / stats['total_files'] * 100 if stats['total_files'] > 0 else 0
            bar_length = 30
            filled = int(bar_length * progress / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # Print progress line (overwrite same line)
            sys.stdout.write(f"\r[{date_idx:3d}/{len(all_dates)}] {date} | "
                           f"{bar} {progress:5.1f}% | "
                           f"Files: {files_processed}/{stats['total_files']} | "
                           f"ETA: {eta_str}     ")
            sys.stdout.flush()
        
        # Print date summary
        if date_downloaded > 0 or date_exists > 0:
            print(f"  ✓ {date_downloaded} new, {date_exists} cached, {date_failed} failed")
        elif date_failed > 0:
            print(f"  ✗ {date_failed} failed")
    
    # Final summary
    elapsed_total = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"📊 Summary for {symbol}")
    print(f"{'='*80}")
    print(f"  Total files found:      {stats['total_files']}")
    print(f"  Already cached:         {stats['already_exists']}")
    print(f"  Newly downloaded:       {stats['downloaded']}")
    print(f"  Failed:                 {stats['failed']}")
    print(f"  Corrupted:              {stats['corrupt']}")
    print(f"  Total data size:        {stats['total_bytes'] / 1024 / 1024:.1f} MB")
    print(f"  Time elapsed:           {format_eta(elapsed_total)}")
    if stats['downloaded'] > 0:
        print(f"  Avg download speed:     {stats['downloaded'] / elapsed_total:.1f} files/sec")
    print(f"{'='*80}")
    
    return stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download Bybit ticker data from remote dataminer')
    parser.add_argument('--symbol', type=str, default='SOLUSDT', help='Symbol to download (default: SOLUSDT)')
    parser.add_argument('--start-date', type=str, help='Start date YYYY-MM-DD (default: 2025-01-01)')
    parser.add_argument('--end-date', type=str, help='End date YYYY-MM-DD (default: 2025-12-31)')
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 BYBIT TICKER DATA DOWNLOADER")
    print("="*80)
    print(f"Remote server:  {REMOTE_HOST}")
    print(f"Symbol:         {args.symbol}")
    print(f"Year:           2025")
    print(f"Local base:     {LOCAL_BASE}/")
    print("="*80)
    
    # Verify SSH key exists
    if not Path(SSH_KEY).exists():
        print(f"❌ ERROR: SSH key not found at {SSH_KEY}")
        print("Please ensure the SSH key is in place before running.")
        return 1
    
    # Download data
    stats = download_symbol_data(args.symbol, args.start_date, args.end_date)
    
    # Final status
    if stats['failed'] == 0 and stats['corrupt'] == 0:
        print("\n✅ Download completed successfully!")
    elif stats['downloaded'] > 0:
        print(f"\n⚠️  Download completed with {stats['failed'] + stats['corrupt']} errors")
    else:
        print("\n❌ Download failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
