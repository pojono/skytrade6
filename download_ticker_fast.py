#!/usr/bin/env python3
"""
Fast bulk download of ticker data using rsync with optimized settings.
Disables encryption for maximum speed on trusted AWS network.
"""
import subprocess
import sys
from pathlib import Path
import time
import os

REMOTE_HOST = "ubuntu@13.251.79.76"
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_remote")
REMOTE_BASE = "~/dataminer/data/archive/raw"
LOCAL_BASE = "data_bybit"

def download_symbol_bulk(symbol, start_date, end_date):
    """Download all data for a symbol using single rsync command."""
    print(f"{'='*80}")
    print(f"🚀 FAST BULK DOWNLOAD: {symbol}")
    print(f"{'='*80}")
    print(f"Remote:     {REMOTE_HOST}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Strategy:   rsync with no encryption (arcfour cipher)")
    print(f"{'='*80}\n")
    
    # Setup local directory
    local_dir = Path(LOCAL_BASE) / symbol / "ticker"
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Build rsync command with optimizations:
    # -a: archive mode (preserves timestamps, etc)
    # -v: verbose
    # -z: compress during transfer (gzip already compressed, but helps with protocol overhead)
    # -P: show progress + allow resume
    # --stats: show transfer statistics
    # -e: specify SSH with optimizations:
    #     -c aes128-gcm@openssh.com: fast modern cipher
    #     -o Compression=no: disable SSH compression (files already .gz)
    #     -x: disable X11 forwarding
    # --include/--exclude: filter patterns to only get SOLUSDT ticker data
    
    # Build include patterns for date range
    # We need to include the directory structure: dt=YYYY-MM-DD/hr=*/...
    start_year, start_month, start_day = start_date.split('-')
    end_year, end_month, end_day = end_date.split('-')
    
    rsync_cmd = [
        'rsync',
        '-avzP',
        '--stats',
        '-e', f'ssh -i {SSH_KEY} -c aes128-gcm@openssh.com -o Compression=no -x',
        '--include', 'dt=2025-*/',  # Include 2025 date directories
        '--include', 'hr=*/',  # Include hour directories
        '--include', 'exchange=bybit/',
        '--include', 'source=rest/',
        '--include', 'market=linear/',
        '--include', 'stream=ticker/',
        '--include', f'symbol={symbol}/',
        '--include', 'data.jsonl.gz',  # Include the actual data files
        '--exclude', '*',  # Exclude everything else
        f'{REMOTE_HOST}:{REMOTE_BASE}/',
        str(local_dir) + '/'
    ]
    
    print("📦 Starting rsync transfer...")
    print(f"Command: {' '.join(rsync_cmd[:3])} ... (optimized SSH settings)\n")
    
    start_time = time.time()
    
    try:
        # Run rsync with real-time output
        process = subprocess.Popen(
            rsync_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"\n{'='*80}")
            print(f"✅ Download completed successfully!")
            print(f"⏱️  Total time: {elapsed:.1f} seconds")
            print(f"📁 Data saved to: {local_dir}")
            print(f"{'='*80}")
            return 0
        else:
            print(f"\n❌ rsync failed with exit code {process.returncode}")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        process.kill()
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

def download_symbol_bulk_simple(symbol, start_date, end_date):
    """Simpler approach: use wildcard rsync to grab all matching files."""
    print(f"{'='*80}")
    print(f"🚀 FAST BULK DOWNLOAD: {symbol}")
    print(f"{'='*80}")
    print(f"Remote:     {REMOTE_HOST}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Strategy:   rsync bulk transfer (no encryption)")
    print(f"{'='*80}\n")
    
    # Setup local directory
    local_dir = Path(LOCAL_BASE) / symbol / "ticker"
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Use rsync with relative paths to preserve structure
    # This will download all files matching the pattern and preserve directory structure
    rsync_cmd = f"""rsync -avzP --stats \
        -e 'ssh -i {SSH_KEY} -c aes128-gcm@openssh.com -o Compression=no -x' \
        --relative \
        {REMOTE_HOST}:./dataminer/data/archive/raw/dt=2025-*/hr=*/exchange=bybit/source=rest/market=linear/stream=ticker/symbol={symbol}/data.jsonl.gz \
        {local_dir}/
    """
    
    print("📦 Starting rsync transfer...")
    print("⚡ Using aes128-gcm cipher (fast modern encryption) + no compression\n")
    
    start_time = time.time()
    result = subprocess.run(rsync_cmd, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n{'='*80}")
        print(f"✅ Download completed successfully!")
        print(f"⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"📁 Data saved to: {local_dir}")
        print(f"{'='*80}")
        
        # Count downloaded files
        files = list(local_dir.rglob("data.jsonl.gz"))
        print(f"📊 Total files: {len(files)}")
        total_size = sum(f.stat().st_size for f in files)
        print(f"📦 Total size: {total_size / 1024 / 1024:.1f} MB")
        if elapsed > 0:
            print(f"⚡ Speed: {total_size / 1024 / 1024 / elapsed:.1f} MB/s")
        print(f"{'='*80}")
        return 0
    else:
        print(f"\n❌ rsync failed with exit code {result.returncode}")
        return 1

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fast bulk download of Bybit ticker data')
    parser.add_argument('--symbol', type=str, default='SOLUSDT', help='Symbol to download')
    parser.add_argument('--start-date', type=str, default='2025-05-11', help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, default='2025-08-10', help='End date YYYY-MM-DD')
    args = parser.parse_args()
    
    # Verify SSH key exists
    if not Path(SSH_KEY).exists():
        print(f"❌ ERROR: SSH key not found at {SSH_KEY}")
        return 1
    
    return download_symbol_bulk_simple(args.symbol, args.start_date, args.end_date)

if __name__ == "__main__":
    sys.exit(main())
