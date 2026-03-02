#!/usr/bin/env python3
"""
Download orderbook snapshot data for Flow Impact calculation.
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

def download_orderbook_data(symbol, start_date, end_date):
    """Download WS orderbook data using optimized rsync."""
    print(f"{'='*80}")
    print(f"🚀 DOWNLOADING ORDERBOOK DATA: {symbol}")
    print(f"{'='*80}")
    print(f"Remote:     {REMOTE_HOST}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Stream:     WebSocket orderbook snapshots")
    print(f"{'='*80}\n")
    
    local_dir = Path(LOCAL_BASE) / symbol / "orderbook"
    local_dir.mkdir(parents=True, exist_ok=True)
    
    rsync_cmd = f"""rsync -avzP --stats \
        -e 'ssh -i {SSH_KEY} -c aes128-gcm@openssh.com -o Compression=no -x' \
        --relative \
        {REMOTE_HOST}:./dataminer/data/archive/raw/dt=2025-*/hr=*/exchange=bybit/source=ws/market=linear/stream=orderbook/symbol={symbol}/data.jsonl.gz \
        {local_dir}/
    """
    
    print("📦 Starting rsync transfer...")
    print("⚡ Optimized for speed\n")
    
    start_time = time.time()
    result = subprocess.run(rsync_cmd, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n{'='*80}")
        print(f"✅ Download completed successfully!")
        print(f"⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"📁 Data saved to: {local_dir}")
        print(f"{'='*80}")
        
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
    parser = argparse.ArgumentParser(description='Download Bybit WS orderbook data')
    parser.add_argument('--symbol', type=str, default='SOLUSDT')
    parser.add_argument('--start-date', type=str, default='2025-05-11')
    parser.add_argument('--end-date', type=str, default='2025-08-10')
    args = parser.parse_args()
    
    if not Path(SSH_KEY).exists():
        print(f"❌ ERROR: SSH key not found at {SSH_KEY}")
        return 1
    
    return download_orderbook_data(args.symbol, args.start_date, args.end_date)

if __name__ == "__main__":
    sys.exit(main())
