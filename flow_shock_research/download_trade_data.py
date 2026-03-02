#!/usr/bin/env python3
"""
Download WebSocket trade data for flow shock analysis.
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

def download_trade_data(symbol, start_date, end_date):
    """Download WS trade data using optimized rsync."""
    print(f"{'='*80}")
    print(f"🚀 DOWNLOADING TRADE DATA: {symbol}")
    print(f"{'='*80}")
    print(f"Remote:     {REMOTE_HOST}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Stream:     WebSocket trade (tick-by-tick)")
    print(f"{'='*80}\n")
    
    # Setup local directory
    local_dir = Path(LOCAL_BASE) / symbol / "trade"
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Download WS trade data
    rsync_cmd = f"""rsync -avzP --stats \
        -e 'ssh -i {SSH_KEY} -c aes128-gcm@openssh.com -o Compression=no -x' \
        --relative \
        {REMOTE_HOST}:./dataminer/data/archive/raw/dt=2025-*/hr=*/exchange=bybit/source=ws/market=linear/stream=trade/symbol={symbol}/data.jsonl.gz \
        {local_dir}/
    """
    
    print("📦 Starting rsync transfer...")
    print("⚡ Optimized for speed (fast cipher + no compression)\n")
    
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
    parser = argparse.ArgumentParser(description='Download Bybit WS trade data')
    parser.add_argument('--symbol', type=str, default='SOLUSDT', help='Symbol to download')
    parser.add_argument('--start-date', type=str, default='2025-05-11', help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, default='2025-08-10', help='End date YYYY-MM-DD')
    args = parser.parse_args()
    
    # Verify SSH key exists
    if not Path(SSH_KEY).exists():
        print(f"❌ ERROR: SSH key not found at {SSH_KEY}")
        return 1
    
    return download_trade_data(args.symbol, args.start_date, args.end_date)

if __name__ == "__main__":
    sys.exit(main())
