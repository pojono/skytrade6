import glob
import os
import re

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
symbol = "BTCUSDT"

def get_dates(symbol, is_spot=False):
    pattern = f"{DATALAKE_DIR}/{symbol}/*_kline_1m{'_spot' if is_spot else ''}.csv"
    files = glob.glob(pattern)
    dates = []
    for f in files:
        basename = os.path.basename(f)
        match = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
        if match:
            dates.append(match.group(1))
    return sorted(list(set(dates)))

fut_dates = get_dates(symbol, is_spot=False)
spot_dates = get_dates(symbol, is_spot=True)

print(f"Futures dates: {len(fut_dates)} days. First: {fut_dates[0] if fut_dates else None}, Last: {fut_dates[-1] if fut_dates else None}")
print(f"Spot dates: {len(spot_dates)} days. First: {spot_dates[0] if spot_dates else None}, Last: {spot_dates[-1] if spot_dates else None}")

common_dates = sorted(list(set(fut_dates).intersection(set(spot_dates))))
print(f"Common dates: {len(common_dates)} days. First: {common_dates[0] if common_dates else None}, Last: {common_dates[-1] if common_dates else None}")

