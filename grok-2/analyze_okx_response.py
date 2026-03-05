#!/usr/bin/env python3

"""
Script to analyze the OKX API response for instruments to debug field names and structure.
"""

import json

# Load the saved response
with open('/home/ubuntu/Projects/skytrade6/grok-2/okx_instruments.json', 'r') as f:
    data = json.load(f)

# Check response metadata
print('Response Code:', data.get('code', 'N/A'))
print('Response Message:', data.get('msg', 'N/A'))

# Check data array
_data = data.get('data', [])
print('Data Length:', len(_data))

# Inspect the first few items for field names and values
for i, item in enumerate(_data[:5]):
    print(f'\nItem {i + 1}:')
    for key, value in item.items():
        print(f'  {key}: {value}')

# Count items matching SWAP and USDT criteria
swap_usdt_count = 0
for item in _data:
    if item.get('instType') == 'SWAP' and item.get('quoteCcy') == 'USDT':
        swap_usdt_count += 1
        if swap_usdt_count <= 5:  # Print first few matches
            print(f'Matched SWAP USDT Item: {item.get("instId", "Unknown")}')

print(f'Total SWAP items with quoteCcy=USDT: {swap_usdt_count}')
