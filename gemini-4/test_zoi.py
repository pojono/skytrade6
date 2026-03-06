from extract_zoi_events import load_data, identify_swings
import pandas as pd

df = load_data('BTCUSDT')
swings = identify_swings(df, 0.01)

zoi_events = []
    
for i in range(len(swings) - 1):
    A = swings[i]
    B = swings[i+1]
    
    swing_size = abs(B['price'] - A['price'])
    if swing_size == 0 or swing_size / A['price'] < 0.01:
        continue
        
    trend = 'up' if A['type'] == 'low' else 'down'
    
    # Define the ZOI
    if trend == 'up':
        zoi_upper = B['price'] - (swing_size * 0.58)
        zoi_lower = B['price'] - (swing_size * 0.65)
        invalid_price = A['price'] # Went below origin
    else:
        zoi_lower = B['price'] + (swing_size * 0.58)
        zoi_upper = B['price'] + (swing_size * 0.65)
        invalid_price = A['price'] # Went above origin
        
    post_b_df = df.loc[B['time']:]
    
    entered_zoi = False
    
    for idx, row in post_b_df.iterrows():
        if idx == B['time']:
            continue
            
        # Check ZOI entry FIRST
        if not entered_zoi:
            if trend == 'up' and row['low'] <= zoi_upper:
                entered_zoi = True
            elif trend == 'down' and row['high'] >= zoi_lower:
                entered_zoi = True
                
        # If we invalidate the setup before hitting ZOI, break
        if not entered_zoi:
            if trend == 'up' and (row['low'] <= invalid_price or row['high'] > B['price']):
                break
            if trend == 'down' and (row['high'] >= invalid_price or row['low'] < B['price']):
                break
        else:
            # Determine success or failure after entering ZOI
            if trend == 'up':
                if row['high'] > B['price']:
                    zoi_events.append({'success': True})
                    break
                elif row['low'] < invalid_price:
                    zoi_events.append({'success': False})
                    break
            else:
                if row['low'] < B['price']:
                    zoi_events.append({'success': True})
                    break
                elif row['high'] > invalid_price:
                    zoi_events.append({'success': False})
                    break

print("Found", len(zoi_events), "events")
