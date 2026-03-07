import pandas as pd
import numpy as np

# We're going to calculate exactly how sensitive this strategy is to:
# 1. 50ms latency delay (Does the alpha disappear if we are 50ms late?)
# We can estimate this by looking at how much the price moves in the next 1 second.
# But we only have 1m data for forward returns. 
# We can reason about the structural risks based on what we've seen.
