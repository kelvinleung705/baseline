import pandas as pd
import numpy as np

# Load train_trips.csv
df = pd.read_csv("D40_2023_5-12.csv", header=None) # adjust path if needed

# 1. Check trips where total travel time (columns 9:22) is 0
times_sum = df.iloc[:, 9:22].sum(axis=1)
zero_time_trips = df[times_sum <= 0]
print(f"Number of trips with 0 travel time: {len(zero_time_trips)}")
if len(zero_time_trips) > 0:
    print("Row indices:", zero_time_trips.index.tolist())

# 2. Check column 75 (start segment)
start_segs = df.iloc[:, 75]
bad_start_segs = df[(start_segs < 1) | (start_segs >= 13)]
print(f"Number of trips with invalid start segment: {len(bad_start_segs)}")
if len(bad_start_segs) > 0:
    print("Row indices:", bad_start_segs.index.tolist())