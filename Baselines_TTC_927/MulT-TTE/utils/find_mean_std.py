import pandas as pd
import numpy as np

# Load your training data
df = pd.read_csv("../data/toronto/train.csv")
data = df.to_numpy(dtype=np.float32)

# Compute target travel times based on your collate_func logic
total_times = []
for row in data:
    start_seg = int(row[55])
    # Sum valid segment times from cols 9 to 17
    valid_seg_times = row[9:18][start_seg - 1 : 9]
    total_times.append(np.sum(valid_seg_times))

# Calculate mean and std
time_mean = float(np.mean(total_times))
time_std = float(np.std(total_times))

print(f'"time_mean": {time_mean:.2f},')
print(f'"time_std": {time_std:.2f},')