import numpy as np
import pandas as pd

train = pd.read_csv("samples/train_trips.csv").to_numpy().astype(np.float32)
test = pd.read_csv("samples/test_trips.csv").to_numpy().astype(np.float32)

print("1. Total columns:", "Train =", train.shape[1], "| Test =", test.shape[1])
print(
    "2. Col 55 unique values in Train:",
    np.unique(train[:, 55])[:10],
    "| Test:",
    np.unique(test[:, 55])[:10],
)
print(
    "3. Sum of ALL 9 segments (cols 9:18):",
    "Train =",
    train[:, 9:18].sum(axis=1).mean(),
    "| Test =",
    test[:, 9:18].sum(axis=1).mean(),
)