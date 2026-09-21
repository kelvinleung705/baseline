import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Paths (adjust folder path to your project's data directory, e.g., 'data/toronto')
DATA_DIR = "data/toronto_927_2025_Jul_Dec"

train_path = os.path.join(DATA_DIR, "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_Jan_Jun.csv")
test_path = os.path.join(DATA_DIR, "trip_info_9_section_ver2_simplify_ultra_no_variance_2025_Jul_Dec.csv")

# Output paths
new_train_path = os.path.join(DATA_DIR, "train.csv")
new_val_path = os.path.join(DATA_DIR, "val.csv")
new_test_path = os.path.join(DATA_DIR, "test.csv")  # Remains untouched

# 2. Load the original training data
print(f"Loading {train_path}...")
df_train = pd.read_csv(train_path)
print(f"Original train size: {len(df_train)} rows")

# 3. Split train into Train (90%) and Val (10%)
# Use val_size=0.10 or 0.15 depending on dataset size
# random_state ensures reproducibility
train_df, val_df = train_test_split(
    df_train, 
    test_size=0.10, 
    random_state=42, 
    shuffle=True
)

print(f"New Train size: {len(train_df)} rows")
print(f"Validation size: {len(val_df)} rows")

# 4. Save to CSV
train_df.to_csv(new_train_path, index=False)
val_df.to_csv(new_val_path, index=False)

print("Split completed successfully!")