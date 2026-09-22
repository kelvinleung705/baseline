import numpy as np
import os
from pprint import pprint  # Makes printing dictionaries/lists much cleaner

# ==========================================
# 1. Configure the path to your .npy file
# ==========================================
file_path = "train.npy"  # <-- Replace with your actual file path


# Example using os.path.join like in your pipeline:
# file_path = os.path.join("data_dir", "train.npy")


def load_and_print_npy(path):
    if not os.path.exists(path):
        print(f"Error: File not found at '{path}'")
        return

    print(f"Loading: {path}\n" + "=" * 40)

    # 2. Load the .npy file using allow_pickle=True
    tdata = np.load(path, allow_pickle=True)

    # 3. Handle 0-D array wrapper (common when saving dicts/objects into .npy)
    if isinstance(tdata, np.ndarray) and tdata.ndim == 0:
        tdata = tdata.item()

    # 4. Print Metadata / Information
    print("=== DATA METADATA ===")
    print(f"Type: {type(tdata)}")

    if isinstance(tdata, np.ndarray):
        print(f"Shape: {tdata.shape}")
        print(f"Dtype: {tdata.dtype}")
    elif isinstance(tdata, dict):
        print(f"Dictionary Keys ({len(tdata.keys())}): {list(tdata.keys())[:10]}")  # Prints up to first 10 keys

    print("\n=== DATA CONTENTS (tdata) ===")

    # 5. Pretty Print the contents
    if isinstance(tdata, dict):
        # Use pprint for nice, readable dictionary output
        pprint(tdata)
    else:
        print(tdata)

    return tdata


# Run the function
if __name__ == "__main__":
    tdata = load_and_print_npy(file_path)