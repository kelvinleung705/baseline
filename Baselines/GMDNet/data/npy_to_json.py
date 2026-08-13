import json
import numpy as np
import os

# =========================================================
# Custom JSON Encoder to handle NumPy types (arrays, floats, ints)
# =========================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to standard python lists
        if isinstance(obj, np.integer):
            return int(obj)      # Convert np.int32/64 to python int
        if isinstance(obj, np.floating):
            return float(obj)    # Convert np.float32/64 to python float
        if isinstance(obj, np.bool_):
            return bool(obj)     # Convert np.bool_ to python bool
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')  # Convert bytes to string
        return super(NumpyEncoder, self).default(obj)


def npy_to_json(npy_path, json_output_path=None):
    if not os.path.exists(npy_path):
        print(f"Error: File not found at '{npy_path}'")
        return

    # 1. Load the .npy file
    tdata = np.load(npy_path, allow_pickle=True)

    # 2. Unwrap 0-D array if it contains a dictionary or object
    if isinstance(tdata, np.ndarray) and tdata.ndim == 0:
        tdata = tdata.item()

    # 3. Convert tdata to a formatted JSON string
    json_string = json.dumps(tdata, cls=NumpyEncoder, indent=4)

    # 4. Print JSON output to the terminal
    print("=== JSON OUTPUT ===")
    print(json_string)

    # 5. (Optional) Save JSON to a file
    if json_output_path:
        with open(json_output_path, "w", encoding="utf-8") as f:
            f.write(json_string)
        print(f"\nSuccessfully saved JSON to: '{json_output_path}'")

    return json_string


# =========================================================
# Example Usage
# =========================================================
if __name__ == "__main__":
    input_npy = "train.npy"   # <-- Replace with your input .npy file path
    output_json = "output.json"            # <-- Path where JSON file will be saved (or set to None)

    npy_to_json(input_npy, output_json)