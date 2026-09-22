import json

# Replace 'data.json' with your actual file path
with open('output.json', 'r') as file:
    data = json.load(file)

def print_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(key)
            # Recursively check if the value is another dictionary
            print_keys(value)
    elif isinstance(obj, list):
        # Recursively check inside lists (in case they contain dictionaries)
        for item in obj:
            print_keys(item)

print_keys(data)