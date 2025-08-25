import pickle
import json
import sys

def pickle_to_json(pickle_file, json_file):
    # Load pickle
    with open(pickle_file, "rb") as pf:
        data = pickle.load(pf)

    # Save as JSON
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(data, jf, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pickle_to_json.py input.pkl output.json")
    else:
        pickle_to_json(sys.argv[1], sys.argv[2])
        print(f"Converted {sys.argv[1]} -> {sys.argv[2]}")