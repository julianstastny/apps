import json
import glob
import os
import argparse

def combine_results(directory):
    combined = {}
    
    # Find all result files
    result_files = glob.glob(os.path.join(directory, "*_results.json"))
    
    # Combine them
    for file in result_files:
        with open(file, 'r') as f:
            results = json.load(f)
            combined.update(results)
    
    # Save combined results
    with open(os.path.join(directory, "all_results.json"), 'w') as f:
        json.dump(combined, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine JSON result files from a directory')
    parser.add_argument('directory', type=str, help='Directory containing the result files')
    args = parser.parse_args()
    
    combine_results(args.directory)