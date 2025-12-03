# ================================================
# To move "deferred" decisioned directories
# ================================================

import os
import json
import shutil
import logging

def move_deferred_subdirectories(results_file_path, base_directory):
    """
    Moves all subdirectories with a decision of "deferred" to a parent directory called "deferred".
    
    Args:
        results_file_path (str): Path to the aggregated_results.json file.
        base_directory (str): The base directory where subdirectories are located.

    Returns:
        None
    """
    # Load the results from the JSON file
    if not os.path.exists(results_file_path):
        raise FileNotFoundError(f"Results file not found: {results_file_path}")
    
    with open(results_file_path, 'r') as f:
        results = json.load(f)

    # Move the "deferred" directory one level higher
    parent_directory = os.path.abspath(os.path.join(base_directory, os.pardir))  # Parent directory of base_directory
    deferred_directory = os.path.join(parent_directory, "deferred")  # Create "deferred" in parent directory
    os.makedirs(deferred_directory, exist_ok=True)
    
    # Process each result
    for result in results:
        subdir_name = result['subdirectory']
        decision = result.get('decision', '').strip().lower()  # Normalize decision text

        if decision == "deferred":
            # Construct full path to the subdirectory
            subdir_path = os.path.join(base_directory, subdir_name)
            
            if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                # Move the deferred subdirectory
                destination_path = os.path.join(deferred_directory, subdir_name)
                shutil.move(subdir_path, destination_path)
                print(f"Moved '{subdir_name}' to '{deferred_directory}'.")
            else:
                print(f"Subdirectory not found or invalid: {subdir_path}")

    print("\n✅ All deferred subdirectories have been processed.")


# ================================================
# To change the "triggered" decisioned file names
# ================================================


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def rename_files_for_triggered_subdirectories(json_path):
    """
    Reads a JSON file, processes multiple subdirectories, and renames files in subdirectories
    where the decision is 'triggered'. The first part of the filenames is enclosed in square brackets.
    Skips files that already contain '[' in their names.
    
    Args:
        json_path (str): Path to the JSON file.
    """
    # Load the JSON data
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, "r") as f:
        data = json.load(f)

    # Process each entry in the JSON file
    for entry in data:
        subdir = entry.get("subdirectory")
        decision = entry.get("decision", "").strip().lower()

        # Skip if the decision is not 'triggered'
        if decision != "triggered":
            logging.info(f"Skipping subdirectory '{subdir}' (decision: {decision}).")
            continue

        # Process the files in the subdirectory
        logging.info(f"Processing subdirectory '{subdir}' (decision: triggered).")
        for metric in entry.get("metrics_list", []):
            file_path = metric.get("file")

            if not file_path or not os.path.exists(file_path):
                logging.warning(f"File not found or invalid: {file_path}")
                continue

            # Extract directory and filename
            directory, original_name = os.path.split(file_path)

            # Skip if the file name already contains '['
            if "[" in original_name:
                logging.info(f"Skipping file '{original_name}' (already renamed).")
                continue

            # Split the filename to isolate the first part
            if "-" in original_name:
                # Handle cases like "xxxx-yyyyy-zzzz.npy"
                parts = original_name.split("-")
                parts[0] = f"[{parts[0]}]"
                new_name = "-".join(parts)
            else:
                # Handle cases with no dash, like "00001.npy" or "00006_1.flac"
                name_part, extension = os.path.splitext(original_name)
                new_name = f"[{name_part}]{extension}"

            # Rename the file
            new_path = os.path.join(directory, new_name)
            os.rename(file_path, new_path)
            logging.info(f"Renamed: {file_path} -> {new_path}")

    logging.info("✅ All triggered subdirectory files have been renamed.")

# ================================================
# Main Script
# ================================================
if __name__ == "__main__":
    # Specify the path to the aggregated_results.json file
    results_file = "aggregated_results.json"

    # Specify the base directory where subdirectories are located
    # base_dir = "C:/Users/s222343272/Downloads/datasets/test_test/" 
    base_dir = "../data/sample_dataset/voxceleb_data/vox_3sec/" 

    try:
        move_deferred_subdirectories(results_file, base_dir)
        rename_files_for_triggered_subdirectories(results_file)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
