import os
import json
from typing import List, Dict

# Parameters (set these directly in the script)
# DIRECTORY = "C:/Users/s222343272/Downloads/datasets/test_test/"  # Replace with your target directory
DIRECTORY = "../data/sample_dataset/voxceleb_data/test/vox_3sec_test"  # Replace with your target directory
PATTERN = "11111"                     # Replace with the pattern to match
REVERSE = False                       # Set to True to reverse changes, False to apply changes

def validate_directory(directory: str) -> None:
    """Validate that the directory exists."""
    if not os.path.isdir(directory):
        raise ValueError(f"Directory does not exist: {directory}")

def process_file(root: str, filename: str, pattern: str) -> Dict[str, str] or None:
    """Process a file to determine if it needs renaming."""
    original_path = os.path.abspath(os.path.join(root, filename))
    basename, ext = os.path.splitext(filename)
    parts = basename.split('-')
    
    if len(parts) < 2:
        return None
    
    first_part, second_part = parts[0], parts[1]
    
    if (second_part == pattern and 
        not (first_part.startswith('[') and first_part.endswith(']'))):
        new_first = f'[{first_part}]'
        new_basename = '-'.join([new_first] + parts[1:])
        new_filename = f"{new_basename}{ext}"
        new_path = os.path.abspath(os.path.join(root, new_filename))
        
        if new_path == original_path:
            return None
        
        return {'original_path': original_path, 'new_path': new_path}
    return None

def apply_changes(directory: str, pattern: str) -> None:
    """Apply renaming changes to files matching the pattern."""
    log_path = os.path.join(directory, '.rename_log.json')
    log_entries: List[Dict[str, str]] = []
    
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                log_entries = json.load(f)
        except (json.JSONDecodeError, IOError):
            print("Warning: Existing log file is corrupted or unreadable. Starting fresh.")
    
    new_entries: List[Dict[str, str]] = []
    
    for root, _, files in os.walk(directory):
        for filename in files:
            rename_entry = process_file(root, filename, pattern)
            if rename_entry:
                original = rename_entry['original_path']
                new = rename_entry['new_path']
                
                if os.path.exists(new):
                    print(f"Warning: Target path exists, skipping: {new}")
                    continue
                
                try:
                    os.rename(original, new)
                    new_entries.append(rename_entry)
                    print(f"Renamed: {original} -> {new}")
                except OSError as e:
                    print(f"Error renaming {original}: {str(e)}")
    
    log_entries += new_entries
    
    try:
        with open(log_path, 'w') as f:
            json.dump(log_entries, f, indent=2)
        print(f"Log file saved to: {log_path}")
    except IOError as e:
        print(f"Error saving log file: {str(e)}")

def reverse_changes(directory: str) -> None:
    """Reverse renaming changes using the log file."""
    log_path = os.path.join(directory, '.rename_log.json')
    
    if not os.path.exists(log_path):
        print("Error: No log file found. Cannot reverse changes.")
        return
    
    try:
        with open(log_path, 'r') as f:
            log_entries: List[Dict[str, str]] = json.load(f)
    except (json.JSONDecodeError, IOError):
        print("Error: Log file is corrupted or unreadable.")
        return
    
    success_count = 0
    for entry in reversed(log_entries):
        original = entry['original_path']
        new = entry['new_path']
        
        if not os.path.exists(new):
            print(f"Warning: File not found, skipping reversal: {new}")
            continue
        
        try:
            os.rename(new, original)
            success_count += 1
            print(f"Restored: {new} -> {original}")
        except OSError as e:
            print(f"Error restoring {new}: {str(e)}")
    
    try:
        os.remove(log_path)
        print(f"Removed log file: {log_path}")
    except OSError as e:
        print(f"Error removing log file: {str(e)}")
    
    print(f"Successfully restored {success_count}/{len(log_entries)} files")

def main() -> None:
    """Main function to execute the script based on provided parameters."""
    try:
        validate_directory(DIRECTORY)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return
    
    if REVERSE:
        reverse_changes(DIRECTORY)
    else:
        if not PATTERN:
            print("Error: PATTERN must be specified for apply mode")
            return
        apply_changes(DIRECTORY, PATTERN)

if __name__ == "__main__":
    main()