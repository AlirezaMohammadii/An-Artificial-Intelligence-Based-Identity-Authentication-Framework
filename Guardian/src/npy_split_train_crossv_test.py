import os
import shutil
from sklearn.model_selection import train_test_split

# Ask user for splitting preference
split_option = input(
    "Choose the dataset splitting option (type '2' for train-test or '3' for train-val-test): "
)

# Directory paths
source_dir = "../data/sample_dataset/KNN/train"
train_dir = os.path.join(source_dir, "train")
val_dir = os.path.join(source_dir, "val")  # Only used if user selects 3-way split
test_dir = os.path.join(source_dir, "test")
# parent_test_dir = "../data/sample_dataset/test"  # New test directory location if needed

# Initialize dictionaries to track file movements and poisoned files
moved_files = {"train": [], "val": [], "test": []}
poisoned_files = {"train": [], "val": [], "test": []}


# Function definitions
def extract_identifier(filename):
    return (
        filename.split("-")[0] if filename.startswith("(") else filename.split("-")[0]
    )


def is_poisoned(filename):
    return "(" in filename


def move_files_to_set(ids_set, destination, label):
    if not os.path.exists(destination):
        os.makedirs(destination)
    for id in ids_set:
        for filename in features_by_identifier[id]:
            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(destination, filename)
            shutil.move(src_path, dest_path)
            moved_files[label].append(filename)  # Track moved file
            if is_poisoned(filename):
                poisoned_files[label].append(filename)  # Track poisoned file


# Collect filenames and categorize
filenames = os.listdir(source_dir)
features_by_identifier = {extract_identifier(filename): [] for filename in filenames}
for filename in filenames:
    identifier = extract_identifier(filename)
    features_by_identifier[identifier].append(filename)

# Stratify and split based on unique identifiers
identifiers = list(features_by_identifier.keys())
labels = [1 if id.startswith("(") else 0 for id in identifiers]

if split_option == "3":
    train_ids, test_val_ids, _, _ = train_test_split(
        identifiers, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_ids, test_ids, _, _ = train_test_split(
        test_val_ids,
        [labels[identifiers.index(id)] for id in test_val_ids],
        test_size=0.5,
        stratify=[labels[identifiers.index(id)] for id in test_val_ids],
        random_state=42,
    )
elif split_option == "2":
    train_ids, test_ids, _, _ = train_test_split(
        identifiers, labels, test_size=0.2, stratify=labels, random_state=42
    )
    val_ids = []  # No validation set for 2-way split
else:
    print("Invalid option selected. Please type '2' or '3'.")
    exit()

# Move files to respective directories
move_files_to_set(train_ids, train_dir, "train")
if val_ids:  # For 3-way split
    move_files_to_set(val_ids, val_dir, "val")
move_files_to_set(test_ids, test_dir, "test")

# Move the test directory one level up after populating it
# shutil.move(test_dir, parent_test_dir)
# print(f"Moved 'test' directory to {parent_test_dir}")

# Print summary of poisoned file movements
for partition in poisoned_files:
    num_poisoned = len(poisoned_files[partition])
    print(f"{partition.upper()} SET: {num_poisoned} poisoned files.")
    if num_poisoned > 0:
        for filename in poisoned_files[partition][
            :10
        ]:  # Print first 10 poisoned filenames for brevity
            print(f" - {filename}")
    else:
        print(" - No poisoned files in this set.")
    print()
