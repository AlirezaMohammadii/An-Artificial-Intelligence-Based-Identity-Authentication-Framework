import os
import glob
from sklearn.model_selection import train_test_split


def list_and_categorize_files(directory):
    """
    Lists and categorizes files in the directory into normal and poisoned groups.
    """
    all_files = glob.glob(os.path.join(directory, "*.npy"))
    user_groups_normal, user_groups_poisoned = {}, {}
    for file in all_files:
        basename = os.path.basename(file)
        parts = basename.split("-")
        user_id, is_poisoned = parts[1], "(" in parts[1]
        if is_poisoned:
            user_id = user_id.strip("()")
            user_groups_poisoned.setdefault(user_id, []).append(file)
        else:
            user_groups_normal.setdefault(user_id, []).append(file)
    return user_groups_normal, user_groups_poisoned


def stratified_split(user_groups_normal, user_groups_poisoned, split_ratios):
    """
    Performs a stratified split of the user groups into training, validation, and test sets.
    """
    all_groups = list(user_groups_normal.values()) + list(user_groups_poisoned.values())
    labels = [0] * len(user_groups_normal) + [1] * len(user_groups_poisoned)
    if not all_groups:
        raise ValueError(
            "No groups to split. Check if the dataset directory is correct and not empty."
        )
    train_val, test_groups, _, _ = train_test_split(
        all_groups, labels, test_size=split_ratios[2], stratify=labels, random_state=42
    )
    train_groups, val_groups, _, _ = train_test_split(
        train_val,
        [labels[i] for i in range(len(train_val))],
        test_size=split_ratios[1] / (split_ratios[0] + split_ratios[1]),
        stratify=[labels[i] for i in range(len(train_val))],
        random_state=42,
    )
    return (
        [item for sublist in train_groups for item in sublist],
        [item for sublist in val_groups for item in sublist],
        [item for sublist in test_groups for item in sublist],
    )


def move_files(files, target_directory):
    """
    Moves the specified files into the target directory.
    """
    os.makedirs(target_directory, exist_ok=True)
    for file_path in files:
        basename = os.path.basename(file_path)
        target_path = os.path.join(target_directory, basename)
        os.rename(file_path, target_path)


def print_split_summary(files_list, label):
    """
    Prints a summary of the split, including the percentage of poisoned vs normal files.
    """
    poisoned_count = sum(1 for f in files_list if "(" in os.path.basename(f))
    normal_count = len(files_list) - poisoned_count
    print(
        f"{label}: {len(files_list)} files - {poisoned_count} or {poisoned_count / len(files_list) * 100:.2f}% poisoned, {normal_count} or {normal_count / len(files_list) * 100:.2f}% normal"
    )


def perform_dataset_split(dataset_directory, split_ratios=(0.6, 0.2, 0.2)):
    """
    Orchestrates the dataset split process.
    """
    user_groups_normal, user_groups_poisoned = list_and_categorize_files(
        dataset_directory
    )
    train_files, val_files, test_files = stratified_split(
        user_groups_normal, user_groups_poisoned, split_ratios
    )
    # Move files to their respective directories and print summary
    move_files(train_files, os.path.join(dataset_directory, "train"))
    move_files(val_files, os.path.join(dataset_directory, "val"))
    move_files(test_files, os.path.join(dataset_directory, "test"))
    print_split_summary(train_files, "Training Set")
    print_split_summary(val_files, "Validation Set")
    print_split_summary(test_files, "Test Set")


if __name__ == "__main__":
    dataset_directory = "../data/sample_dataset/embedding"
    split_ratios = (0.6, 0.2, 0.2)
    perform_dataset_split(dataset_directory, split_ratios)
