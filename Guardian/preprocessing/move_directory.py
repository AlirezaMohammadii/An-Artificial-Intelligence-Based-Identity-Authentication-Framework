import os
import random
import shutil
from typing import List


def move_random_subdirectories(
    source_dir: str,
    destination_dir: str,
    quantity: int,
    seed: int = None
) -> None:
    """
    Randomly selects a specified number of subdirectories from a source directory
    and moves them to a destination directory.

    Parameters:
        source_dir (str): The path of the source directory containing subdirectories.
        destination_dir (str): The path of the destination directory where selected subdirectories will be moved.
        quantity (int): The number of subdirectories to move.
        seed (int, optional): A seed for reproducibility of the random selection. Defaults to None.

    Raises:
        ValueError: If the source directory does not exist or is not a directory.
        ValueError: If the quantity exceeds the number of available subdirectories.
    """
    # Validate source directory
    if not os.path.exists(source_dir) or not os.path.isdir(source_dir):
        raise ValueError(f"The source directory '{source_dir}' does not exist or is not a directory.")

    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # List all subdirectories in the source directory
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    # Validate quantity
    if quantity > len(subdirs):
        raise ValueError(f"Requested quantity ({quantity}) exceeds the number of available subdirectories ({len(subdirs)}).")

    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)

    # Randomly select subdirectories to move
    selected_subdirs = random.sample(subdirs, quantity)

    # Move each selected subdirectory
    for subdir in selected_subdirs:
        source_path = os.path.join(source_dir, subdir)
        destination_path = os.path.join(destination_dir, subdir)
        shutil.move(source_path, destination_path)

    print(f"Successfully moved {quantity} subdirectories from '{source_dir}' to '{destination_dir}'.")


# Example Usage
# Uncomment the lines below to use the script
source_directory = "C:/Users/s222343272/Downloads/datasets/test/"
destination_directory = "C:/Users/s222343272/Downloads/datasets/new_dir/"
number_of_subdirs_to_move = 3
random_seed = 42

move_random_subdirectories(source_directory, destination_directory, number_of_subdirs_to_move, random_seed)


