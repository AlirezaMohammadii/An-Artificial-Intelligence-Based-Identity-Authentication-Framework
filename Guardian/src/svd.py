import os
import numpy as np


def combine_feature_vectors(input_dir, output_dir, batch_size=10):
    # Get all ".npy" files in the directory and sort them
    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    npy_files.sort()  # Ensure a consistent order

    # Group files into batches of 10
    batches = [
        npy_files[i : i + batch_size] for i in range(0, len(npy_files), batch_size)
    ]

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each batch
    for batch_index, batch_files in enumerate(batches):
        if len(batch_files) == batch_size:  # Only process complete batches
            # Load feature vectors and stack them into a 10x512 matrix
            vectors = [np.load(os.path.join(input_dir, file)) for file in batch_files]
            matrix = np.vstack(vectors)

            # Save the resulting 10x512 matrix to a new .npy file
            output_file = os.path.join(output_dir, f"batch_{batch_index + 1}.npy")
            np.save(output_file, matrix)

            print(f"Saved batch {batch_index + 1} to {output_file}")
        else:
            print(f"Skipping incomplete batch {batch_index + 1}")


def print_matrix_shapes(directory):
    # Get all ".npy" files in the directory
    npy_files = [f for f in os.listdir(directory) if f.endswith(".npy")]

    # Loop through each file and print its shape
    for npy_file in npy_files:
        file_path = os.path.join(directory, npy_file)

        try:
            # Load the matrix and get its shape
            matrix = np.load(file_path)
            shape = matrix.shape

            # Print the first 10 elements of the first row and the file name with its shape
            print(f"First row elements: {matrix[0][:10]}")
            print(f"File: {npy_file} - Shape: {shape}")
        except Exception as e:
            print(f"Error loading file {npy_file}: {e}")


def perform_svd_and_reconstruct(matrix):
    # Ensure the matrix has the expected shape (10, 512)
    if matrix.shape != (10, 512):
        raise ValueError("Expected matrix shape is (10, 512)")

    # Compute SVD
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)

    # Ensure shapes of U, S, and VT are as expected
    if U.shape != (10, 10):
        raise ValueError("Unexpected shape for U")
    if len(S) != 10:
        raise ValueError("Unexpected number of singular values")
    if VT.shape != (10, 512):
        raise ValueError("Unexpected shape for VT")

    # Reconstruct the matrix using the SVD decomposition
    reconstructed_matrix = sum(
        S[i] * np.outer(U[:, i], VT[i, :]) for i in range(len(S))
    )

    # Check if the reconstructed matrix is close to the original
    is_close = np.allclose(matrix, reconstructed_matrix)
    S0 = S[0] * np.outer(U[:, 0], VT[0, :])
    print(f"Singular Values: {S}")
    # print(f"First 10 elements of left eigen vector: {U[0:10, 0]}\n")
    # print(f"First 10 elements of right eigen vector transpose: {VT[0, 0:10]}\n")
    # print(f"First 10 values of first composition element: {S0[0][0:10]}")
    # print(f"Reconstruction matrix first 10 elements: {reconstructed_matrix[0][0:10]}")
    # print(f"Original matrix first 10 elements: {matrix[0][0:10]}")
    # print("Reconstruction matches original:", is_close)


# Usage of the integrated script
# Set the input and output directories for the combined matrices
input_directory = "../data/sample_dataset/npy_test/embedd/"
output_directory = "../data/sample_dataset/npy_test/embedd/matrices"

# Combine feature vectors into batches of 10x512 matrices
# combine_feature_vectors(input_directory, output_directory)

# Print the shapes of the matrices in the output directory
# print_matrix_shapes(output_directory)

# Perform SVD and check reconstruction for each matrix in the output directory
i = 0
for npy_file in os.listdir(output_directory):
    if npy_file.endswith(".npy"):
        file_path = os.path.join(output_directory, npy_file)
        i += 1
        matrix = np.load(file_path)
        print(f"\n The {i}th matrix:")
        perform_svd_and_reconstruct(matrix)
        if i > 40:
            break
