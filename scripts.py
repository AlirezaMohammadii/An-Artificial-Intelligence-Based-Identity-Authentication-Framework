import numpy as np


def combine_vectors_to_matrix(vector1, vector2):
    """
    Combine two input vectors into a 32x32 matrix.

    Parameters:
    - vector1 (numpy.ndarray): The first input vector.
    - vector2 (numpy.ndarray): The second input vector.

    Returns:
    - numpy.ndarray: A 32x32 matrix created by interleaving the elements of the input vectors.
    """
    assert isinstance(vector1, np.ndarray) and vector1.shape == (
        512,
    ), "vector1 must be a numpy array with shape (512,)"
    assert isinstance(vector2, np.ndarray) and vector2.shape == (
        512,
    ), "vector2 must be a numpy array with shape (512,)"

    matrix = np.zeros((32, 32))

    for i, j in zip(range(16), range(1, 32, 2)):
        matrix[i * 2 : (i * 2) + 1] = vector1[i * 32 : (i + 1) * 32]
        matrix[j : j + 1] = vector2[(j - 1) * 16 : (j + 1) * 16]

    return matrix


def generate_matrices(vector_list):
    """
    Generate matrices based on an orderly chosen pair order of vectors from a given list.

    Parameters:
    - vector_list (list): A list containing 10 vectors.

    Returns:
    - list: A list of 10 matrices, each generated by combining two vectors based on the specified pair order.
    """
    assert (
        isinstance(vector_list, list) and len(vector_list) == 10
    ), "vector_list must be a list with 10 vectors"

    # Define the order of pairs
    pair_order = [
        i if i % 2 == 0 else len(vector_list) - i for i in range(len(vector_list))
    ]

    # Generate matrices based on pair_order
    matrices = []
    for i in range(0, len(pair_order), 2):
        index1 = pair_order[i]
        index2 = pair_order[i + 1]

        # Extract the vectors based on the specified order
        vector1 = vector_list[index1]
        vector2 = vector_list[index2]

        # Combine vectors into a matrix using the function
        result_matrix = combine_vectors_to_matrix(vector1, vector2)

        # Append the resulting matrix to the list
        matrices.append(result_matrix)

    return matrices


# Example usage:
np.random.seed(43)
vector_list = [np.round(10 * np.random.rand(512)) for _ in range(10)]

# Generate matrices based on the given vector list
result_matrices = generate_matrices(vector_list)

# Print or use the generated matrices as needed
for i, matrix in enumerate(result_matrices, 1):
    print(f"Matrix {i}:\n{matrix}\n")
