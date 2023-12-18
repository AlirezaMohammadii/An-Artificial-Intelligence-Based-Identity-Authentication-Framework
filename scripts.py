import numpy as np


def combine_vectors_to_matrix(vector1, vector2):
    matrix = np.zeros((32, 32))

    for i, j in zip(range(16), range(1, 32, 2)):
        matrix[i * 2 : (i * 2) + 1] = vector1[i * 32 : (i + 1) * 32]
        matrix[j : j + 1] = vector2[(j - 1) * 16 : (j + 1) * 16]

    return matrix


np.random.seed(43)
vector_list = [np.round(10 * np.random.rand(512)) for _ in range(10)]


print(vector_list)
# Define the order of pairs
pair_order = [0, 9, 1, 8, 2, 7, 3, 6, 4, 5, 5, 4, 6, 3, 7, 2, 8, 1, 9, 0]

# Perform operations on pairs
for i in range(0, len(pair_order), 2):
    index1 = pair_order[i]
    index2 = pair_order[i + 1]

    # Extract the vectors based on the specified order
    vector1 = vector_list[index1]
    vector2 = vector_list[index2]
