# python -m unittest test_matrix_generator.py
import unittest
import numpy as np
from matrix32 import MatrixGenerator

matrix_generator = MatrixGenerator()
np.random.seed(43)
vector_list = [np.round(10 * np.random.rand(512)) for _ in range(10)]

result_matrices = matrix_generator.generate_matrices(vector_list)


class TestYourCode(unittest.TestCase):
    # Replace this with your actual code that generates vector_list and result_matrices
    # Example usage:

    def test_vector_list_and_result_matrices_first_pair(self):
        # assert vector_list[9][0:32] == result_matrices[1]
        # Your assertion
        np.testing.assert_array_equal(vector_list[0][0:32], result_matrices[0][0])
        np.testing.assert_array_equal(vector_list[9][0:32], result_matrices[0][1])
        np.testing.assert_array_equal(vector_list[0][32:64], result_matrices[0][2])
        np.testing.assert_array_equal(vector_list[9][32:64], result_matrices[0][3])
        np.testing.assert_array_equal(vector_list[0][64:96], result_matrices[0][4])
        np.testing.assert_array_equal(vector_list[9][64:96], result_matrices[0][5])
        np.testing.assert_array_equal(vector_list[0][480:512], result_matrices[0][30])
        np.testing.assert_array_equal(vector_list[9][480:512], result_matrices[0][31])

    def test_vector_list_and_result_matrices_second_pair(self):
        np.testing.assert_array_equal(vector_list[1][0:32], result_matrices[1][0])
        np.testing.assert_array_equal(vector_list[8][0:32], result_matrices[1][1])
        np.testing.assert_array_equal(vector_list[1][32:64], result_matrices[1][2])
        np.testing.assert_array_equal(vector_list[8][32:64], result_matrices[1][3])
        np.testing.assert_array_equal(vector_list[1][64:96], result_matrices[1][4])
        np.testing.assert_array_equal(vector_list[8][64:96], result_matrices[1][5])
        np.testing.assert_array_equal(vector_list[1][480:512], result_matrices[1][30])
        np.testing.assert_array_equal(vector_list[8][480:512], result_matrices[1][31])

    def test_vector_list_and_result_matrices_third_pair(self):
        np.testing.assert_array_equal(vector_list[2][0:32], result_matrices[2][0])
        np.testing.assert_array_equal(vector_list[7][0:32], result_matrices[2][1])
        np.testing.assert_array_equal(vector_list[2][32:64], result_matrices[2][2])
        np.testing.assert_array_equal(vector_list[7][32:64], result_matrices[2][3])
        np.testing.assert_array_equal(vector_list[2][64:96], result_matrices[2][4])
        np.testing.assert_array_equal(vector_list[7][64:96], result_matrices[2][5])
        np.testing.assert_array_equal(vector_list[2][480:512], result_matrices[2][30])
        np.testing.assert_array_equal(vector_list[7][480:512], result_matrices[2][31])


if __name__ == "__main__":
    unittest.main()
