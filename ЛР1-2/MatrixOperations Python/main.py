import time
import multiprocessing
import numpy as np
import concurrent.futures
import random


class MatrixBuiltinPython:
    def __init__(self, dimensions):
        self.width, self.height = dimensions
        self.matrix = [[0] * self.width for _ in range(self.height)]

    def __str__(self):
        matrix_str = ""
        for row in self.matrix:
            matrix_str += " ".join(map(str, row)) + "\n"
        return matrix_str

    def generate(self):
        for i in range(self.height):
            for j in range(self.width):
                self.matrix[i][j] = random.randint(0, 1000)

    def simple_python_multiplication(self, other):
        if self.width != other.height:
            raise ValueError("Dimensions do not match")

        result = MatrixBuiltinPython((self.height, other.width))

        for i in range(self.height):
            for j in range(other.width):
                res = 0
                for k in range(self.width):
                    res += self.matrix[i][k] * other.matrix[k][j]
                result.matrix[i][j] = res

        return result

    def parallel_matrix_multiplication(self, matrix2, num_threads):
        if len(self.matrix[0]) != len(matrix2.matrix):
            raise ValueError("Dimensions do not match")

        num_rows = len(self.matrix)
        num_cols = len(matrix2.matrix[0])

        result = [[0] * num_cols for _ in range(num_rows)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(MatrixBuiltinPython.multiply_row,
                         [(i, self, matrix2, result) for i in range(num_rows)])

        return result

    @staticmethod
    def multiply_row(args):
        row_index, matrix1, matrix2, result = args
        row = matrix1.matrix[row_index]
        num_cols = len(matrix2.matrix[0])
        result_row = [0] * num_cols

        for i in range(num_cols):
            for j in range(len(row)):
                result_row[i] += row[j] * matrix2.matrix[j][i]

        result[row_index] = result_row


class MatrixNumpy:
    def __init__(self, dimensions):
        self.width, self.height = dimensions
        self.matrix = np.zeros((self.height, self.width), dtype=int)

    def __str__(self):
        matrix_str = ""
        for row in self.matrix:
            matrix_str += " ".join(map(str, row)) + "\n"
        return matrix_str

    def generate(self):
        self.matrix = np.random.randint(0, 1001, size=(self.height, self.width), dtype=int)

    def simple_python_multiplication(self, other):
        if self.width != other.height:
            raise ValueError("Dimensions do not match")

        result = MatrixNumpy((self.height, other.width))

        for i in range(self.height):
            for j in range(other.width):
                res = 0
                for k in range(self.width):
                    res += self.matrix[i, k] * other.matrix[k, j]
                result.matrix[i, j] = res

        return result

    def numpy_multiplication(self, other):
        return np.matmul(self.matrix, other.matrix)

    @staticmethod
    def multiply_row_numpy(args):
        row, main, other, result = args
        result.matrix[row, :] = np.dot(main.matrix[row, :], other.matrix)

    def threads_numpy_multiplication(self, other, num_threads):
        if len(self.matrix[0]) != len(other.matrix):
            raise ValueError("Dimensions does not match")

        num_rows = len(self.matrix)
        num_cols = len(other.matrix[0])

        result = MatrixNumpy((len(m1.matrix), len(other.matrix[0])))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(MatrixNumpy.multiply_row_numpy,
                         [(i, self, other, result) for i in range(num_rows)])

        return result


if __name__ == "__main__":
    dimensions, threads = (250, 250), 2

    m1 = MatrixNumpy(dimensions)
    m2 = MatrixNumpy(dimensions)

    m1.generate()
    m2.generate()

    start_ = time.time()
    result = m1.simple_python_multiplication(m2)
    end_ = time.time()

    print(f"Dimensions: {dimensions}")
    single_thread_time = end_ - start_
    print(f"Single thread time: {single_thread_time:.4f}s")

    threads = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for i in threads:
        current_time = []
        for _ in range(3):
            start_ = time.time()
            m_threads = m1.threads_numpy_multiplication(m2, i)
            end_ = time.time()

            current_time.append(end_ - start_)

        current_time = sum(current_time) / 3
        acceleration = single_thread_time / current_time
        efficiency = acceleration / i

        print(f"Threads: {i} \t"
              f"Time: {current_time:.4f}s \t"
              f"Acceleration: {acceleration:.4f} \t"
              f"Efficiency: {efficiency:.4f}")

