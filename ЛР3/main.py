import concurrent.futures
import numpy as np
import time
import threading
import itertools
from scipy.linalg import solve
import random


class Matrix:
    def __init__(self, height):
        self.height, self.width = height, height + 1
        self.matrix = np.zeros((self.height, self.width), dtype=float)

    def __str__(self):
        matrix_str = ""
        for row in self.matrix:
            matrix_str += " ".join(map(str, row)) + "\n"
        return matrix_str

    def generate_big(self):
        self.matrix = np.random.randint(-100, 100,
                                        size=(self.height, self.width), dtype=int)

    def generate(self):
        self.matrix = np.random.randint(-1, 1,
                                        size=(self.height, self.width), dtype=int)

    def cramer_method_simple(self):
        solution = np.array([])
        det = np.linalg.det(self.matrix[:, :-1])

        if det == 0:
            raise ValueError("Determinant can not be zero!")

        last_column = self.matrix[:, -1].copy()

        for i in range(len(self.matrix)):
            temp = self.matrix.copy()
            temp[:, i] = last_column
            solution = np.append(solution, np.linalg.det(temp[:, :-1]))

        return solution / det

    @staticmethod
    def worker(args, result, lock):
        col, temp, last_column = args
        temp_copy = temp.copy()
        temp_copy[:, col] = last_column
        determinant = np.linalg.det(temp_copy[:, :-1])

        with lock:
            result.append(determinant)

    def cramer_method_parallel(self, num_threads):
        solution = np.array([])
        det = np.linalg.det(self.matrix[:, :-1])

        if det == 0:
            raise ValueError("Determinant can not be zero!")

        last_column = self.matrix[:, -1].copy()
        temp = self.matrix.copy()

        result = []
        lock = threading.Lock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(Matrix.worker,
                         [(i, temp, last_column) for i in range(len(self.matrix))],
                         itertools.repeat(result),
                         itertools.repeat(lock))

        solution = np.array(result)
        return solution / det

    def generate_diagonally_dominant_matrix(self):
        n = self.height
        random.seed()

        vector = [0.0] * n

        for i in range(n):
            for j in range(n):
                self.matrix[i][j] = random.random() * 10.0

            row_sum = 0.0
            for j in range(n):
                if j != i:
                    row_sum += abs(self.matrix[i][j])

            self.matrix[i][i] = row_sum + 1.0

        self.matrix[:, -1] = np.random.rand(n)

    def solve_seidel(self, tolerance=1e-6, max_iterations=1000):
        A = self.matrix[:, :-1]
        L = np.tril(self.matrix[:, :-1])
        U = A - L
        B = self.matrix[:, -1]
        x = np.zeros_like(B)
        for i in range(max_iterations):
            x = np.dot(np.linalg.inv(L), B - np.dot(U, x))
        return x

    @staticmethod
    def worker_matrix_method(args, result, lock):
        A, b = args
        solution = solve(A, b)
        with lock:
            result.append(solution)

    def solve_matrix_parallel(self, right_hand_sides, num_threads):
        result = []
        lock = threading.Lock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(Matrix.worker_matrix_method,
                         [(self.matrix, b) for b in right_hand_sides],
                         itertools.repeat(result),
                         itertools.repeat(lock))

        return result


def small_matrices():
    matrix = Matrix(3)
    matrix.generate_big()

    print(f"Dimensions: {(matrix.height, matrix.width)}")
    print(matrix.cramer_method_simple())
    print(matrix.cramer_method_parallel(2))
    print(matrix.cramer_method_parallel(3))


def big_matrices():
    matrix = Matrix(200)
    matrix.generate()

    start_ = time.time()
    matrix.cramer_method_simple()
    end_ = time.time()

    print(f"Dimensions: {(matrix.height, matrix.width)}")
    single_thread_time = end_ - start_
    print(f"Single thread time: {single_thread_time:.4f}s")

    threads = range(2, 11)

    for i in threads:
        current_time = []
        for _ in range(2):
            start_ = time.time()
            matrix.cramer_method_parallel(i)
            end_ = time.time()

            current_time.append(end_ - start_)

        current_time = sum(current_time) / 2
        acceleration = single_thread_time / current_time
        efficiency = acceleration / i

        print(f"Threads: {i} \t"
              f"Time: {current_time:.4f}s \t"
              f"Acceleration: {acceleration:.4f} \t"
              f"Efficiency: {efficiency:.4f}")


def seidel():
    matrix = Matrix(1000)
    matrix.generate_diagonally_dominant_matrix()


    start = time.time()
    solution = matrix.solve_seidel()
    end = time.time()

    print(f"Seidel method:\n{solution}")

    print("Time for Seidel method:", end - start)


def matrix_method():
    matrix = Matrix(15000)
    matrix.generate_big()

    start = time.time()
    solution = solve(matrix.matrix[:, :-1], matrix.matrix[:, -1])
    end = time.time()

    print(f"Matrix method:\n{solution}")

    print("Time for matrix method:", end - start)


def matrix_method_parallel():
    matrix = Matrix(10)

    num_linear_systems = 5
    right_hand_sides = [np.random.rand(10) for _ in range(num_linear_systems)]

    num_threads = 4

    solutions = matrix.solve_matrix_parallel(right_hand_sides, num_threads)
    print(solutions)


if __name__ == '__main__':
    matrix_method()
