import numpy as np
import concurrent.futures
from numba import cuda
import numba


class Matrix:
    def __init__(self, dimensions):
        self.width, self.height = dimensions
        self.matrix = np.zeros(dimensions, dtype=int)

    def __str__(self):
        matrix_str = ""
        for row in self.matrix:
            matrix_str += " ".join(map(str, row)) + "\n"
        return matrix_str

    def __eq__(self, other):
        return np.array_equal(self.matrix, other.matrix)

    def generate(self):
        self.matrix = np.random.randint(0, 1001, size=(self.height, self.width), dtype=int)

    def simple_multiplication(self, other):
        if self.width != other.height:
            raise ValueError("Dimensions do not match")

        result = np.empty((self.height, other.width), dtype=int)

        for i in range(self.height):
            for j in range(other.width):
                result[i, j] = np.dot(self.matrix[i, :], other.matrix[:, j])
        resultMatrix = Matrix((self.height, other.width))
        resultMatrix.matrix = result
        return resultMatrix

    def numpy_multiplication(self, other):
        return np.matmul(self.matrix, other.matrix)

    @staticmethod
    def multiply_row(args):
        row, main, other, result = args
        result.matrix[row, :] = np.dot(main.matrix[row, :], other.matrix)

    def threads_multiplication(self, other, num_threads):
        if len(self.matrix[0]) != len(other.matrix):
            raise ValueError("Dimensions does not match")

        num_rows = self.matrix.shape[0]
        num_cols = other.matrix.shape[1]
        result = Matrix((num_rows, num_cols))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(Matrix.multiply_row,
                         [(i, self, other, result) for i in range(num_rows)])
        return result

    @staticmethod
    @cuda.jit
    def multiply_row_cuda(result, main, other):
        i, j = cuda.grid(2)
        if i < result.shape[0] and j < result.shape[1]:
            res = 0
            for k in range(main.shape[1]):
                res += main[i, k] * other[k, j]
            result[i, j] = res

    def cuda_multiplication(self, other):
        result = np.empty((self.height, other.width), dtype=np.int32)
        blockspergrid = ((result.shape[0] + 32 - 1) // 32, (result.shape[0] + 32 - 1) // 32)

        d_main = cuda.to_device(self.matrix)
        d_other = cuda.to_device(other.matrix)
        d_result = cuda.to_device(result)

        self.multiply_row_cuda_1[blockspergrid, (32, 32)](d_result, d_main, d_other)
        d_result.copy_to_host(result)
        return result

    @staticmethod
    @cuda.jit
    def multiply_row_cuda_1(result, main, other):
        i, j = cuda.grid(2)
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        bw = cuda.blockDim.x
        bh = cuda.blockDim.y

        shared_main = cuda.shared.array(shape=(32, 32), dtype=numba.int32)
        shared_other = cuda.shared.array(shape=(32, 32), dtype=numba.int32)

        row = by * bh + ty
        col = bx * bw + tx

        acc = 0

        for t in range(main.shape[1] // 32):
            shared_main[ty, tx] = main[row, t * 32 + tx]
            shared_other[ty, tx] = other[t * 32 + ty, col]
            cuda.syncthreads()

            for k in range(32):
                acc += shared_main[ty, k] * shared_other[k, tx]

            cuda.syncthreads()

        if row < result.shape[0] and col < result.shape[1]:
            result[row, col] = acc
