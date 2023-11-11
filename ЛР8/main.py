import time
from matrix import Matrix
from numba import cuda, float32, jit, njit, prange
import numpy as np


@cuda.jit
def matmul(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


dim = 16


@cuda.jit
def fast_matmul(A, B, C):
    sA = cuda.shared.array(shape=(dim, dim), dtype=float32)
    sB = cuda.shared.array(shape=(dim, dim), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    if x >= C.shape[0] and y >= C.shape[1]:
        return

    tmp = 0.
    for i in range(bpg):
        sA[tx, ty] = A[x, ty + i * dim]
        sB[tx, ty] = B[tx + i * dim, y]

        cuda.syncthreads()

        for j in range(dim):
            tmp += sA[tx, j] * sB[j, ty]

        cuda.syncthreads()

    C[x, y] = tmp


def generate_random_matrix(rows, columns, low=0, high=1000):
    return np.random.randint(low, high, size=(rows, columns), dtype=int)


def cuda_doc_test():
    dimx, dimy = (25000, 25000)
    print(f"Dimensions: {dimx}x{dimy}")
    A_host = generate_random_matrix(dimx, dimy)
    B_host = generate_random_matrix(dimx, dimy)
    C_host = np.zeros((dimx, dimy), dtype=np.float32)

    A_device = cuda.to_device(A_host)
    B_device = cuda.to_device(B_host)
    C_device = cuda.device_array_like(C_host)
    threadsperblock = (dim, dim)
    blockspergrid_x = (C_host.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (C_host.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    start = time.time()
    fast_matmul[blockspergrid, threadsperblock](A_device, B_device, C_device)
    end = time.time()
    print(f"Time: {end - start:.4f}s")
    exit(1)
    # C_device.copy_to_host(C_host)
    # print(C_host[:4, :4])


def test_1():
    dimensions, threads = (250, 250), 2
    m1 = Matrix(dimensions)
    m2 = Matrix(dimensions)
    m1.generate()
    m2.generate()

    start_ = time.time()
    result = m1.simple_multiplication(m2)
    end_ = time.time()

    print(f"Dimensions: {dimensions}")
    single_thread_time = end_ - start_
    print(f"Single thread time: {single_thread_time:.4f}s")

    for i in range(2, 11):
        current_time = []
        for _ in range(3):
            start_ = time.time()
            m_threads = m1.threads_multiplication(m2, i)
            end_ = time.time()

            current_time.append(end_ - start_)

        current_time = sum(current_time) / 3
        acceleration = single_thread_time / current_time
        efficiency = acceleration / i

        print(f"Threads: {i} \t"
              f"Time: {current_time:.4f}s \t"
              f"Acceleration: {acceleration:.4f} \t"
              f"Efficiency: {efficiency:.4f}")


def test_2():
    dimensions = (2000, 2000)

    m1 = Matrix(dimensions)
    m2 = Matrix(dimensions)
    m1.generate()
    m2.generate()

    m_cuda1 = Matrix(dimensions)
    m_cuda1.matrix = generate_random_matrix(dimensions[0], dimensions[1])
    m_cuda2 = Matrix(dimensions)
    m_cuda2.matrix = generate_random_matrix(dimensions[0], dimensions[1])
    m_cuda3 = Matrix(dimensions)
    m_cuda3.matrix = np.zeros(dimensions, dtype=np.float32)

    # single threaded matrix multiplication
    start_ = time.time()
    result = m1.simple_multiplication(m2)
    end_ = time.time()

    print(f"Dimensions: {dimensions}")
    single_thread_time = end_ - start_
    print(f"Single thread Multiplication: {single_thread_time:.4f}s")

    # CUDA matrix multiplication
    A_device = cuda.to_device(m_cuda1.matrix)
    B_device = cuda.to_device(m_cuda2.matrix)
    C_device = cuda.device_array_like(m_cuda3.matrix)

    threadsperblock = (16, 16)

    blockspergrid_x = (m_cuda3.matrix.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (m_cuda3.matrix.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    start_ = time.time()
    fast_matmul[blockspergrid, threadsperblock](A_device, B_device, C_device)
    end_ = time.time()
    cuda_time = end_ - start_
    print(f"CUDA Time: {cuda_time:.4f}s")
    print(f"Acceleration: {single_thread_time / cuda_time}")
    for i in range(2, 11):
        start_ = time.time()
        m_threads = m1.threads_multiplication(m2, i)
        end_ = time.time()

        current_time_threads = end_ - start_
        acceleration_th = single_thread_time / current_time_threads
        efficiency_th = acceleration_th / i

        acceleration_cuda = current_time_threads / cuda_time
        efficiency_cuda = acceleration_cuda / i

        print(f"Threads: {i} \t"
              f"Time: {current_time_threads:.4f}s \t"
              f"Acceleration threads: {acceleration_th:.4f} \t"
              f"Efficiency threads: {efficiency_th:.4f} \t"
              f"Acceleration cuda: {acceleration_cuda:.4f} \t"
              f"Efficiency cuda: {efficiency_cuda:.4f}s")


if __name__ == "__main__":
    test_2()
