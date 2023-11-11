import time
from graph import Graph


def equal_matrices(matr1, matr2):
    for i in range(len(matr1)):
        for j in range(len(matr1[0])):
            if matr1[i][j] != matr2[i][j]:
                return False
    return True


# floyd small matrices example
def test1():
    dimension = 7
    graph = Graph(dimension)
    graph.fill_graph(25)

    print("Input graph:")
    print(graph)

    start = time.time()
    shortest_paths_seq = graph.floyd_algorithm()
    end = time.time()
    print(f"Sequential: {(end - start):.5f}s")
    graph.set_threads_number(4)

    print("Single thread shortest paths:")
    for i in shortest_paths_seq:
        print(i)

    start = time.time()
    shortest_paths_par = graph.floyd_algorithm_parallel()
    end = time.time()
    print(f"Parallel: {(end - start):.5f}s")

    print("Multithread shortest paths:")
    for i in shortest_paths_par:
        print(i)
    print(f"Single and multi thread answer matrices are equal: "
          f"{equal_matrices(shortest_paths_seq, shortest_paths_par)}")


# floyd big matrices example, time measuring
def test2():
    dimension = 200
    graph = Graph(dimension)
    edges = 250
    graph.fill_graph(edges)

    start_time = time.time()
    shortest_paths_seq = graph.floyd_algorithm()
    end_time = time.time()
    single_thread_duration = (end_time - start_time)
    print(f"Sequential: {single_thread_duration:.5f}s")
    print(f"Dimension: {dimension}")
    print(f"Edges: {edges}")

    tests_num = 3
    threads_num = list(range(2, 11))  # + list(range(20, 51, 10))
    comparison = []

    for i in threads_num:
        graph.set_threads_number(i)
        current_time = []
        for _ in range(tests_num):
            start_ = time.time()
            shortest_paths_par = graph.floyd_algorithm_parallel()
            end_ = time.time()

            current_time.append(end_ - start_)
            comparison.append(equal_matrices(shortest_paths_seq, shortest_paths_par))

        current_time = sum(current_time) / tests_num
        acceleration = single_thread_duration / current_time
        efficiency = acceleration / i

        print(f"Threads: {i} \t"
              f"Time: {current_time:.4f}s \t"
              f"Acceleration: {acceleration:.4f} \t"
              f"Efficiency: {efficiency:.4f}\t"
              f"Correct: {comparison[i]}")


if __name__ == "__main__":
    test1()
    test2()
