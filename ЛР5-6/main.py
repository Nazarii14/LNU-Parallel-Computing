import time
from graph import Graph


def test1():
    dimension = 10
    graph = Graph(dimension)
    graph.fill_graph(35)

    start_time = time.time()
    shortest_paths_seq = graph.floyd_algorithm()
    end_time = time.time()
    duration = (end_time - start_time)
    print("Sequential:", duration)

    graph.set_threads_number(4)

    start_time = time.time()
    shortest_paths_par = graph.floyd_algorithm_parallel_2()
    end_time = time.time()
    duration = (end_time - start_time)
    print("Parallel:", duration)

    print(shortest_paths_par)
    print(shortest_paths_seq)
    print(shortest_paths_par == shortest_paths_seq)


def test2():
    dimension = 10
    graph = Graph(dimension)
    graph.fill_graph(30)

    start_time = time.time()
    shortest_paths_seq = graph.dijkstra(15)
    end_time = time.time()
    print("Sequential:", end_time - start_time)

    graph.set_threads_number(4)

    start_time = time.time()
    shortest_paths_par = graph.dijkstra_algorithm_parallel(5)
    end_time = time.time()
    print("Parallel:", end_time - start_time)

    print(shortest_paths_seq)
    print(shortest_paths_par)
    print(shortest_paths_seq == shortest_paths_par)


# if __name__ == "__main__":
    # test1()
    # test2()
