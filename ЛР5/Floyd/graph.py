import numpy as np
import concurrent.futures
import multiprocessing
import itertools

INFINITY = np.inf


class Graph:
    threads_number = 1

    def __init__(self, num_of_vertices):
        self.num_of_vertices = num_of_vertices
        self.vertices = np.full((num_of_vertices, num_of_vertices), INFINITY)
        np.fill_diagonal(self.vertices, 0)

    def __str__(self):
        return str(self.vertices)

    def set_threads_number(self, threads_number):
        self.threads_number = threads_number

    def add_edge(self, source, destination, weight):
        if source == destination:
            return
        self.vertices[source][destination] = weight

    def remove_edge(self, source, destination):
        if source == destination:
            return
        self.vertices[source][destination] = np.inf

    def fill_graph(self, number_of_edges):
        if number_of_edges > self.num_of_vertices * self.num_of_vertices:
            raise ValueError("Number of edges is too big")
        if number_of_edges < self.num_of_vertices - 1:
            raise ValueError("Number of edges is too small")

        for _ in range(number_of_edges):
            source = np.random.randint(0, self.num_of_vertices)
            destination = np.random.randint(0, self.num_of_vertices)
            weight = np.random.randint(1, 100)
            self.add_edge(source, destination, weight)

    def floyd_algorithm(self):
        to_return = self.vertices

        for k in range(self.num_of_vertices):
            for i in range(self.num_of_vertices):
                for j in range(self.num_of_vertices):
                    if (
                            to_return[i][k] < INFINITY
                            and to_return[k][j] < INFINITY
                            and to_return[i][k] + to_return[k][j] < to_return[i][j]
                    ):
                        to_return[i][j] = to_return[i][k] + to_return[k][j]
        return to_return

    @staticmethod
    def floyd_worker(args, shortest_paths, lock):
        num_of_vertices, start, end, k = args
        for i in range(num_of_vertices):
            for j in range(start, end):
                if shortest_paths[i][k] < INFINITY and shortest_paths[k][j] < INFINITY and shortest_paths[i][k] + \
                        shortest_paths[k][j] < shortest_paths[i][j]:
                    with lock:
                       shortest_paths[i][j] = shortest_paths[i][k] + shortest_paths[k][j]

    def floyd_algorithm_parallel(self):
        shortest_paths = self.vertices
        vertices_per_thread = self.num_of_vertices // self.threads_number

        params = []
        lock = multiprocessing.Lock()

        for k in range(self.num_of_vertices):
            for i in range(self.threads_number):
                start = i * vertices_per_thread
                end = start + vertices_per_thread if i < self.threads_number - 1 else self.num_of_vertices
                params.append((self.num_of_vertices, start, end, k))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads_number) as executor:
            executor.map(Graph.floyd_worker, params, itertools.repeat(shortest_paths), itertools.repeat(lock))

        return shortest_paths

    def floyd_algorithm_parallel_2(self):
        shortest_paths = self.vertices
        vertices_per_thread = self.num_of_vertices // self.threads_number

        params = []
        lock = multiprocessing.Lock()

        for k in range(self.num_of_vertices):
            start = 0
            end = vertices_per_thread
            for i in range(self.threads_number):
                params.append((self.num_of_vertices, start, end, k))
                start = end
                end = self.num_of_vertices if i == self.threads_number - 2 else end + vertices_per_thread

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads_number) as executor:
            executor.map(Graph.floyd_worker, params, itertools.repeat(shortest_paths), itertools.repeat(lock))
        return shortest_paths
