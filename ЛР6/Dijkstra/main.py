import multiprocessing
import time
import random
import copyreg


class GraphPoint:
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.connections = []
        self.weight = float('inf')


class GraphConnection:
    def __init__(self, first_point, second_point, weight):
        self.first_point = first_point
        self.second_point = second_point
        self.weight = weight


class Graph:
    def __init__(self, point_num):
        self.points = []
        self.generate_points(point_num)

    def generate_points(self, point_num):
        for i in range(point_num):
            new_point = GraphPoint("Point" + str(i))
            self.add_connections(new_point, point_num)
            self.points.append(new_point)

    def add_connections(self, new_point, point_num):
        if self.points:
            connection = random.choice(self.points)
            weight = random.randint(1, point_num)
            self.create_connection(connection, new_point, weight)
            self.create_connection(new_point, connection, weight)

            for existing_point in self.points:
                if existing_point != connection and random.random() < 0.5:
                    weight = random.randint(1, point_num)
                    self.create_connection(existing_point, new_point, weight)
                    self.create_connection(new_point, existing_point, weight)

    @staticmethod
    def create_connection(first_point, second_point, weight):
        connection = GraphConnection(first_point, second_point, weight)
        first_point.connections.append(connection)

    def __str__(self):
        result = ""
        for point in self.points:
            result += f"{point.name}:\n"
            for con in point.connections:
                result += f"\t{con.second_point.name} - {con.weight}\n"
        return result


def reduce_graph_point(graph_point):
    return (GraphPoint, (graph_point.name,))


copyreg.pickle(GraphPoint, reduce_graph_point)


def worker(args):
    start_con, to_handle, connections, current = args
    for i in range(start_con, start_con + to_handle):
        con = connections[i]
        if not con.second_point.passed:
            if con.second_point.weight > current.weight + con.weight:
                con.second_point.weight = current.weight + con.weight
    return connections


def update_weights(connections, current):
    for con in connections:
        if not con.second_point.passed:
            if con.second_point.weight > current.weight + con.weight:
                con.second_point.weight = current.weight + con.weight


def parallel_dijkstra(graph, start, process_num):
    current = graph.points[start]
    connections = []
    for point in graph.points:
        connections.extend(point.connections)

    args_list = []
    for i in range(process_num):
        start_con = i * len(connections) // process_num
        to_handle = (i + 1) * len(connections) // process_num - start_con
        args_list.append((start_con, to_handle, connections, current))

    with multiprocessing.Pool(processes=process_num) as pool:
        results = pool.map(worker, args_list)

    for connections in results:
        update_weights(connections, current)

    return [point.weight for point in graph.points]


def sequential_dijkstra(graph, start):
    for point in graph.points:
        point.weight = float('inf')
        point.passed = False

    current = graph.points[start]
    current.weight = 0

    while current:
        for connection in current.connections:
            if not connection.second_point.passed:
                if connection.second_point.weight > current.weight + connection.weight:
                    connection.second_point.weight = current.weight + connection.weight

        current.passed = True
        current = min((point for point in graph.points if not point.passed), key=lambda x: x.weight, default=None)

    return [point.weight for point in graph.points]


def test1():
    size = 20
    test = Graph(size)

    sequential_result = sequential_dijkstra(test, 0)

    threads = 2
    parallel_result = parallel_dijkstra(test, 0, threads)

    print(f"Sequential Dijkstra Result: {sequential_result}")
    print(f"Parallel Dijkstra Result: {parallel_result}")
    print(f"Parallel and Sequential Dijkstra results are the same: {parallel_result == sequential_result}")


def test2():
    size = 451
    test = Graph(size)

    start_time = time.time()
    sequential_result = sequential_dijkstra(test, 0)
    sequential_time = time.time() - start_time

    print(f"Sequential Time: {sequential_time:.3f} seconds")

    truth = []
    for thread_num in range(2, 11):
        start_time = time.time()
        parallel_result = parallel_dijkstra(test, 0, thread_num)
        end_time = time.time()

        parallel_time = end_time - start_time
        truth.append(parallel_result == sequential_result)

        acceleration = sequential_time / parallel_time
        efficiency = acceleration / thread_num
        print(f"{thread_num} threads. Time: {parallel_time:.3f}s\t"
              f"Acceleration: {acceleration:.3f}\t"
              f"Efficiency: {efficiency:.3f}")
    print(f"Parallel Dijkstra is correct: {all(truth)}")


if __name__ == '__main__':
    test2()
