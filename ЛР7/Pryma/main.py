import time
import random
import copyreg
import multiprocessing


class GraphPoint:
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.weight = float('inf')
        self.connections = []


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


def prim_worker(args):
    start_con, to_handle, connections, current = args
    processed_connections = []
    for i in range(start_con, start_con + to_handle):
        con = connections[i]
        if not con.second_point.passed:
            processed_connections.append(con)
    return processed_connections


def parallel_prim(graph, start, process_num):
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
        processed_connections_list = pool.map(prim_worker, args_list)

    processed_connections = [connection for sublist in processed_connections_list for connection in sublist]

    min_connections = {}
    for connection in processed_connections:
        if connection.second_point not in min_connections or connection.weight < min_connections[connection.second_point].weight:
            min_connections[connection.second_point] = connection

    for point, connection in min_connections.items():
        point.weight = connection.weight

    return sum(point.weight for point in graph.points)


def sequential_prim(graph, start):
    for point in graph.points:
        point.passed = False
        point.weight = float('inf')

    graph.points[start].weight = 0
    current = graph.points[start]

    def find_min_unprocessed_point():
        unprocessed_points = [connection.second_point for point in graph.points if not point.passed for connection in point.connections if not connection.second_point.passed]
        return min(unprocessed_points, key=lambda x: x.weight if x.weight else float('inf'), default=None)

    while current is not None:
        for connection in current.connections:
            if not connection.second_point.passed:
                if connection.second_point.weight > connection.weight:
                    connection.second_point.weight = connection.weight

        current.passed = True
        current = find_min_unprocessed_point()

    return sum(point.weight for point in graph.points)


def test1():
    size = 10
    graph = Graph(size)

    result_seq = sequential_prim(graph, 0)
    result_par = parallel_prim(graph, 0, 4)
    return result_seq == result_par


def test2():
    return all(test1() for _ in range(100))


def test3():
    size = 300
    graph = Graph(size)

    start_time = time.time()
    result_seq = sequential_prim(graph, 0)
    end_time = time.time()

    seq_time = end_time - start_time
    print(f"Sequential time: {seq_time:.4f}s")
    print(f"Result: {result_seq}")

    for i in range(2, 11):
        start_time = time.time()
        result_par = parallel_prim(graph, 0, i)
        end_time = time.time()
        duration = end_time - start_time
        acceleration = seq_time / duration
        print(f"{i} threads: {duration:.4f}s\t"
              f"Acceleration: {acceleration:.4f}\t"
              f"Efficiency: {(acceleration / i):.4f}\t"
              f"Correct: {result_par == result_seq}")


if __name__ == "__main__":
    test3()
