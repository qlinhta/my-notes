# Random Graph Generator

import random

class RandomGraphGenerator:
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.graph = []

    def generate(self):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if random.random() < self.p:
                    self.graph.append((i, j))

    def get_graph(self):
        return self.graph

    def get_adjacency_matrix(self):
        matrix = [[0 for i in range(self.n)] for j in range(self.n)]
        for edge in self.graph:
            matrix[edge[0]][edge[1]] = 1
            matrix[edge[1]][edge[0]] = 1
        return matrix

    def get_adjacency_list(self):
        adjacency_list = [[] for i in range(self.n)]
        for edge in self.graph:
            adjacency_list[edge[0]].append(edge[1])
            adjacency_list[edge[1]].append(edge[0])
        return adjacency_list

    def get_degree_sequence(self):
        degree_sequence = [0 for i in range(self.n)]
        for edge in self.graph:
            degree_sequence[edge[0]] += 1
            degree_sequence[edge[1]] += 1
        return degree_sequence

    def get_degree_distribution(self):
        degree_distribution = {}
        for edge in self.graph:
            degree_distribution[edge[0]] = degree_distribution.get(edge[0], 0) + 1
            degree_distribution[edge[1]] = degree_distribution.get(edge[1], 0) + 1
        return degree_distribution

    def get_average_degree(self):
        return 2 * len(self.graph) / self.n

    def get_average_clustering_coefficient(self):
        adjacency_list = self.get_adjacency_list()
        clustering_coefficient = 0
        for i in range(self.n):
            if len(adjacency_list[i]) > 1:
                count = 0
                for j in adjacency_list[i]:
                    for k in adjacency_list[i]:
                        if j in adjacency_list[k]:
                            count += 1
                clustering_coefficient += count / (len(adjacency_list[i]) * (len(adjacency_list[i]) - 1))
        return clustering_coefficient / self.n

    def get_average_shortest_path_length(self):
        adjacency_matrix = self.get_adjacency_matrix()
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if adjacency_matrix[i][k] and adjacency_matrix[k][j]:
                        adjacency_matrix[i][j] = 1
        count = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if adjacency_matrix[i][j]:
                    count += 1
        return count / (self.n * (self.n - 1) / 2)

    def get_diameter(self):
        adjacency_matrix = self.get_adjacency_matrix()
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if adjacency_matrix[i][k] and adjacency_matrix[k][j]:
                        adjacency_matrix[i][j] = 1
        diameter = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if adjacency_matrix[i][j]:
                    diameter = max(diameter, len(self.get_shortest_path(i, j)))
        return diameter

    def get_shortest_path(self, i, j):
        adjacency_list = self.get_adjacency_list()
        queue = [[i]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == j:
                return path
            for adjacent in adjacency_list[node]:
                new_path = list(path)
                new_path.append(adjacent)
                queue.append(new_path)
        return None

    def get_connected_components(self):
        adjacency_list = self.get_adjacency_list()
        visited = [False for i in range(self.n)]
        connected_components = []
        for i in range(self.n):
            if not visited[i]:
                connected_components.append([])
                queue = [i]
                while queue:
                    node = queue.pop(0)
                    if not visited[node]:
                        visited[node] = True
                        connected_components[-1].append(node)
                        for adjacent in adjacency_list[node]:
                            queue.append(adjacent)
        return connected_components

    def get_average_connected_component_size(self):
        connected_components = self.get_connected_components()
        return sum([len(component) for component in connected_components]) / len(connected_components)

    # Visualize the graph using matplotlib
    def visualize(self):
        import matplotlib.pyplot as plt
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(self.graph)
        nx.draw(G, with_labels = True)
        plt.show()


if __name__ == "__main__":
    random_graph_generator = RandomGraphGenerator(10, 0.5)
    random_graph_generator.generate()
    print(random_graph_generator.get_graph())
    print(random_graph_generator.get_adjacency_matrix())
    print(random_graph_generator.get_adjacency_list())
    print(random_graph_generator.get_degree_sequence())
    print(random_graph_generator.get_degree_distribution())
    print(random_graph_generator.get_average_degree())
    print(random_graph_generator.get_average_clustering_coefficient())
    print(random_graph_generator.get_average_shortest_path_length())
    print(random_graph_generator.get_diameter())
    print(random_graph_generator.get_shortest_path(0, 9))
    print(random_graph_generator.get_connected_components())
    print(random_graph_generator.get_average_connected_component_size())
    random_graph_generator.visualize()

