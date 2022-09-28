# Depth first search & breadth first search

from collections import defaultdict

# Graph class
class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance

    # Visualize the graph with matplotlib
    def visualize(self):
        import matplotlib.pyplot as plt
        import networkx as nx
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node)
        for edge in self.edges:
            for node in self.edges[edge]:
                G.add_edge(edge, node)
        nx.draw(G, with_labels=True)
        plt.show()

# Depth first search
def dfs(graph, start, goal):
    explored = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in explored:
            explored.append(node)
            neighbours = graph.edges[node]
            for neighbour in neighbours:
                if neighbour == goal:
                    return explored
                else:
                    stack.append(neighbour)
    return explored

# Breadth first search
def bfs(graph, start, goal):
    explored = []
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node not in explored:
            neighbours = graph.edges[node]
            for neighbour in neighbours:
                if neighbour == goal:
                    return explored
                else:
                    queue.append(neighbour)
            explored.append(node)
    return explored

# Main function
if __name__ == '__main__':
    graph = Graph()
    graph.add_node('A')
    graph.add_node('B')
    graph.add_node('C')
    graph.add_node('D')
    graph.add_node('E')
    graph.add_node('F')
    graph.add_node('G')
    graph.add_node('H')
    graph.add_node('I')
    graph.add_node('J')
    graph.add_node('K')
    graph.add_edge('A', 'B', 1)
    graph.add_edge('A', 'C', 1)
    graph.add_edge('A', 'D', 1)
    graph.add_edge('B', 'E', 1)
    graph.add_edge('B', 'F', 1)
    graph.add_edge('C', 'G', 1)
    graph.add_edge('C', 'H', 1)
    graph.add_edge('D', 'I', 1)
    graph.add_edge('D', 'J', 1)
    graph.add_edge('E', 'K', 1)
    graph.visualize()
    print("Depth first search: ", dfs(graph, 'A', 'K'))
    print("Breadth first search: ", bfs(graph, 'A', 'K'))
    graph.visualize()

