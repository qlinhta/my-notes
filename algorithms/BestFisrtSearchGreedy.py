# Best First Search Greedy Algorithm

import sys
import math
import heapq

# Class Graph
class Graph:
    # Constructor
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.distances = {}

    # Add Node
    def add_node(self, value):
        self.nodes.add(value)

    # Add Edge
    def add_edge(self, from_node, to_node, distance):
        self._add_edge(from_node, to_node, distance)
        self._add_edge(to_node, from_node, distance)

    # Add Edge
    def _add_edge(self, from_node, to_node, distance):
        self.edges.setdefault(from_node, [])
        self.edges[from_node].append(to_node)
        self.distances[(from_node, to_node)] = distance

    # Heuristic
    def heuristic(self, from_node, to_node):
        # Manhattan distance on a square grid
        (x1, y1) = from_node
        (x2, y2) = to_node
        return abs(x1 - x2) + abs(y1 - y2)

    # Best First Search Greedy
    def best_first_search_greedy(self, initial, goal):
        frontier = []
        heapq.heappush(frontier, (0, initial))
        came_from = {}
        came_from[initial] = None
        while frontier:
            current = heapq.heappop(frontier)[1]
            if current == goal:
                break
            for next in self.edges[current]:
                if next not in came_from:
                    priority = self.heuristic(goal, next)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current
        return came_from

    # Reconstruct Path
    def reconstruct_path(self, came_from, initial, goal):
        current = goal
        path = []
        while current != initial:
            path.append(current)
            current = came_from[current]
        path.append(initial)
        path.reverse()
        return path

    # Print Path
    def print_path(self, path):
        for node in path:
            print(node)

# Main
def main():
    # Create Graph
    graph = Graph()
    # Add Nodes
    graph.add_node((0, 0))
    graph.add_node((0, 1))
    graph.add_node((0, 2))
    graph.add_node((1, 0))
    graph.add_node((1, 1))
    graph.add_node((1, 2))
    graph.add_node((2, 0))
    graph.add_node((2, 1))
    graph.add_node((2, 2))
    # Add Edges
    graph.add_edge((0, 0), (0, 1), 1)
    graph.add_edge((0, 0), (1, 0), 1)
    graph.add_edge((0, 1), (0, 2), 1)
    graph.add_edge((0, 1), (1, 1), 1)
    graph.add_edge((0, 2), (1, 2), 1)
    graph.add_edge((1, 0), (1, 1), 1)
    graph.add_edge((1, 0), (2, 0), 1)
    graph.add_edge((1, 1), (1, 2), 1)
    graph.add_edge((1, 1), (2, 1), 1)
    graph.add_edge((1, 2), (2, 2), 1)
    graph.add_edge((2, 0), (2, 1), 1)
    graph.add_edge((2, 1), (2, 2), 1)
    # Best First Search Greedy
    came_from = graph.best_first_search_greedy((0, 0), (2, 2))
    # Reconstruct Path
    path = graph.reconstruct_path(came_from, (0, 0), (2, 2))
    # Print Path
    graph.print_path(path)

    # Visualize Graph with NetworkX
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.Graph()
    G.add_nodes_from(graph.nodes)
    G.add_edges_from(graph.edges.keys())
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_edges(G, pos, width=6)
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    plt.axis('off')
    plt.show()

# Call Main
if __name__ == '__main__':
    main()