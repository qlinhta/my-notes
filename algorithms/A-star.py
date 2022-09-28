# Quyen Linh TA
# 2022-09-28

# Class graph
class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.adjacency_list = {}
        for node in self.nodes:
            self.adjacency_list[node] = []
        for edge in self.edges:
            self.adjacency_list[edge[0]].append(edge[1])
            self.adjacency_list[edge[1]].append(edge[0])

    def get_neighbors(self, node):
        return self.adjacency_list[node]

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges

    def get_adjacency_list(self):
        return self.adjacency_list

    def get_heuristic(self, node):
        return self.heuristic[node]

    def set_heuristic(self, heuristic):
        self.heuristic = heuristic

    def get_cost(self, node1, node2):
        return self.cost[(node1, node2)]

    def set_cost(self, cost):
        self.cost = cost

# A* algorithm
def A_star(graph, start, goal):
    open_set = [start]
    closed_set = []
    came_from = {}
    g_score = {}
    f_score = {}
    for node in graph.get_nodes():
        g_score[node] = float('inf')
        f_score[node] = float('inf')
    g_score[start] = 0
    f_score[start] = graph.get_heuristic(start)
    while open_set:
        current = open_set[0]
        for node in open_set:
            if f_score[node] < f_score[current]:
                current = node
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        open_set.remove(current)
        closed_set.append(current)
        for neighbor in graph.get_neighbors(current):
            if neighbor in closed_set:
                continue
            if neighbor not in open_set:
                open_set.append(neighbor)
            tentative_g_score = g_score[current] + graph.get_cost(current, neighbor)
            if tentative_g_score >= g_score[neighbor]:
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + graph.get_heuristic(neighbor)
    return None

# Visualize the path with matplotlib
def visualize(graph, path):
    import matplotlib.pyplot as plt
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(graph.get_nodes())
    G.add_edges_from(graph.get_edges())
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=graph.get_edges(), width=6)
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    plt.axis('off')
    plt.show()

# Main
if __name__ == "__main__":
    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    edges = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'E'), ('B', 'F'), ('C', 'G'), ('D', 'H'), ('D', 'I')]
    graph = Graph(nodes, edges)
    graph.set_heuristic({'A': 9, 'B': 8, 'C': 7, 'D': 6, 'E': 5, 'F': 4, 'G': 3, 'H': 2, 'I': 1})
    graph.set_cost({('A', 'B'): 1, ('A', 'C'): 1, ('A', 'D'): 1, ('B', 'E'): 1, ('B', 'F'): 1, ('C', 'G'): 1, ('D', 'H'): 1, ('D', 'I'): 1})
    print(A_star(graph, 'A', 'I'))
    visualize(graph, A_star(graph, 'A', 'I'))
