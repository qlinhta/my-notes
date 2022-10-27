# Contract sequence of graph

class Graph(object):
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def __repr__(self):
        return "Graph(%r, %r)" % (self.nodes, self.edges)

    def __str__(self):
        return "Graph(%r, %r)" % (self.nodes, self.edges)

    def __eq__(self, other):
        return self.nodes == other.nodes and self.edges == other.edges


class Node(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "Node(%r)" % self.name

    def __str__(self):
        return "Node(%r)" % self.name

    def __eq__(self, other):
        return self.name == other.name


class Edge(object):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __repr__(self):
        return "Edge(%r, %r)" % (self.source, self.target)

    def __str__(self):
        return "Edge(%r, %r)" % (self.source, self.target)

    def __eq__(self, other):
        return self.source == other.source and self.target == other.target


def twinwidth(graph):
    # Return contract sequence of graph
    # Contract sequence is a list of tuples (node, node)
    # where node is a node in the graph
    # and the tuple represents a contraction of the graph
    # where the two nodes are contracted into one node
    # The first node in the tuple is the node that is kept in the graph after contraction
    # The second node in the tuple is the node that is removed from the graph after contraction
    # The contract sequence is the sequence of contractions that results in a graph with minimum width (twinwidth)

    # Initialize contract sequence
    contract_sequence = []

    # Initialize graph with nodes and edges from input graph
    nodes = graph.nodes
    edges = graph.edges

    # if graph has 2 or less nodes, return empty contract sequence
    if len(nodes) <= 2:
        return contract_sequence

    # Group all neighbors of each node into a dictionary
    # Key is a node
    # Value is a list of all neighbors of the node
    neighbors = {}
    for node in nodes:
        neighbors[node] = []

    for edge in edges:
        neighbors[edge.source].append(edge.target)
        neighbors[edge.target].append(edge.source)

