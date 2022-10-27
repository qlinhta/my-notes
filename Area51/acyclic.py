# Detection acyclic graph


class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def __repr__(self):
        return "Graph(%r, %r)" % (self.nodes, self.edges)

    def __str__(self):
        return "Graph(%r, %r)" % (self.nodes, self.edges)

    def __eq__(self, other):
        return self.nodes == other.nodes and self.edges == other.edges


class Node:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "Node(%r)" % self.name

    def __str__(self):
        return "Node(%r)" % self.name

    def __eq__(self, other):
        return self.name == other.name


class Edge:
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __repr__(self):
        return "Edge(%r, %r)" % (self.source, self.target)

    def __str__(self):
        return "Edge(%r, %r)" % (self.source, self.target)

    def __eq__(self, other):
        return self.source == other.source and self.target == other.target


def acyclic(Graph):
    # Return True if graph is acyclic
    # Return False if graph is cyclic

    # Initialize graph with nodes and edges from input graph
    graph = Graph

    # Detection acyclic graph by removing nodes with no incoming edges
    # Repeat until no nodes with no incoming edges are found
    while True:
        # Initialize list of nodes with no incoming edges
        nodes_no_incoming_edges = []

        # Find nodes with no incoming edges
        for node in graph.nodes:
            # Initialize flag for node with no incoming edges
            node_no_incoming_edges = True

            # Check if node has incoming edges
            for edge in graph.edges:
                if edge.target == node:
                    node_no_incoming_edges = False
                    break

            # Add node to list of nodes with no incoming edges
            if node_no_incoming_edges:
                nodes_no_incoming_edges.append(node)

        # Remove nodes with no incoming edges from graph
        for node in nodes_no_incoming_edges:
            # Remove node from graph
            graph.nodes.remove(node)

            # Remove edges from graph
            for edge in graph.edges:
                if edge.source == node or edge.target == node:
                    graph.edges.remove(edge)

        # Check if no nodes with no incoming edges are found
        if not nodes_no_incoming_edges:
            break

    # Check if graph is acyclic
    if not graph.nodes:
        return True
    else:
        return False


def main():
    # Test acyclic function
    # Test graph with no edges
    nodes = [Node(0), Node(1), Node(2), Node(3), Node(4)]
    edges = []
    graph = Graph(nodes, edges)
    print("Graph:", graph)
    print("Acyclic:", acyclic(graph))

    # Test graph with no cycles
    nodes = [Node(0), Node(1), Node(2), Node(3), Node(4)]
    edges = [Edge(Node(0), Node(1)), Edge(Node(1), Node(2)), Edge(Node(2), Node(3)), Edge(Node(3), Node(4))]
    graph = Graph(nodes, edges)
    print("Graph:", graph)
    print("Acyclic:", acyclic(graph))

    # Test graph with cycles
    nodes = [Node(0), Node(1), Node(2), Node(3), Node(4)]
    edges = [Edge(Node(0), Node(1)), Edge(Node(1), Node(2)), Edge(Node(2), Node(3)), Edge(Node(3), Node(4)),
             Edge(Node(4), Node(0))]
    graph = Graph(nodes, edges)
    print("Graph:", graph)
    print("Acyclic:", acyclic(graph))


if __name__ == "__main__":
    main()
