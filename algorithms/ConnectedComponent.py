# Find connected components in a graph algorithm

class Component:
    def __init__(self, graph):
        self.graph = graph
        self.visited = [False] * len(graph)
        self.components = []

    def dfs(self, v):
        self.visited[v] = True
        self.components[-1].append(v)
        for u in self.graph[v]:
            if not self.visited[u]:
                self.dfs(u)

    def find_components(self):
        for v in range(len(self.graph)):
            if not self.visited[v]:
                self.components.append([])
                self.dfs(v)

    def get_components(self):
        return self.components


if __name__ == "__main__":
    graph = [[1, 2], [0, 2], [0, 1, 3], [2]]
    print(graph)
    component = Component(graph)
    component.find_components()
    print(component.get_components())
