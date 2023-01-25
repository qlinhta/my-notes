from collections import defaultdict


class Solution:
    def criticalConnections(self, n: int, connections: list[list[int]]) -> list[list[int]]:
        graph = defaultdict(list)
        for u, v in connections:
            graph[u].append(v)
            graph[v].append(u)
        self.time = 0
        self.low = [0] * n
        self.ids = [0] * n
        self.visited = [False] * n
        self.res = []
        self.dfs(graph, 0, -1)
        return self.res

    def dfs(self, graph, node, parent):
        self.visited[node] = True
        self.ids[node] = self.low[node] = self.time
        self.time += 1
        for neighbor in graph[node]:
            if neighbor == parent:
                continue
            if not self.visited[neighbor]:
                self.dfs(graph, neighbor, node)
                self.low[node] = min(self.low[node], self.low[neighbor])
                if self.ids[node] < self.low[neighbor]:
                    self.res.append([node, neighbor])
            else:
                self.low[node] = min(self.low[node], self.ids[neighbor])



