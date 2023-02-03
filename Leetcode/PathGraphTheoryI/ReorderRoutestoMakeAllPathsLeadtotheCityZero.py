class Solution:
    def minReorder(self, n: int, connections: list[list[int]]) -> int:
        graph = [[] for _ in range(n)]
        for u, v in connections:
            graph[u].append((v, 1))
            graph[v].append((u, 0))
        res = 0
        queue = [0]
        visited = set()
        while queue:
            node = queue.pop()
            visited.add(node)
            for nei, cost in graph[node]:
                if nei not in visited:
                    res += cost
                    queue.append(nei)
        return res
