import collections


class Solution:
    def shortestAlternatingPaths(self, n: int, redEdges: list[list[int]], blueEdges: list[list[int]]) -> list[int]:
        graph = [[[], []] for _ in range(n)]
        for u, v in redEdges:
            graph[u][0].append(v)
        for u, v in blueEdges:
            graph[u][1].append(v)
        res = [float('inf')] * n
        res[0] = 0
        queue = collections.deque([(0, 0), (0, 1)])
        visited = set()
        while queue:
            node, color = queue.popleft()
            for nei in graph[node][color]:
                if (nei, 1 - color) not in visited:
                    res[nei] = min(res[nei], res[node] + 1)
                    queue.append((nei, 1 - color))
                    visited.add((nei, 1 - color))
        return [-1 if x == float('inf') else x for x in res]
