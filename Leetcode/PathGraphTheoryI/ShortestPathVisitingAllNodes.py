import collections


class Solution:
    def shortestPathLength(self, graph: list[list[int]]) -> int:
        n = len(graph)
        queue = collections.deque([(i, 1 << i, 0) for i in range(n)])
        visited = set()
        while queue:
            node, mask, dist = queue.popleft()
            if mask == (1 << n) - 1:
                return dist
            for nei in graph[node]:
                new_mask = mask | (1 << nei)
                if (nei, new_mask) not in visited:
                    visited.add((nei, new_mask))
                    queue.append((nei, new_mask, dist + 1))
        return -1