class Solution:
    def closestMeetingNode(self, edges: list[int], node1: int, node2: int) -> int:
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        queue = deque([(node1, 0)])
        visited = {node1}
        while queue:
            node, dist = queue.popleft()
            if node == node2:
                return dist
            for nei in graph[node]:
                if nei not in visited:
                    visited.add(nei)
                    queue.append((nei, dist + 1))
        return -1
