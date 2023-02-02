class Solution:
    def eventualSafeNodes(self, graph: list[list[int]]) -> list[int]:
        def dfs(node):
            if node in visited:
                return visited[node]
            visited[node] = False
            for i in graph[node]:
                if not dfs(i):
                    return False
            visited[node] = True
            return True

        visited = {}
        return [i for i in range(len(graph)) if dfs(i)]

