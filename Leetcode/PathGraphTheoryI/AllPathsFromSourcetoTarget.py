class Solution:
    def allPathsSourceTarget(self, graph: list[list[int]]) -> list[list[int]]:
        def dfs(node, path):
            if node == N - 1:
                ans.append(path)
                return
            for nei in graph[node]:
                dfs(nei, path + [nei])
        N = len(graph)
        ans = []
        dfs(0, [0])
        return ans
