class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        def dfs(i):
            for j in range(n):
                if isConnected[i][j] and j not in visited:
                    visited.add(j)
                    dfs(j)
        n = len(isConnected)
        visited = set()
        count = 0
        for i in range(n):
            if i not in visited:
                dfs(i)
                count += 1
        return count