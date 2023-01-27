class Solution:
    def countSubIslands(self, grid1: list[list[int]], grid2: list[list[int]]) -> int:
        if not grid1 or not grid1[0]:
            return 0
        m, n = len(grid1), len(grid1[0])
        res = 0
        for i in range(m):
            for j in range(n):
                if grid2[i][j] == 1:
                    if self.dfs(grid1, grid2, i, j):
                        res += 1
        return res

    def dfs(self, grid1, grid2, i, j):
        if i < 0 or i >= len(grid1) or j < 0 or j >= len(grid1[0]) or grid2[i][j] == 0:
            return True
        if grid1[i][j] == 0:
            return False
        grid2[i][j] = 0
        return self.dfs(grid1, grid2, i + 1, j) and self.dfs(grid1, grid2, i - 1, j) and self.dfs(grid1, grid2, i,
                                                                                                  j + 1) and self.dfs(
            grid1, grid2, i, j - 1)
