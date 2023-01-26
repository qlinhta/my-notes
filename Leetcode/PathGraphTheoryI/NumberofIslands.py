class Solution:
    def numIslands(self, grid: list[list[str]]) -> int:
        """
        Solution optimized for time complexity
        """
        if not grid:
            return 0
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == '1':
                    count += 1
                    self.dfs(grid, i, j)
        return count

    def dfs(self, grid: list[list[str]], i: int, j: int) -> None:
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[i]) or grid[i][j] == '0':
            return
        grid[i][j] = '0'
        self.dfs(grid, i + 1, j)
        self.dfs(grid, i - 1, j)
        self.dfs(grid, i, j + 1)
        self.dfs(grid, i, j - 1)
