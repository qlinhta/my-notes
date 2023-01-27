class Solution:

    def dfs(self, grid, i, j):
        if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j]:
            grid[i][j] = 0
            return 1 + self.dfs(grid, i + 1, j) + self.dfs(grid, i - 1, j) + self.dfs(grid, i, j + 1) + self.dfs(grid,
                                                                                                                 i,
                                                                                                                 j - 1)
        return 0

    def closedIsland(self, grid: list[list[int]]) -> int:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (i == 0 or i == len(grid) - 1 or j == 0 or j == len(grid[0]) - 1) and grid[i][j] == 0:
                    self.dfs(grid, i, j)
        return sum(self.dfs(grid, i, j) > 0 for i in range(len(grid)) for j in range(len(grid[0])))


# Generate a hard test case
if __name__=="__main__":
    import random
    grid = [[random.randint(0, 1) for _ in range(100)] for _ in range(100)]
    print(Solution().closedIsland(grid))