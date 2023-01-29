class Solution:
    def shortestPathBinaryMatrix(self, grid: list[list[int]]) -> int:
        if grid[0][0] != 0 or grid[-1][-1] != 0:
            return -1
        if len(grid) == 1:
            return 1
        queue = [(0, 0)]
        grid[0][0] = 1
        while queue:
            x, y = queue.pop(0)
            if x == len(grid) - 1 and y == len(grid) - 1:
                return grid[x][y]
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == j == 0:
                        continue
                    if 0 <= x + i < len(grid) and 0 <= y + j < len(grid) and grid[x + i][y + j] == 0:
                        grid[x + i][y + j] = grid[x][y] + 1
                        queue.append((x + i, y + j))
        return -1
