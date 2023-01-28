class Solution:
    def maxDistance(self, grid: list[list[int]]) -> int:
        n = len(grid)
        q = []
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    q.append((i, j))
        if len(q) == 0 or len(q) == n * n:
            return -1
        ans = 0
        while q:
            ans += 1
            for _ in range(len(q)):
                i, j = q.pop(0)
                for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                    if 0 <= x < n and 0 <= y < n and grid[x][y] == 0:
                        grid[x][y] = 1
                        q.append((x, y))
        return ans - 1
