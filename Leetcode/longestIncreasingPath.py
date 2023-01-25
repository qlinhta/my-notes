class Solution:

    def dfs(self, matrix, i, j, memo):
        if memo[i][j] > 0:
            return memo[i][j]
        memo[i][j] = 1
        for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
            if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and matrix[x][y] > matrix[i][j]:
                memo[i][j] = max(memo[i][j], 1 + self.dfs(matrix, x, y, memo))
        return memo[i][j]

    def longestIncreasingPath(self, matrix: list[list[int]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        memo = [[0] * len(matrix[0]) for _ in matrix]
        return max(self.dfs(matrix, i, j, memo) for i in range(len(matrix)) for j in range(len(matrix[0])))
