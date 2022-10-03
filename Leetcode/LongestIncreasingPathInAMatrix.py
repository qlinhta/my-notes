class Solution:
    def longestIncreasingPath(self, matrix: list[list[int]]) -> int:
        if not matrix:
            return 0
        self.matrix = matrix
        self.m = len(matrix)
        self.n = len(matrix[0])
        self.visited = [[0] * self.n for _ in range(self.m)]
        self.max_path = 0
        for i in range(self.m):
            for j in range(self.n):
                self.max_path = max(self.max_path, self.dfs(i, j))
        return self.max_path

    def dfs(self, i, j):
        if self.visited[i][j]:
            return self.visited[i][j]
        self.visited[i][j] = 1
        for x, y in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
            if 0 <= x < self.m and 0 <= y < self.n and self.matrix[x][y] > self.matrix[i][j]:
                self.visited[i][j] = max(self.visited[i][j], self.dfs(x, y) + 1)
        return self.visited[i][j]

    '''
    The time complexity is O(mn) and the space complexity is O(mn).
    Because we need to visit every element in the matrix, the time complexity is O(mn).
    '''

if __name__ == "__main__":
    matrix = [[9,9,4],[6,6,8],[2,1,1]]
    print(Solution().longestIncreasingPath(matrix))