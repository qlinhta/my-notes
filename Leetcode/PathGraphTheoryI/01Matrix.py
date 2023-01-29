class Solution:
    # With best optimization of time and space complexity
    def updateMatrix(self, mat: list[list[int]]) -> list[list[int]]:
        if not mat:
            return []
        m, n = len(mat), len(mat[0])
        queue = []
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    queue.append((i, j))
                else:
                    mat[i][j] = float('inf')
        while queue:
            x, y = queue.pop(0)
            for i, j in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= i < m and 0 <= j < n and mat[i][j] > mat[x][y] + 1:
                    mat[i][j] = mat[x][y] + 1
                    queue.append((i, j))
        return mat
