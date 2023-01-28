class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights:
            return []
        m, n = len(heights), len(heights[0])
        pacific = [[False] * n for _ in range(m)]
        atlantic = [[False] * n for _ in range(m)]
        for i in range(m):
            self.dfs(heights, i, 0, pacific)
            self.dfs(heights, i, n - 1, atlantic)
        for j in range(n):
            self.dfs(heights, 0, j, pacific)
            self.dfs(heights, m - 1, j, atlantic)
        ans = []
        for i in range(m):
            for j in range(n):
                if pacific[i][j] and atlantic[i][j]:
                    ans.append([i, j])
        return ans

    def dfs(self, heights, i, j, visited):
        m, n = len(heights), len(heights[0])
        visited[i][j] = True
        for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
            if 0 <= x < m and 0 <= y < n and not visited[x][y] and heights[x][y] >= heights[i][j]:
                self.dfs(heights, x, y, visited)