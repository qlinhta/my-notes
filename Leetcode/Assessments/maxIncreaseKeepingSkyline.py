class Solution:
    def maxIncreaseKeepingSkyline(self, grid: list[list[int]]) -> int:
        # O(n^2) time | O(n) space
        # n = len(grid)
        # n = len(grid[0])
        # O(n) time | O(n) space
        top_bottom = []
        left_right = []
        for row in grid:
            top_bottom.append(max(row))
        for col in range(len(grid[0])):
            max_col = 0
            for row in range(len(grid)):
                max_col = max(max_col, grid[row][col])
            left_right.append(max_col)
        # O(n^2) time | O(1) space
        result = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                result += min(top_bottom[row], left_right[col]) - grid[row][col]
        return result
