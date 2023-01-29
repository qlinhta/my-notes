class Solution:
    def uniquePathsIII(self, grid: list[list[int]]) -> int:
        # Initialize variables to keep track of the starting and ending coordinates
        start_x, start_y, end_x, end_y = 0, 0, 0, 0
        empty_count = 0
        # Iterate through the grid to find the starting and ending coordinates, and count the number of empty squares
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    start_x, start_y = i, j
                elif grid[i][j] == 2:
                    end_x, end_y = i, j
                elif grid[i][j] == 0:
                    empty_count += 1

        # Use DFS to traverse the grid and count the number of unique paths
        def dfs(x, y, empty_count):
            if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]) or grid[x][y] == -1:
                return 0
            if x == end_x and y == end_y:
                if empty_count == 0:
                    return 1
                else:
                    return 0
            if grid[x][y] == 0:
                empty_count -= 1
            grid[x][y] = -1
            count = dfs(x + 1, y, empty_count) + dfs(x - 1, y, empty_count) + dfs(x, y + 1, empty_count) + dfs(x, y - 1, empty_count)
            grid[x][y] = 0
            if grid[x][y] == 0:
                empty_count += 1
            return count

        return dfs(start_x, start_y, empty_count)


if __name__ == "__main__":
    solution = Solution()
    print(solution.uniquePathsIII([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]]))
