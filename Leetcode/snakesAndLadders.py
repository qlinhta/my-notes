class Solution:
    def snakesAndLadders(self, board: list[list[int]]) -> int:
        """
        Solution optimized for time complexity
        """
        n = len(board)
        visited = [False] * (n * n + 1)
        queue = deque()
        queue.append(1)
        visited[1] = True
        steps = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                cur = queue.popleft()
                if cur == n * n:
                    return steps
                for i in range(1, 7):
                    next = cur + i
                    if next > n * n:
                        break
                    row, col = self.getCoordinate(next, n)
                    if board[row][col] != -1:
                        next = board[row][col]
                    if not visited[next]:
                        queue.append(next)
                        visited[next] = True
            steps += 1
        return -1

    def getCoordinate(self, index: int, n: int) -> tuple[int, int]:
        row = n - 1 - (index - 1) // n
        col = (index - 1) % n
        if row % 2 == n % 2:
            col = n - 1 - col
        return row, col
