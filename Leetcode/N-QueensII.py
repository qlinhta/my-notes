class Solution:
    def totalNQueens(self, n: int) -> int:
        self.n = n
        self.ans = 0
        self.board = [['.' for i in range(n)] for j in range(n)]
        self.solve(0)
        return self.ans

    def solve(self, row):
        if row == self.n:
            self.ans += 1
            return
        for col in range(self.n):
            if self.is_valid(row, col):
                self.board[row][col] = 'Q'
                self.solve(row + 1)
                self.board[row][col] = '.'

    def is_valid(self, row, col):
        for i in range(row):
            if self.board[i][col] == 'Q':
                return False
        for i in range(1, row + 1):
            if col - i >= 0 and self.board[row - i][col - i] == 'Q':
                return False
            if col + i < self.n and self.board[row - i][col + i] == 'Q':
                return False
        return True

    def print_board(self):
        for i in self.board:
            print(i)
        print()

if __name__ == '__main__':
    s = Solution()
    print(s.totalNQueens(4))
