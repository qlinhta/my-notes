class Solution:
    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        if not matrix:
            return []
        ans = []
        row, col = len(matrix), len(matrix[0])
        for i in range((min(row, col) + 1) // 2):
            for j in range(i, col - i):
                ans.append(matrix[i][j])
            for j in range(i + 1, row - i):
                ans.append(matrix[j][col - i - 1])
            if row - i - 1 > i:
                for j in range(col - i - 2, i - 1, -1):
                    ans.append(matrix[row - i - 1][j])
            if col - i - 1 > i:
                for j in range(row - i - 2, i, -1):
                    ans.append(matrix[j][i])
        return ans

if __name__ == '__main__':
    s = Solution()
    print(s.spiralOrder([[1,2,3],[4,5,6],[7,8,9]]))
