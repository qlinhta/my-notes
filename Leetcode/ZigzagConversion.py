class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        res = ['']*numRows
        index, step = 0, 1
        for x in s:
            res[index] += x
            if index == 0:
                step = 1
            elif index == numRows-1:
                step = -1
            index += step
        return ''.join(res)