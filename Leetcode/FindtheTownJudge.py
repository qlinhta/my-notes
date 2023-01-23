class Solution:
    def findJudge(self, n: int, trust: list[list[int]]) -> int:
        if n == 1:
            return 1
        trustDict = {}
        for i in range(1, n + 1):
            trustDict[i] = 0
        for i in trust:
            trustDict[i[0]] -= 1
            trustDict[i[1]] += 1
        for i in trustDict:
            if trustDict[i] == n - 1:
                return i
        return -1

