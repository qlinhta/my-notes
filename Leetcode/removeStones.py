class Solution:
    def removeStones(self, stones: list[list[int]]) -> int:
        def dfs(i):
            seen.add(i)
            for j in range(len(stones)):
                if j not in seen and (stones[i][0] == stones[j][0] or stones[i][1] == stones[j][1]):
                    dfs(j)

        seen = set()
        ans = 0
        for i in range(len(stones)):
            if i not in seen:
                dfs(i)
                ans += 1
        return len(stones) - ans
