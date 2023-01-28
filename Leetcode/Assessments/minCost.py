class Solution:
    def minCost(self, colors: str, neededTime: list[int]) -> int:
        n = len(colors)
        cost = 0
        if n <= 1:
            return cost

        for i in range(1, n):
            if colors[i] == colors[i - 1]:  # Check if adjacent colors are the same
                cost += min(neededTime[i], neededTime[i - 1])  # record min cost when adjacent colors are the same
                neededTime[i] = max(neededTime[i], neededTime[
                    i - 1])  # update the remaining cost for next iteration since we removed the minimum

        return cost
