import collections
import heapq


class Solution:
    def findCheapestPrice(self, n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
        dp = [float('inf')] * n
        dp[src] = 0
        for _ in range(k + 1):
            dp2 = dp[:]
            for u, v, w in flights:
                dp2[v] = min(dp2[v], dp[u] + w)
            dp = dp2
        return dp[dst] if dp[dst] != float('inf') else -1