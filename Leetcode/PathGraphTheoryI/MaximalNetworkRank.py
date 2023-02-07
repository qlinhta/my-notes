class Solution:
    def maximalNetworkRank(self, n: int, roads: list[list[int]]) -> int:
        if len(roads) == 0:
            return 0

        # Create an adjacency list.
        adj_list = {}
        for i in range(n):
            adj_list[i] = []

        for road in roads:
            adj_list[road[0]].append(road[1])
            adj_list[road[1]].append(road[0])

        max_rank = 0
        for i in range(n):
            for j in range(i + 1, n):
                rank = len(adj_list[i]) + len(adj_list[j])
                if j in adj_list[i]:
                    rank -= 1
                max_rank = max(max_rank, rank)

        return max_rank