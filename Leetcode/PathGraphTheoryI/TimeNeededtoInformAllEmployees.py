class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: list[int], informTime: list[int]) -> int:
        if n == 1:
            return 0

        manager_dict = {}
        for i in range(len(manager)):
            if manager[i] == -1:
                continue
            if manager[i] in manager_dict:
                manager_dict[manager[i]].append(i)
            else:
                manager_dict[manager[i]] = [i]

        def dfs(node):
            if node not in manager_dict:
                return informTime[node]
            else:
                return informTime[node] + max([dfs(i) for i in manager_dict[node]])

        return dfs(headID)
