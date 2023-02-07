class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        # Create an adjacency list.
        adj_list = {}
        for i in range(len(graph)):
            adj_list[i] = graph[i]

        # Create a color map.
        color_map = {}
        for i in range(len(graph)):
            color_map[i] = -1

        # Perform a BFS.
        for i in range(len(graph)):
            if color_map[i] == -1:
                queue = [i]
                color_map[i] = 0
                while len(queue) > 0:
                    node = queue.pop(0)
                    for neighbor in adj_list[node]:
                        if color_map[neighbor] == -1:
                            color_map[neighbor] = 1 - color_map[node]
                            queue.append(neighbor)
                        elif color_map[neighbor] == color_map[node]:
                            return False

        return True