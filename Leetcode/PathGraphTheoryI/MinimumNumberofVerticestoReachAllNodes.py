class Solution:
    def findSmallestSetOfVertices(self, n: int, edges: list[list[int]]) -> list[int]:

        # Create an array to store the number of incoming edges for each node
        incomingEdges = [0] * n

        # Loop through the edges and increment the incoming edge count for each node
        for edge in edges:
            incomingEdges[edge[1]] += 1

        # Create an array to store the nodes with no incoming edges
        noIncomingEdges = []

        # Loop through the incoming edges array and add the nodes with no incoming edges to the noIncomingEdges array
        for i in range(len(incomingEdges)):
            if incomingEdges[i] == 0:
                noIncomingEdges.append(i)

        # Return the array of nodes with no incoming edges
        return noIncomingEdges