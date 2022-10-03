'''
You are given an array trees where trees[i] = [xi, yi] represents the location of a tree in the garden.

You are asked to fence the entire garden using the minimum length of rope as it is expensive.
The garden is well fenced only if all the trees are enclosed.

Return the coordinates of trees that are exactly located on the fence perimeter.
'''

# Still wrong answer

import math

class Solution:
    def outerTrees(self, trees: list[list[int]]) -> list[list[int]]:
        # Find the leftmost point
        leftmost = trees[0]
        for tree in trees:
            if tree[0] < leftmost[0]:
                leftmost = tree
        # Sort the points by polar angle
        trees.sort(key=lambda tree: math.atan2(tree[1] - leftmost[1], tree[0] - leftmost[0]))
        # Remove the points with the same polar angle
        i = 1
        while i < len(trees):
            if trees[i][0] == trees[i-1][0] and trees[i][1] == trees[i-1][1]:
                trees.pop(i)
            else:
                i += 1
        # Remove the points that are not on the convex hull
        i = 2
        while i < len(trees):
            if self.crossProduct(trees[i-2], trees[i-1], trees[i]) < 0:
                trees.pop(i-1)
            else:
                i += 1
        return trees

    def crossProduct(self, a: list[int], b: list[int], c: list[int]) -> int:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

if __name__ == "__main__":
    solution = Solution()
    print(solution.outerTrees([[1,1],[2,2],[2,0],[2,4],[3,3],[4,2]]))