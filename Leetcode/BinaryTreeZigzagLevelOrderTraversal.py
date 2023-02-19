# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = [root]
        result = []
        level = 0
        while queue:
            level += 1
            level_result = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                level_result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if level % 2 == 0:
                result.append(level_result[::-1])
            else:
                result.append(level_result)
        return result