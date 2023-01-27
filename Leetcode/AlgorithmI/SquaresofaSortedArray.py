class Solution:
    def sortedSquares(self, nums: list[int]) -> list[int]:
        return sorted([x * x for x in nums])