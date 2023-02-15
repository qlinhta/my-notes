class Solution:
    def removeElement(self, nums: list[int], val: int) -> int:
        nums[:] = [x for x in nums if x != val]
        return len(nums)