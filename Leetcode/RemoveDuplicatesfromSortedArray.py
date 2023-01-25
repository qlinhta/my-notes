class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        if not nums:
            return 0
        i = 0
        # using enumerate() to get index and value
        for j, num in enumerate(nums):
            if num != nums[i]:
                i += 1
                nums[i] = num
        return i + 1
