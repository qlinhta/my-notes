class Solution:
    def findMin(self, nums: list[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return min(nums[0], nums[1])
        if nums[0] < nums[-1]:
            return nums[0]
        mid = len(nums) // 2
        return min(self.findMin(nums[:mid]), self.findMin(nums[mid:]))


if __name__ == "__main__":
    nums = [2,2,2,0,1]
    print(Solution().findMin(nums))

