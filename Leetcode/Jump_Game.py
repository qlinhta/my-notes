class Solution:
    def canJump(self, nums: list[int]) -> bool:
        if len(nums) == 1:
            return True
        max_jump = 0
        for i in range(len(nums)):
            if i > max_jump:
                return False
            max_jump = max(max_jump, i + nums[i])
            if max_jump >= len(nums) - 1:
                return True
        return False

if __name__ == "__main__":
    nums = [2,3,1,1,4]
    print(Solution().canJump(nums))
