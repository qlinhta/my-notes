class Solution:
    def jump(self, nums: List[int]) -> int:
        jumps = 0
        cur_pos = 0
        while cur_pos < len(nums) - 1:
            if cur_pos + nums[cur_pos] >= len(nums) - 1:
                jumps += 1
                break
            max_pos = cur_pos + 1
            for i in range(cur_pos + 1, cur_pos + nums[cur_pos] + 1):
                if i + nums[i] > max_pos + nums[max_pos]:
                    max_pos = i
            cur_pos = max_pos
            jumps += 1
        return jumps