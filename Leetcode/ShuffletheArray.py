class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        first_half = nums[:n]
        second_half = nums[n:]
        shuffled = []
        for i in range(n):
            shuffled.append(first_half[i])
            shuffled.append(second_half[i])
        return shuffled
