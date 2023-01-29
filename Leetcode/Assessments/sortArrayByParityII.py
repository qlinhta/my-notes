class Solution:
    def sortArrayByParityII(self, nums: list[int]) -> list[int]:
        # Best solution for time complexity
        # O(n) time | O(n) space
        even = []
        odd = []
        for num in nums:
            if num % 2 == 0:
                even.append(num)
            else:
                odd.append(num)
        result = []
        for i in range(len(nums)):
            if i % 2 == 0:
                result.append(even.pop())
            else:
                result.append(odd.pop())
        return result
