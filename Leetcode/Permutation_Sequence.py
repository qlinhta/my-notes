import math


class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        nums = [i for i in range(1, n+1)]
        res = []
        k -= 1
        while n > 0:
            n -= 1
            index, k = divmod(k, math.factorial(n))
            res.append(str(nums[index]))
            nums.remove(nums[index])
        return ''.join(res)

if __name__ == '__main__':
    s = Solution()
    print(s.getPermutation(3, 3))

