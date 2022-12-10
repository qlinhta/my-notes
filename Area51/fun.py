def find_max(nums):
    max_num = float("-inf")  # smaller than all other numbers
    for num in nums:
        if num > max_num:
            max_num = num
    return max_num

# Path: Area51/other.py
from fun import find_max

nums = [1, 2, 3, 4, 5]
