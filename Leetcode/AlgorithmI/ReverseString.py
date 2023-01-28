"""
Write a function that reverses a string. The input string is given as an array of characters s.

You must do this by modifying the input array in-place with O(1) extra memory.
"""


# Do not use built-in reverse() method
class Solution:
    def reverseString(self, s: list[str]) -> None:  # s is a list of characters
        left, right = 0, len(s) - 1  # left and right pointers to swap
        while left < right:  # swap until left and right pointers meet
            s[left], s[right] = s[right], s[left]  # swap
            left += 1  # move left pointer right
            right -= 1  # move right pointer left
