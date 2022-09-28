class Solution:
    def trap(self, height: list[int]) -> int:
        if len(height) < 3:
            return 0
        left = 0
        right = len(height) - 1
        left_max = height[left]
        right_max = height[right]
        water = 0
        while left < right:
            if left_max < right_max:
                left += 1
                if height[left] < left_max:
                    water += left_max - height[left]
                else:
                    left_max = height[left]
            else:
                right -= 1
                if height[right] < right_max:
                    water += right_max - height[right]
                else:
                    right_max = height[right]
        return water

if __name__ == "__main__":
    s = Solution()
    print(s.trap([0,1,0,2,1,0,1,3,2,1,2,1]))
