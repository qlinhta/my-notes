class Solution:
    # Optimized solution for time complexity
    def numIdenticalPairs(self, nums: list[int]) -> int:
        # Create a dictionary to store the number of times a number appears
        # in the list
        num_dict = {}
        # Create a variable to store the number of identical pairs
        identical_pairs = 0
        # Iterate through the list
        for num in nums:
            # If the number is not in the dictionary, add it
            if num not in num_dict:
                num_dict[num] = 1
            # If the number is in the dictionary, add the number of times it
            # appears to the number of identical pairs
            else:
                identical_pairs += num_dict[num]
                # Increment the number of times the number appears in the
                # dictionary
                num_dict[num] += 1
        # Return the number of identical pairs
        return identical_pairs