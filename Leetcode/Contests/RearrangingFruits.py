class Solution:
    def minCost(self, basket1: List[int], basket2: List[int]) -> int:
        """You have two fruit baskets containing n fruits each. You are given two 0-indexed integer arrays basket1 and basket2 representing the cost of fruit in each basket. You want to make both baskets equal. To do so, you can use the following operation as many times as you want:

        Chose two indices i and j, and swap the ith fruit of basket1 with the jth fruit of basket2.
        The cost of the swap is min(basket1[i],basket2[j]).
        Two baskets are considered equal if sorting them according to the fruit cost makes them exactly the same baskets.

        Return the minimum cost to make both the baskets equal or -1 if impossible."""
        # 1. Get the number of fruits in each basket
        num_fruits1 = len(basket1)
        num_fruits2 = len(basket2)
        # 2. Get the total number of fruits
        total_fruits = num_fruits1 + num_fruits2
        # 3. Get the number of fruits in each basket
        num_fruits = total_fruits / 2
        # 4. Check if the number of fruits is an integer
        if num_fruits.is_integer():
            # 5. Get the number of fruits
            num_fruits = int(num_fruits)
            # 6. Get the total number of fruits
            total_fruits = num_fruits * 2
            # 7. Get the total number of fruits in each basket
            num_fruits1 = num_fruits
            num_fruits2 = num_fruits
        else:
            # 8. Get the total number of fruits in each basket
            num_fruits1 = int(num_fruits)
            num_fruits2 = int(num_fruits) + 1
        # 9. Sort the baskets
        basket1.sort()
        basket2.sort()
        # 10. Get the minimum cost
        min_cost = 0
        # 11. Loop through the baskets
        for i in range(num_fruits1):
            # 12. Get the cost of the fruit
            cost = min(basket1[i], basket2[i])
            # 13. Add the cost to the minimum cost
            min_cost += cost
        # 14. Return the minimum cost
        return min_cost
