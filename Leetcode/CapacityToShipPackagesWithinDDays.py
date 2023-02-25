class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        def canShip(weights, days, capacity):
            daysUsed = 0
            currentWeight = 0
            for weight in weights:
                if currentWeight + weight > capacity:
                    daysUsed += 1
                    currentWeight = weight
                else:
                    currentWeight += weight
            daysUsed += 1
            return daysUsed <= days

        left = max(weights)
        right = sum(weights)
        while left < right:
            mid = (left + right) // 2
            if canShip(weights, days, mid):
                right = mid
            else:
                left = mid + 1
        return left