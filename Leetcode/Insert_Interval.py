class Solution:
    def insert(self, intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
        if not intervals:
            return [newInterval]
        if not newInterval:
            return intervals
        intervals.append(newInterval)
        intervals.sort(key=lambda x: x[0])
        result = [intervals[0]]
        for i in range(1, len(intervals)):
            if intervals[i][0] <= result[-1][1]:
                result[-1][1] = max(result[-1][1], intervals[i][1])
            else:
                result.append(intervals[i])
        return result

if __name__ == "__main__":
    solution = Solution()
    print(solution.insert([[1,3],[6,9]], [2,5]))
    print(solution.insert([[1,2],[3,5],[6,7],[8,10],[12,16]], [4,8]))
    print(solution.insert([], [5,7]))
    print(solution.insert([[1,5]], [2,3]))
    print(solution.insert([[1,5]], [2,7]))

