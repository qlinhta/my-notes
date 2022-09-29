class Solution:
    def merge(self, intervals: list[list[int]]) -> list[list[int]]:
        intervals.sort()
        result = []
        for interval in intervals:
            if result and result[-1][1] >= interval[0]:
                result[-1][1] = max(result[-1][1], interval[1])
            else:
                result.append(interval)
        return result



if __name__ == "__main__":
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    print(Solution().merge(intervals))