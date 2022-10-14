'''
Input: points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
Output: 4
'''

class Solution:
    def maxPoints(self, points: list[list[int]]) -> int:
        if len(points) <= 2:
            return len(points)
        max_points = 0
        for i in range(len(points)):
            slopes = {}
            duplicates = 1
            for j in range(i+1, len(points)):
                if points[i] == points[j]:
                    duplicates += 1
                    continue
                if points[i][0] == points[j][0]:
                    slope = 'inf'
                else:
                    slope = (points[i][1] - points[j][1]) / (points[i][0] - points[j][0])
                if slope in slopes:
                    slopes[slope] += 1
                else:
                    slopes[slope] = 1
            max_points = max(max_points, max(slopes.values(), default=0) + duplicates)
        return max_points


if __name__ == '__main__':
    points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
    print(Solution().maxPoints(points))
