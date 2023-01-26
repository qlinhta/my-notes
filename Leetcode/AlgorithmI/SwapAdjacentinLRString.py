class Solution:
    def canTransform(self, start: str, end: str) -> bool:
        """
        Solution optimized for time complexity
        """
        if start == end:
            return True
        if len(start) != len(end):
            return False
        if start.count('X') != end.count('X') or start.count('L') != end.count('L') or start.count('R') != end.count(
                'R'):
            return False
        i = 0
        j = 0
        while i < len(start) and j < len(end):
            if start[i] == 'X':
                i += 1
                continue
            if end[j] == 'X':
                j += 1
                continue
            if start[i] != end[j]:
                return False
            if start[i] == 'L' and i < j:
                return False
            if start[i] == 'R' and i > j:
                return False
            i += 1
            j += 1
        return True
