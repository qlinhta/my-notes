import random


class Solution:
    '''
    Create a class:
    - Insert a value (no duplicates)
    - Remove a value
    - Get a random value that been inserted (with equal probability)
    '''

    def __init__(self):
        self.vals_set = set()
        self.n = 0

    def insert(self, val):
        if val in self.vals_set:
            return False
        self.vals_set.add(val)
        self.n += 1

    def remove(self, val):
        if val not in self.vals_set:
            return False
        self.vals_set.remove(val)
        self.n -= 1

    def getRandom(self):
        if self.n == 0:
            return None
        return random.sample(self.vals_set, 1)[0]


if __name__ == '__main__':
    s = Solution()
    s.insert(1)
    s.insert(2)
    s.insert(3)
    s.insert(4)
    s.insert(5)
    s.remove(3)
    print(s.getRandom())
