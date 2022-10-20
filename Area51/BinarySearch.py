# Binary search algorithm
import random
import time


class BinarySearch(object):
    def __init__(self, array):
        self.array = array

    def search(self, value):
        return self._search(value, 0, len(self.array) - 1)

    def _search(self, value, start, end):
        if start > end:
            return -1

        mid = (start + end) // 2

        if self.array[mid] == value:
            return mid
        elif self.array[mid] > value:
            return self._search(value, start, mid - 1)
        else:
            return self._search(value, mid + 1, end)


if __name__ == '__main__':
    start = time.time()
    # Create a array with 1000000 random numbers between 0 and 1000000
    array = [random.randint(0, 1000000) for _ in range(1000000)]
    # Target value
    value = 3038
    # Sort the array
    array.sort()
    # Create a BinarySearch object
    binary_search = BinarySearch(array)
    # Search the value
    index = binary_search.search(value)
    # Print the result
    print('Index of {}: {}'.format(value, index))
    # Print the time
    print('Time: {}s'.format(time.time() - start))