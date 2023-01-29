from collections import deque, defaultdict, OrderedDict


class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.freq = 1


class LFUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.minFreq = 0
        self.keyToNode = dict()
        self.freqToList = defaultdict(deque)
        self.freqToKey = defaultdict(set)

    def get(self, key: int) -> int:
        if key not in self.keyToNode:
            return -1

        node = self.keyToNode[key]
        self.touch(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return

        if key in self.keyToNode:
            node = self.keyToNode[key]
            node.value = value
            self.touch(node)
            return

        if len(self.keyToNode) == self.capacity:
            keyToEvict = self.freqToList[self.minFreq].pop()
            self.freqToKey[self.minFreq].remove(keyToEvict)
            del self.keyToNode[keyToEvict]

        self.minFreq = 1
        self.freqToList[1].appendleft(key)
        self.freqToKey[1].add(key)
        self.keyToNode[key] = Node(key, value)

    def touch(self, node):
        prevFreq = node.freq
        newFreq = node.freq + 1
        self.freqToList[prevFreq].remove(node.key)
        self.freqToKey[prevFreq].remove(node.key)

        if len(self.freqToList[prevFreq]) == 0:
            del self.freqToList[prevFreq]
            if prevFreq == self.minFreq:
                self.minFreq += 1

        if newFreq not in self.freqToList:
            self.freqToList[newFreq] = deque()
            self.freqToKey[newFreq] = set()

        self.freqToList[newFreq].appendleft(node.key)
        self.freqToKey[newFreq].add(node.key)
        node.freq = newFreq
