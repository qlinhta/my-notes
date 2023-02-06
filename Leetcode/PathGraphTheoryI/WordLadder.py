class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: list[str]) -> int:
        wordList = set(wordList)
        if endWord not in wordList:
            return 0
        dq = deque([beginWord])
        visited = set([beginWord])
        level = 1
        while dq:
            for _ in range(len(dq)):
                cur = dq.popleft()
                if cur == endWord:
                    return level
                for i in range(len(cur)):
                    for j in range(26):
                        nxt = cur[:i] + chr(ord('a') + j) + cur[i + 1:]
                        if nxt in wordList and nxt not in visited:
                            dq.append(nxt)
                            visited.add(nxt)
            level += 1
        return 0
    