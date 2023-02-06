class Solution:
    def openLock(self, deadends: list[str], target: str) -> int:
        target, turns = int(target), [0] * 10000
        for el in deadends:
            turns[int(el)] = -1
        dq = deque([0] * (turns[0] + 1))

        while dq:
            cur = dq.popleft()
            if cur == target:
                return turns[cur]

            for x in (10, 100, 1000, 10000):
                for k in (1, 9):
                    nxt = cur // x * x + (cur + k * x // 10) % x
                    if not turns[nxt]:
                        dq.append(nxt)
                        turns[nxt] = turns[cur] + 1

        return -1