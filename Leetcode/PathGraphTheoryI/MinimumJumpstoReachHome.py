class Solution:
    def minimumJumps(self, forbidden: List[int], a: int, b: int, x: int) -> int:
        limit = 2000 + a + b
        visited = set(forbidden)
        myque = collections.deque([(0, True)]) # (pos, isForward)
        hops = 0
        while(myque):
            l = len(myque)
            while(l > 0):
                l -= 1
                pos, isForward = myque.popleft()
                if pos == x:
                    return hops
                if pos in visited: continue
                visited.add(pos)
                if isForward:
                    nxt_jump = pos - b
                    if nxt_jump >= 0:
                        myque.append((nxt_jump, False))
                nxt_jump = pos + a
                if nxt_jump <= limit:
                    myque.append((nxt_jump, True))
            hops += 1
        return -1