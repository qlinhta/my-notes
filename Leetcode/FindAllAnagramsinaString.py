class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        res = []
        if len(s) < len(p):
            return res
        p_counter = collections.Counter(p)
        s_counter = collections.Counter(s[:len(p) - 1])
        for i in range(len(p) - 1, len(s)):
            s_counter[s[i]] += 1
            if s_counter == p_counter:
                res.append(i - len(p) + 1)
            s_counter[s[i - len(p) + 1]] -= 1
            if s_counter[s[i - len(p) + 1]] == 0:
                del s_counter[s[i - len(p) + 1]]
        return res