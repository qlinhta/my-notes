class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False
        s1_dict = {}
        for i in s1:
            if i not in s1_dict:
                s1_dict[i] = 1
            else:
                s1_dict[i] += 1
        s2_dict = {}
        for i in s2[:len(s1)]:
            if i not in s2_dict:
                s2_dict[i] = 1
            else:
                s2_dict[i] += 1
        if s1_dict == s2_dict:
            return True
        for i in range(len(s1), len(s2)):
            s2_dict[s2[i - len(s1)]] -= 1
            if s2_dict[s2[i - len(s1)]] == 0:
                del s2_dict[s2[i - len(s1)]]
            if s2[i] not in s2_dict:
                s2_dict[s2[i]] = 1
            else:
                s2_dict[s2[i]] += 1
            if s1_dict == s2_dict:
                return True
        return False