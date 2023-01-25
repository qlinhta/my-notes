class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        words = s.split()
        if len(words) != len(pattern):
            return False
        return len(set(zip(pattern, words))) == len(set(pattern)) == len(set(words))