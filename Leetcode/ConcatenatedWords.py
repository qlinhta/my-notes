class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        words.sort(key=len)
        word_set = set()
        res = []
        for word in words:
            if self.canForm(word, word_set):
                res.append(word)
            word_set.add(word)
        return res

    def canForm(self, word, word_set):
        if not word:
            return False
        dp = [False] * (len(word) + 1)
        dp[0] = True
        for i in range(1, len(word) + 1):
            for j in range(i):
                if not dp[j]:
                    continue
                if word[j:i] in word_set:
                    dp[i] = True
                    break
        return dp[-1]

