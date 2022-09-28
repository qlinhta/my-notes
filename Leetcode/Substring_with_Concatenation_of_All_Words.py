class Solution:
    def findSubstring(self, s: str, words: list[str]) -> list[int]:
        if len(words) == 0:
            return []
        word_len = len(words[0])
        word_num = len(words)
        word_dict = {}
        for word in words:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
        res = []
        for i in range(len(s) - word_len * word_num + 1):
            tmp_dict = {}
            for j in range(word_num):
                word = s[i + j * word_len: i + (j + 1) * word_len]
                if word in tmp_dict:
                    tmp_dict[word] += 1
                else:
                    tmp_dict[word] = 1
            if tmp_dict == word_dict:
                res.append(i)
        return res
