class Solution:
    def isAlienSorted(self, words: list[str], order: str) -> bool:
        if len(words) == 1:
            return True
        order_dict = {}
        for i in range(len(order)):
            order_dict[order[i]] = i
        for i in range(len(words) - 1):
            for j in range(min(len(words[i]), len(words[i + 1]))):
                if order_dict[words[i][j]] < order_dict[words[i + 1][j]]:
                    break
                elif order_dict[words[i][j]] > order_dict[words[i + 1][j]]:
                    return False
                elif j == min(len(words[i]), len(words[i + 1])) - 1 and len(words[i]) > len(words[i + 1]):
                    return False
        return True