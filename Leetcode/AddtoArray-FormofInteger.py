class Solution:
    def addToArrayForm(self, num: list[int], k: int) -> list[int]:
        num = int(''.join(map(str, num)))
        return list(map(int, str(num + k)))