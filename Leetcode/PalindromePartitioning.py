class Solution:
    def partition(self, s: str) -> list[list[str]]:
        """
        Solution optimized for time complexity
        """
        result = []
        self.dfs(s, [], result)
        return result

    def dfs(self, s: str, path: list[str], result: list[list[str]]) -> None:
        if not s:
            result.append(path)
            return
        for i in range(1, len(s) + 1):
            if self.isPalindrome(s[:i]):
                self.dfs(s[i:], path + [s[:i]], result)

    def isPalindrome(self, s: str) -> bool:
        return s == s[::-1]
