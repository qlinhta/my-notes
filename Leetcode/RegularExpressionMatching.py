# Solution: Dynamic Programming
# Time Complexity: O(n^2)
# Space Complexity: O(n^2)

class Solution:
    # @param s, an input string
    # @param p, a pattern string
    # @return a boolean
    def isMatch(self, s, p):
        # Initialize the DP table
        dp = [[False for j in range(len(p) + 1)] for i in range(len(s) + 1)]
        dp[0][0] = True
        for i in range(len(p)):
            if p[i] == '*':
                dp[0][i + 1] = dp[0][i - 1]
        # DP
        for i in range(len(s)):
            for j in range(len(p)):
                if p[j] == '.' or p[j] == s[i]:
                    dp[i + 1][j + 1] = dp[i][j]
                elif p[j] == '*':
                    if p[j - 1] == '.' or p[j - 1] == s[i]:
                        dp[i + 1][j + 1] = dp[i][j + 1] or dp[i + 1][j] or dp[i + 1][j - 1]
                    else:
                        dp[i + 1][j + 1] = dp[i + 1][j - 1]
        return dp[len(s)][len(p)]

    # Unit Test
    def test(self):
        print("Testing isMatch()...", end="")
        assert (self.isMatch("aa", "a") == False)
        assert (self.isMatch("aa", "a*") == True)
        assert (self.isMatch("ab", ".*") == True)
        assert (self.isMatch("aab", "c*a*b") == True)
        assert (self.isMatch("mississippi", "mis*is*p*.") == False)
        print("Passed!")


# Main
if __name__ == "__main__":
    solution = Solution()
    solution.test()
