class Solution:
    def licenseKeyFormatting(self, s: str, k: int) -> str:

        s = s.upper()
        s = s.replace("-", "")
        result = ""
        for i in range(len(s) - 1, -1, -1):
            if len(result) % (k + 1) == k:
                result += "-"
            result += s[i]
        return result[::-1]
