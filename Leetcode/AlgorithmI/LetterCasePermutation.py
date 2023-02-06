class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        def dfs(s, path, res):
            if not s:
                res.append(path)
                return
            if s[0].isalpha():
                dfs(s[1:], path + s[0].lower(), res)
                dfs(s[1:], path + s[0].upper(), res)
            else:
                dfs(s[1:], path + s[0], res)
        res = []
        dfs(s, '', res)
        return res