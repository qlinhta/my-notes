class Solution:
    def lengthLongestPath(self, input: str) -> int:
        stack = []
        result = 0
        for line in input.splitlines():
            level = line.count("\t")
            while len(stack) > level:
                stack.pop()
            stack.append(len(line) - level)
            if "." in line:
                result = max(result, sum(stack) + len(stack) - 1)
        return result