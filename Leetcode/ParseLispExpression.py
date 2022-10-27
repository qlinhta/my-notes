class Solution:

    def parse(self, expression):
        expression = expression.replace('(', ' ( ').replace(')', ' ) ')
        return expression.split()

    def evaluate(self, expression: str) -> int:
        def eval(expression, scope):
            if expression[0] != '(':
                if expression[0].isdigit() or expression[0][1:].isdigit():
                    return int(expression[0])
                return scope[expression[0]]
            if expression[1] == 'let':
                i = 2
                while i < len(expression) - 1:
                    scope[expression[i]] = eval(expression[i + 1], scope)
                    i += 2
                return eval(expression[-1], scope)
            if expression[1] == 'add':
                return eval(expression[2], scope) + eval(expression[3], scope)
            if expression[1] == 'mult':
                return eval(expression[2], scope) * eval(expression[3], scope)

        return eval(self.parse(expression), {})

    def test(self):
        testCases = [
            [
                "(add 1 2)",
                3,
            ],
            [
                "(mult 3 (add 2 3))",
                15,
            ],
            [
                "(let x 2 (mult x 5))",
                10,
            ],
            [
                "(let x 2 (mult x (let x 3 y 4 (add x y))))",
                14,
            ],
            [
                "(let x 3 x 2 x)",
                2,
            ],
            [
                "(let x 1 y 2 x (add x y) (add x y))",
                5,
            ],
            [
                "(let x 2 (add (let x 3 (let x 4 x)) x))",
                6,
            ],
            [
                "(let a1 3 b2 (add a1 1) b2)",
                4,
            ],
        ]
        for expression, expectedOutput in testCases:
            assert self.evaluate(expression) == expectedOutput


# Driver Code
if __name__ == '__main__':
    Solution().test()
