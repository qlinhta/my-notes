class Solution:
    def asteroidCollision(self, asteroids: list[int]) -> list[int]:
        stack = []
        for i in asteroids:
            if i > 0:
                stack.append(i)
            else:
                while stack and stack[-1] > 0 and stack[-1] < abs(i):
                    stack.pop()
                if not stack or stack[-1] < 0:
                    stack.append(i)
                elif stack[-1] == abs(i):
                    stack.pop()
        return stack


def main():
    asteroids = [5, 10, -5]
    print(Solution().asteroidCollision(asteroids))


if __name__ == '__main__':
    main()
