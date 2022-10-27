class Solution:
    def floodFill(self, image: list[list[int]], sr: int, sc: int, color: int) -> list[list[int]]:
        if image[sr][sc] == color:
            return image
        self.dfs(image, sr, sc, image[sr][sc], color)
        return image

    def dfs(self, image, sr, sc, oldColor, newColor):
        if sr < 0 or sr >= len(image) or sc < 0 or sc >= len(image[0]) or image[sr][sc] != oldColor:
            return
        image[sr][sc] = newColor
        self.dfs(image, sr + 1, sc, oldColor, newColor)
        self.dfs(image, sr - 1, sc, oldColor, newColor)
        self.dfs(image, sr, sc + 1, oldColor, newColor)
        self.dfs(image, sr, sc - 1, oldColor, newColor)


if __name__ == "__main__":
    image = [[1, 1, 1], [1, 1, 0], [1, 0, 1]]
    sr = 1
    sc = 1
    color = 2
    print(Solution().floodFill(image, sr, sc, color))
