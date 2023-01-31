class Solution:
    def bestTeamScore(self, scores: list[int], ages: list[int]) -> int:
        players = sorted(zip(ages, scores))
        dp = [0] * len(players)
        for i, (age, score) in enumerate(players):
            dp[i] = score
            for j in range(i):
                if players[j][1] <= score:
                    dp[i] = max(dp[i], dp[j] + score)
        return max(dp)