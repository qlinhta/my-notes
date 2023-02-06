class Solution:
    """
    You are given a 0-indexed array of strings words and a 2D array of integers queries.
    Each query queries[i] = [li, ri] asks us to find the number of strings present in the range li to ri (both inclusive) of words that start and end with a vowel.
    Return an array ans of size queries.length, where ans[i] is the answer to the ith query.
    Note that the vowel letters are 'a', 'e', 'i', 'o', and 'u'.
    """

    def vowelStrings(self, words: list[str], queries: list[list[int]]) -> list[int]:
        # Using lambda function
        vowelStrings = ['a', 'e', 'i', 'o', 'u']
        result = []
        return [len(list(filter(lambda x: x[0] in vowelStrings and x[-1] in vowelStrings, words[query[0]:query[1] + 1]))) for query in queries]


