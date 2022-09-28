class Solution:
    def intToRoman(self, num: int) -> str:
        roman = ''
        roman += 'M' * (num // 1000)
        num %= 1000
        roman += 'CM' * (num // 900)
        num %= 900
        roman += 'D' * (num // 500)
        num %= 500
        roman += 'CD' * (num // 400)
        num %= 400
        roman += 'C' * (num // 100)
        num %= 100
        roman += 'XC' * (num // 90)
        num %= 90
        roman += 'L' * (num // 50)
        num %= 50
        roman += 'XL' * (num // 40)
        num %= 40
        roman += 'X' * (num // 10)
        num %= 10
        roman += 'IX' * (num // 9)
        num %= 9
        roman += 'V' * (num // 5)
        num %= 5
        roman += 'IV' * (num // 4)
        num %= 4
        roman += 'I' * (num // 1)
        num %= 1
        return roman

if __name__ == "__main__":
    solution = Solution()
    print(solution.intToRoman(1994))