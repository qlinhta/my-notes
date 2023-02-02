//
// Created by Quyen Linh TA on 02/02/2023.
//

class Solution {
public:
    int countDigitOne(int n) {
        int res = 0;
        for (long long i = 1; i <= n; i *= 10) {
            long long divider = i * 10;
            res += (n / divider) * i + min(max(n % divider - i + 1, 0LL), i);
        }
        return res;
    }
};