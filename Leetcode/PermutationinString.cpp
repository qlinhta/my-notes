//
// Created by Quyen Linh TA on 30/01/2023.
//

class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        int n = s1.size();
        int m = s2.size();
        if (n > m) return false;
        vector<int> cnt1(26, 0);
        vector<int> cnt2(26, 0);
        for (int i = 0; i < n; i++) {
            cnt1[s1[i] - 'a']++;
            cnt2[s2[i] - 'a']++;
        }
        for (int i = n; i < m; i++) {
            if (cnt1 == cnt2) return true;
            cnt2[s2[i] - 'a']++;
            cnt2[s2[i - n] - 'a']--;
        }
        return cnt1 == cnt2;

    }
};