//
// Created by Quyen Linh TA on 31/01/2023.
//

class Solution {
public:
    bool isNumber(string s) {
        int i = 0;
        int n = s.length();
        while (i < n && s[i] == ' ') i++;
        if (i < n && (s[i] == '+' || s[i] == '-')) i++;
        bool isNumeric = false;
        while (i < n && isdigit(s[i])) {
            i++;
            isNumeric = true;
        }
        if (i < n && s[i] == '.') {
            i++;
            while (i < n && isdigit(s[i])) {
                i++;
                isNumeric = true;
            }
        }
        if (isNumeric && i < n && s[i] == 'e') {
            i++;
            isNumeric = false;
            if (i < n && (s[i] == '+' || s[i] == '-')) i++;
            while (i < n && isdigit(s[i])) {
                i++;
                isNumeric = true;
            }
        }
        if ((isNumeric && i < n && s[i] == 'E')) {
            i++;
            isNumeric = false;
            if (i < n && (s[i] == '+' || s[i] == '-')) i++;
            while (i < n && isdigit(s[i])) {
                i++;
                isNumeric = true;
            }
        }
        while (i < n && s[i] == ' ') i++;
        return isNumeric && i == n;
    }
};
