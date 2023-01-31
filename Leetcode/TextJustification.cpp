//
// Created by Quyen Linh TA on 31/01/2023.
//

class Solution {
public:
    vector<string> fullJustify(vector<string>& words, int maxWidth) {
        vector<string> res;
        int i = 0;
        while (i < words.size()) {
            int j = i + 1;
            int len = words[i].size();
            while (j < words.size() && len + 1 + words[j].size() <= maxWidth) {
                len += 1 + words[j].size();
                j++;
            }
            string s = words[i];
            int space = 1;
            int extra = 0;
            if (j != i + 1 && j != words.size()) {
                space = (maxWidth - len) / (j - i - 1) + 1;
                extra = (maxWidth - len) % (j - i - 1);
            }
            for (int k = i + 1; k < j; k++) {
                s += string(extra > 0 ? space + 1 : space, ' ') + words[k];
                extra--;
            }
            s += string(maxWidth - s.size(), ' ');
            res.push_back(s);
            i = j;
        }
        return res;

    }
};