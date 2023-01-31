class Solution {
public:
    string minWindow(string s, string t) {
        int n = s.size();
        int m = t.size();
        if (n < m) return "";
        vector<int> map(128, 0);
        for (char c : t) map[c]++;
        int counter = m;
        int begin = 0, end = 0, d = INT_MAX, head = 0;
        while (end < n) {
            if (map[s[end++]]-- > 0) counter--;
            while (counter == 0) {
                if (end - begin < d) d = end - (head = begin);
                if (map[s[begin++]]++ == 0) counter++;
            }
        }
        return d == INT_MAX ? "" : s.substr(head, d);
    }
};