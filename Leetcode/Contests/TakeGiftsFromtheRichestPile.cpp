//
// Created by Quyen Linh TA on 05/02/2023.
//
class Solution {
public:
    long long pickGifts(vector<int>& gifts, int k) {
        for (int i = 0; i < k; i++) {
            int max = *max_element(gifts.begin(), gifts.end());
            int index = 0;
            for (int j = 0; j < gifts.size(); j++) {
                if (gifts[j] == max) {
                    index = j;
                    break;
                }
            }
            gifts[index] = floor(sqrt(gifts[index]));
        }
        long long sum = 0;
        for (int i = 0; i < gifts.size(); i++) {
            sum += gifts[i];
        }
        return sum;
    }
};