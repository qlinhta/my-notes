//
// Created by Quyen Linh TA on 31/01/2023.
//

class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;
        deque<int> dq;
        for (int i = 0; i < nums.size(); i++) {
            while (!dq.empty() && nums[dq.back()] < nums[i]) {
                dq.pop_back();
            }
            dq.push_back(i);
            if (dq.front() == i - k) {
                dq.pop_front();
            }
            if (i >= k - 1) {
                res.push_back(nums[dq.front()]);
            }
        }
        return res;

    }
};