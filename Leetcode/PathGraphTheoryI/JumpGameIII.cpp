class Solution {
public:
    bool canReach(vector<int>& arr, int start) {
        int n = arr.size();
        vector<bool> visited(n, false);
        queue<int> q;
        q.push(start);
        visited[start] = true;
        while (!q.empty()) {
            int cur = q.front();
            q.pop();
            if (arr[cur] == 0) {
                return true;
            }
            if (cur + arr[cur] < n && !visited[cur + arr[cur]]) {
                q.push(cur + arr[cur]);
                visited[cur + arr[cur]] = true;
            }
            if (cur - arr[cur] >= 0 && !visited[cur - arr[cur]]) {
                q.push(cur - arr[cur]);
                visited[cur - arr[cur]] = true;
            }
        }
        return false;
    }
};