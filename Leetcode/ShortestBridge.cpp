//
// Created by Quyen Linh TA on 30/01/2023.
//

class Solution {
public:
    int shortestBridge(vector <vector<int>> &grid) {
        int n = grid.size();
        int m = grid[0].size();
        vector <vector<bool>> visited(n, vector<bool>(m, false));
        queue <pair<int, int>> q;
        bool found = false;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == 1) {
                    dfs(grid, visited, q, i, j);
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        int res = 0;
        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                auto [x, y] = q.front();
                q.pop();
                if (x > 0 && !visited[x - 1][y]) {
                    if (grid[x - 1][y] == 1) return res;
                    visited[x - 1][y] = true;
                    q.push({x - 1, y});
                }
                if (x < n - 1 && !visited[x + 1][y]) {
                    if (grid[x + 1][y] == 1) return res;
                    visited[x + 1][y] = true;
                    q.push({x + 1, y});
                }
                if (y > 0 && !visited[x][y - 1]) {
                    if (grid[x][y - 1] == 1) return res;
                    visited[x][y - 1] = true;
                    q.push({x, y - 1});
                }
                if (y < m - 1 && !visited[x][y + 1]) {
                    if (grid[x][y + 1] == 1) return res;
                    visited[x][y + 1] = true;
                    q.push({x, y + 1});
                }
            }
            res++;
        }
        return res;
    }

    void dfs(vector <vector<int>> &grid, vector <vector<bool>> &visited, queue <pair<int, int>> &q, int x, int y) {
        if (x < 0 || x >= grid.size() || y < 0 || y >= grid[0].size() || visited[x][y] || grid[x][y] == 0) return;
        visited[x][y] = true;
        q.push({x, y});
        dfs(grid, visited, q, x - 1, y);
        dfs(grid, visited, q, x + 1, y);
        dfs(grid, visited, q, x, y - 1);
        dfs(grid, visited, q, x, y + 1);
    }
};