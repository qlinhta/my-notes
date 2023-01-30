//
// Created by Quyen Linh TA on 30/01/2023.
//

class Solution {
public:
    int nearestExit(vector<vector<char>>& maze, vector<int>& entrance) {
        int n = maze.size();
        int m = maze[0].size();
        vector <vector<bool>> visited(n, vector<bool>(m, false));
        queue <pair<int, int>> q;
        q.push({entrance[0], entrance[1]});
        visited[entrance[0]][entrance[1]] = true;
        int res = 0;
        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                auto [x, y] = q.front();
                q.pop();
                if (x > 0 && !visited[x - 1][y] && maze[x - 1][y] == '.') {
                    if (x - 1 == 0 || x - 1 == n - 1 || y == 0 || y == m - 1) return res + 1;
                    visited[x - 1][y] = true;
                    q.push({x - 1, y});
                }
                if (x < n - 1 && !visited[x + 1][y] && maze[x + 1][y] == '.') {
                    if (x + 1 == 0 || x + 1 == n - 1 || y == 0 || y == m - 1) return res + 1;
                    visited[x + 1][y] = true;
                    q.push({x + 1, y});
                }
                if (y > 0 && !visited[x][y - 1] && maze[x][y - 1] == '.') {
                    if (x == 0 || x == n - 1 || y - 1 == 0 || y - 1 == m - 1) return res + 1;
                    visited[x][y - 1] = true;
                    q.push({x, y - 1});
                }
                if (y < m - 1 && !visited[x][y + 1] && maze[x][y + 1] == '.') {
                    if (x == 0 || x == n - 1 || y + 1 == 0 || y + 1 == m - 1) return res + 1;
                    visited[x][y + 1] = true;
                    q.push({x, y + 1});
                }
            }
            res++;
        }
        return -1;
    }
};