class Solution {
public:
    bool possibleBipartition(int n, vector<vector<int>>& dislikes) {
        vector<vector<int>> graph(n+1);
        for(auto d:dislikes){
            graph[d[0]].push_back(d[1]);
            graph[d[1]].push_back(d[0]);
        }
        vector<int> color(n+1,0);
        for(int i=1;i<=n;i++){
            if(color[i]==0){
                queue<int> q;
                q.push(i);
                color[i]=1;
                while(!q.empty()){
                    int cur=q.front();
                    q.pop();
                    for(auto next:graph[cur]){
                        if(color[next]==0){
                            color[next]=-color[cur];
                            q.push(next);
                        }
                        else if(color[next]==color[cur]){
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }
};