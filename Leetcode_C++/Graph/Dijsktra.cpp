#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<stack>
#include<utility>
#include<cstring>
#include<string>
#include<unordered_map>
#include<unordered_set>
#include<hash_map>
#include<deque>
using namespace std;

class Solution {
public:
    int n = 10;
    vector<vector<int>> adjust_matrix;
    // a 邻接矩阵
    // p 起始点
    // n 一共有n个点
    void Dijsktra(int p) {
        vector<int> dist(n, INT_MAX);
        unordered_set<int> visited;
        //每个点到起始点(已加入集合所有点)的最短距离
        for (int i = 0; i < n; i++) {
            dist[i] = adjust_matrix[p][i]; 
        }
        while (visited.size() < n) {
            int len = INT_MAX;
            //未加入的点，到已加入集合所有点中，最短的距离
            int next_vertex = 0;
            //遍历所有未加入的点，找到距离已加入集合中，所有点，最近的距离
            for (int i = 0; i < n; i++) {
                if (visited.count(i) == 0 && len > dist[i]) {
                    len = dist[i];
                    next_vertex = i;
                }
            }
            //松弛, dist[i]表示（已加入集合） -》 （未加入集合的点 i）的最短距离
            //当前最短距离dist[next_vertex], 未加入点next_vertex，用这个点去松弛剩下的未加入点
            for (int i = 0; i < n; i++) {
                if (visited.count(i) == 0) {
                    dist[i] = min(dist[i], adjust_matrix[next_vertex][i]);
                }
            }
            visited.insert(next_vertex);
        }
        for (int i = 0; i < n; i++) {
            count<< "点" + p + "到点" + i + "距离为" + dist[i] << endl;
        }
    }
};

