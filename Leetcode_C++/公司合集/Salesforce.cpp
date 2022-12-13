#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<stack>
#include<list>
#include<utility>
#include<cstring>
#include<string>
#include<unordered_map>
#include<unordered_set>
#include<ext/hash_map>
#include<deque>
using namespace std;

class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> temp;
        helper(0, target, candidates, temp, res);
        return res;
    }

    void helper(int i, int cur_sum, vector<int>& candidates, vector<int> &temp, vector<vector<int>> res){
        if (cur_sum == 0) {
            res.push_back(temp);
            return;
        }
        if (i > candidates.size()) return;
        for (int j = i; j < candidates.size(); j++) {
            temp.push_back(candidates[j]);
            helper(j, cur_sum + candidates[j], candidates, temp, res);
            temp.pop_back();
        }
    }
};

class SolutionT1254 {
public:
    // int findRoot(int r, vector<int> root) {
    //     return root[r] == r ? r : findRoot(root[r], root);
    // }

    // void union(int p, int q, vector<int> &root) {
    //     int r1 = findRoot(p, root);
    //     int r2 = findRoot(q, root);

    // }
    vector<int> dirX{0, 1, 0, -1};
    vector<int> dirY{1, 0, -1, 0};

    bool bfs(int i, int j, vector<vector<int>>& grid, vector<int> &visited) {
        int m = grid.size(), n = grid[0].size();
        queue<int> q; q.push(i * n + j);
        bool flag = true;
        while (!q.empty()) {
            auto t = q.front(); q.pop();
            int x = t / n, y = t % n;
            if (x == 0 or y == 0 or x == m - 1 or y == n - 1) flag = false;
            visited[x * n + y] = 1;
            for (int k = 0; k < 4; k++) {
                int tx = x + dirX[k], ty = y + dirY[k];
                if (tx >= 0 && tx < m && ty >= 0 && ty < n && visited[tx * n + ty] == 0 && grid[tx][ty] == 0) {
                    q.push(tx * n + ty);
                }
            }
        }
        return flag;
    }

    int closedIsland(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<int> root(m * n, 0);
        vector<int> visited(m * n, 0);
        int res = 0;
        for (int i = 0; i < root.size(); i++) root[i] = i;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0 && visited[i * n + j] == 0) {
                    if (bfs(i, j, grid, visited)) res++;
                }
            }
        }
        return res;
    }
};

class LRUCache{
public:
    int cap;
    unordered_map<int, list<pair<int, int>>::iterator> map;
    list<pair<int,int>> lru;

    LRUCache(int capacity){
        cap = capacity;
    }

    int get(int key) {
        if (map.count(key) == 0) return -1;
        auto t = map[key];
        int res = t->second;
        lru.erase(t);
        lru.push_front(make_pair(key, res));
        map[key] = lru.begin();
        return res;
    }

    void put(int key, int val) {
        if (map.count(key) != 0) {
            auto t = map[key];
            lru.erase(t);
        }
        lru.push_front(make_pair(key, val));
        map[key] = lru.begin();
        while (lru.size() > cap) {
            auto t = lru.rbegin();
            int del_key = t->first;
            lru.pop_back();
            map.erase(del_key);
        }
    }
};

class SolutionT978 {
public:
    int maxTurbulenceSize(vector<int>& arr) {
        int n = arr.size();
        vector<int> dp1(n, 1);
        vector<int> dp2(n, 1);
        int res = 0;
        for (int i = 0; i < n - 1; i++) {
            if (i % 2 == 0) {
                if (arr[i] < arr[i+1]) dp1[i+1] = dp1[i]+1;
                else if (arr[i] > arr[i+1]) dp2[i+1] = dp2[i]+1;
            }
            else {
                if (arr[i] > arr[i+1]) dp1[i+1] = dp2[i-1]+1;
                else if (arr[i] > arr[i-1]) dp2[i] = dp2[i-1]+1;
            }
            res = max(res, max(dp1[i], dp2[i]));
        }
        return res;
    }
};
