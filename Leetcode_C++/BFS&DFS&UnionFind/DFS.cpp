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

class SolutionT1755 {
public:
    vector<int> make(vector<int> nums) {
        vector<int> ans(1 << nums.size());
        for (int i = 0; i < nums.size(); i++) {
            for (int j = 0; j < (1 << i); j++) {
                ans[j + (1 << i)] = ans[j] + nums[i];
            }
        }
        return ans;
    }

    int minAbsDifference(vector<int>& nums, int goal) {
        int n = nums.size();
        vector<int> left = make({nums.begin(), nums.begin() +n/2});
        vector<int> right = make({nums.begin()+n/2, nums.end()});
        sort(left.begin(), left.end());
        sort(right.rbegin(), right.rend());
        int ans = INT_MAX, i = 0, j = 0;
        while(i < left.size() && j < right.size()) {
            int temp = left[i] + right[j];
            ans = min(ans, abs(goal - temp));
            if (t > goal) j++;
            else if(t < goal) i++;
            else return 0;
        }
        return ans;
    }
};

//N皇后
 class SolutionT51 {
public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> res;
        vector<string> queens(n, string(n, '.'));
        helper(0, queens, res);
        return res;
    }
    void helper(int curRow, vector<string>& queens, vector<vector<string>>& res) {
        int n = queens.size();
        if (curRow == n) {
            res.push_back(queens);
            return;
        }
        for (int i = 0; i < n; ++i) {
            if (isValid(queens, curRow, i)) {
                queens[curRow][i] = 'Q';
                helper(curRow + 1, queens, res);
                queens[curRow][i] = '.';
            }
        }
    }
    bool isValid(vector<string>& queens, int row, int col) {
        for (int i = 0; i < row; ++i) {
            if (queens[i][col] == 'Q') return false;
        }
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; --i, --j) {
            if (queens[i][j] == 'Q') return false;
        }
        for (int i = row - 1, j = col + 1; i >= 0 && j < queens.size(); --i, ++j) {
            if (queens[i][j] == 'Q') return false;
        }
        return true;
    }
};

class SolutionT323 {
public:
    int countComponents(int n, vector<pair<int, int> >& edges) {
        int res = 0;
        vector<vector<int> > g(n);
        vector<bool> v(n, false);
        for (auto a : edges) {
            g[a.first].push_back(a.second);
            g[a.second].push_back(a.first);
        }
        for (int i = 0; i < n; ++i) {
            if (!v[i]) {
                ++res;
                dfs(g, v, i);
            }
        }
        return res;
    }

    void dfs(vector<vector<int> > &g, vector<bool> &v, int i) {
        if (v[i]) return ;
        v[i] = true;
        for (int j = 0; j < g[i].size(); j++) {
            dfs(g, v, g[i][j]);
        }
    }
};

//DFS
class SolutionT547 {
public:
    int findCircleNum(vector<vector<int>>& M) {
        int n = M.size(), res = 0;
        vector<bool> visited(n, false);
        for (int i = 0; i < n; ++i) {
            if (visited[i]) continue;
            helper(M, i, visited);
            ++res;
        }
        return res;
    }
    void helper(vector<vector<int>>& M, int k, vector<bool>& visited) {
        visited[k] = true;
        for (int i = 0; i < M.size(); ++i) {
            if (!M[k][i] || visited[i]) continue;
            helper(M, i, visited);
        }
    }
};

// DFS
class SolutionT261 {
public:
    bool validTree(int n, vector<pair<int, int>>& edges) {
        vector<vector<int>> g(n, vector<int>());
        vector<bool> v(n, false);
        for (auto a : edges) {
            g[a.first].push_back(a.second);
            g[a.second].push_back(a.first);
        }
        if (!dfs(g, v, 0, -1)) return false;
        for (auto a : v) {
            if (!a) return false;
        }
        return true;
    }

    bool dfs(vector<vector<int>> &g, vector<bool> &v, int cur, int pre) {
      if (v[cur]) return false;
      v[cur] = true;
      for (auto a : g[cur]) {
          if (a != pre) {
              //pre 是为了防止 a->b 遍历b的邻接数组的时候肯定有a，防止回流
              if(!dfs(g, v, a, cur)) return false;
          }
      }
      return false;
    }
};

class Solution {
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
    }
}