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
#include<hash_map>
#include<deque>
using namespace std;

//Connected Group
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        if (grid.empty() || grid[0].empty()) return 0;
        int m = grid.size(), n = grid[0].size(), res = 0;
        vector<vector<bool>> visited(m, vector<bool>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == '0' || visited[i][j]) continue;
                helper(grid, visited, i, j);
                ++res;
            }
        }
        return res;
    }
    void helper(vector<vector<char>>& grid, vector<vector<bool>>& visited, int x, int y) {
        if (x < 0 || x >= grid.size() || y < 0 || y >= grid[0].size() || grid[x][y] == '0' || visited[x][y]) return;
        visited[x][y] = true;
        helper(grid, visited, x - 1, y);
        helper(grid, visited, x + 1, y);
        helper(grid, visited, x, y - 1);
        helper(grid, visited, x, y + 1);
    }
};

//Disk Space Analysis
class Solution {
public:
    int maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> dq;
        for (int i = 0; i < k; i++) {
            while (!dq.empty() && nums[i] <= nums[dq.back()]) dq.pop_back();
            dq.push_back(i);
        }
        vector<int> res;
        for (int i = k; i < nums.size(); i++) {
            res.push_back(nums[dq.front()]);
            if (dq.front() <= i - k) dq.pop_front();
            while (!dq.empty() && nums[i] <= nums[dq.back()]) dq.pop_back();
            dq.push_back(i);
        }
        res.push_back(nums[dq.front()]);
        int t = INT_MAX;
        for (int i = 0; i < res.size(); i++) {
            t = min(t, res[i]);
        }
        return t;
    }
};

class mypair {
public:
    int count, prevIndex;
    mypair(int count, int prevIndex):count(count), prevIndex(prevIndex) {}
};

// Distance Metric
class Solution {
public:
    vector<int> distanceMatrix(vector<int>& nums) {
        unordered_map<int, vector<int>> map;
        vector<int> res(nums.size(), 0);
        for (int i = 0; i < nums.size(); i++) {
            map[nums[i]].push_back(i);
        }
        for (int j = 0; j < nums.size(); j++) {
            int t = 0;
            for (auto p : map[nums[j]]) {
                t += abs(p - j);
            }
            res[j] = t;
        }
        return res;
    }

    //Distance Metric map 存个数已经上一个位置
    vector<int> distanceMatrix(vector<int> nums) {
        unordered_map<int, mypair> map;
        vector<int> left(nums.size(), 0);
        vector<int> right(nums.size(), 0);
        for (int i = 0; i < nums.size(); i++) {
            if (map.count(nums[i]) == 0) {
                left[i] = 0;
                map[nums[i]] = mypair(1, i);
            } else {
                mypair temp = map[nums[i]];
                left[i] = temp.count * (i - temp.prevIndex) + left[temp.prevIndex];
                map[nums[i]] = mypair(temp.count + 1, i);
            }
        }
        map.clear();
        for (int i = nums.size() - 1; i >= 0; i--) {
            if (map.count(nums[i]) == 0) {
                right[i] = 0;
                map[nums[i]] = mypair(1, i);
            } else {
                mypair temp = map[nums[i]];
                right[i] = temp.count * (abs(i - temp.prevIndex)) + right[temp.prevIndex];
                map[nums[i]] = mypair(temp.count + 1, i);
            }
        }

        vector<int> res;
        for (int i = 0 ; i < nums.size(); i++) {
            res.push_back(left[i] + right[i]);
        };

        //Distance Metric DP
    }

    vector<int> distanceMatrix(vector<int> nums) {
        unordered_map<int, pair<int, int>> map;
        vector<int> left(nums.size(), 0);
        vector<int> right(nums.size(), 0);
        for (int i = 0; i < nums.size(); i++) {
            if (map.count(nums[i]) == 0) {
                left[i] = 0;
                map[nums[i]] = {1, i};
            } else {
                auto temp = map[nums[i]];
                left[i] = temp.first * (i - temp.second) + left[temp.second];
                map[nums[i]] = {temp.first + 1, i};
            }
        }
        map.clear();
        for (int i = nums.size() - 1; i >= 0; i--) {
            if (map.count(nums[i]) == 0) {
                right[i] = 0;
                map[nums[i]] = {1, i};
            } else {
                auto temp = map[nums[i]];
                right[i] = temp.first * (abs(i - temp.second)) + right[temp.second];
                map[nums[i]] = {temp.first+ 1, i};
            }
        }

        vector<int> res;
        for (int i = 0 ; i < nums.size(); i++) {
            res.push_back(left[i] + right[i]);
        };
        return res;
    }
    //Distance Metric DP
    // vector<int> distanceMatrix(vector<int> nums) {
    //     int n = nums.size();
    //     vector<int> res(n, 0);
    //     for (int i = 0; i < n; i++) {
    //         for (int j = i; j < n; j++) {
    //             if (i != j && nums[i] == nums[j]) {

    //             }
    //         }
    //     }
    // }    

    //Web Pagination
    vector<string> Web(vector<vector<string>> items, int sortP, int sortO, int perP, int pageNum) {
        int n = items.size();
        // int start = pageNum * perP, end = min(start + perP - 1, n - 1);
        // vector<int> curPage(items.begin() + start, items.begin() + end);
        // sort(curPage.begin(), curPage.end(), [](int a, int b) {
        //     return a < b;
        // });
        
        if (sortP == 1) 
            sort(items.begin(), items.end(), [](vector<int> &a, vector<int> &b) {
                return a[1] < b[1];
            });
        else if (sortP == 2)
            sort(items.begin(), items.end(), [](vector<int> &a, vector<int> &b) {
                return a[2] < b[2];
            });
        if (sortO == 1) reverse(items.begin(), items.end());
        int start = perP * pageNum;
        vector<string> res;
        for (int i = start; i < n && i < start + perP; i++) {
            res.push_back(items[i][0]);
        }
        return res;
    }
};

//Distance Metric DP


//Social Media Connections
class Solution {
    int findMinimum(int n, vector<int> edges_from, vector<int> edges_to) {
        unordered_map<int, unordered_set<int>> graph;
        unordered_set<int> keySet;
        int res = INT_MAX;
        for (int i = 0; i < edges_from.size(); i++) {
            keySet.insert(edges_from[i]);
            graph[edges_from[i]].insert(edges_to[i]);
            graph[edges_to[i]].insert(edges_from[i]);
        }

        for (auto node : keySet) {
            auto neighbors = graph[node];
            if (neighbors.size() < 2) continue;
            for (auto nei1 : neighbors) {
                auto nei2 = graph[nei1];
                auto nei3 = findSharedNeighbors(neighbors, nei2);
                for (auto n : nei3) {
                    int productSum = neighbors.size() + nei2.size() + graph[n].size() - 6;
                    res = max(res, productSum);
                }
            }
        }
    }

    unordered_set<int> findSharedNeighbors(unordered_set<int> l1, unordered_set<int> l2) {
        unordered_set<int> res;
        for (auto node : l1) {
            if (l2.count(node)) {
                res.insert(node);
            }
        }
        return res;
    }
};


//Intelligent Substring
class Solution{
    int getSpecialSubstring(string s, int k, string charValue) {
        unordered_map<char, int> dict;
        unordered_set<char> normal, special;
        for (int i = 0; i < charValue.size(); i++) {
            if (charValue[i] == 0) {
                dict['a' + i] = 0;
                normal.insert('a' + i);
            } else {
                dict['a' + i] = 1;
                special.insert('a' + i);
            }
        }
        int count = 0, left = 0, right = 0, n = s.size(), res = INT_MIN;
        for (int i = 0 ; i < s.size(); i++) {
            if(normal.count(s[i])) {
                count++;
            }
            while(count > k) {
                if (normal.count(s[left])) count--;
                left++;
            }
            res = max(res, right - left + 1);
            right++;
        }

        // while (right < n) {
        //     if (normal.count(s[right])) {
        //         count++;
        //     }
        //     //res = max(res, right - left + 1);
        //     while (count > k && left <= right) {
        //         if (normal.count(s[left])) count--;
        //         left++;
        //     }
        //     res = max(res, right - left + 1);
        //     right++;
        // }
        return res;
    }
};

//Web Pagination

class Solution{
public:
    vector<vector<string>> displayOrders(int n) {
        vector<vector<string>> res;
        vector<string> out;
        vector<string> pool;
        for (int i = 1; i <= n; i++) {
            pool.push_back("P" + to_string(i));
        }
        recursion(n, 0, pool, out, res);
        return res;
    }

    void recursion(int p, int d, vector<string> pool, vector<string> &out, vector<vector<string>> &res) {
        if (p == 0 && d == 0) {
            res.push_back(out);
            return ;
        }
        int size = pool.size();
        for (int i = 0; i < size; i++) {
            string item = pool[i];
            string type = item.substr(0, 1);
            string sid = item.substr(1);
            int id = stoi(sid);

            if (type == "P") {
                out.push_back(item);
                vector<string> temp_pool = pool;
                temp_pool.push_back("D"+sid);
                temp_pool.erase(temp_pool.begin() + i);
                recursion(p - 1, d + 1, temp_pool, out, res);
                out.pop_back();
            }

            if (type == "D") {
                out.push_back(item);
                vector<string> temp_pool = pool;
                temp_pool.erase(temp_pool.begin() + i);
                recursion(p, d - 1, temp_pool, out, res);
                out.pop_back();
            }
        }
    }
};

int main() {
    int n = 4;
    Solution obj;
    auto ret = obj.displayOrders(n);
    for (int i = 0; i < ret.size(); i++) {
        for (int j = 0; j < ret[i].size(); j++) {
            cout << ret[i][j] << " ";
        }
        cout << " " << endl;
    }
}
