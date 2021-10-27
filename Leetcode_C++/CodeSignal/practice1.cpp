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
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        vector<int> out, visited(nums.size(), 0);
        backtracking(res, out, nums, 0, visited);
        return res;
    }
    void backtracking(vector<vector<int>> &res, vector<int>& out, vector<int>& nums, int k, vector<int> &visited) {
        if (k == nums.size()) {
            res.push_back(out);
            return ;
        }
        for (int i = 0; i < nums.size(); i++) {
            if(visited[i] == 1 || (i > 0 && nums[i] == nums[i-1] && visited[i-1] == 0)) continue;
            visited[i] = 1;
            out.push_back(nums[i]);
            backtracking(res, out, nums, k + 1, visited);
            out.pop_back();
            visited[i] = 0;
        }
        return ;
    }

    vector<bool> boundedRatio(vector<int> a, int l, int r) {
        vector<bool> res(a.size(), false);
        for (int i = 0 ; i < a.size() ; i++) {
            double div = a[i] / (i + 1);
            if (l <= div && div <= r && (a[i] % (i + 1)) == 0) res[i] = true;
        }
        return res;
    }


    long long countDecreasingSubarrays(vector<int> arr) {
        for (int i = 0; i < arr.size(); i++) {
            for (int j = i + 1; j < arr.size(); j++) {

            }
        }
        vector<long long> dp(arr.size(), 1);
        dp[0] = 1;
        for (int i = 1; i < arr.size(); i++) {
            if (arr[i] < arr[i - 1]) {
                dp[i] = dp[i] + dp[i-1];
            }
        }
        long long res = 0;
        for (int i = 0; i < arr.size(); i++) {
            res += dp[i];
        }
        return res;
    }
};

class Solution{
    bool isInc(vector<int> num) {
        for (int i = 0; i < num.size() - 1; i++) {
            if (num[i] >= num[i + 1]) return false;
        }
        return true;
    }

    bool makeIncreasing(vector<int> nums) {
        int idx = 0, count = 0;
        for (int i = 0; i < nums.size() - 1; i++) {
            if (nums[i] >= nums[i + 1]) {
                idx = i;
                count++;
            }
        }
        if (count == 0) return true;
        if (count > 1) return false;
        int org_num = nums[idx];
        string t = to_string(nums[idx]);
        for (int i = 0; i < t.size(); i++) {
            string cur = t;
            swap(cur[i], cur[(i + 1) % t.size()]);
            while (cur[0] == '0') cur = cur.substr(1);
            cout << cur + " first " << endl;
            int newNum = stoi(cur);
            nums[idx] = newNum;
            if (isInc(nums)) return true;
            nums[idx] = org_num;
        }
        idx = idx + 1;
        if (idx < nums.size()) {
            org_num = nums[idx ];
            string t = to_string(nums[idx]);
            for (int i = 0; i < t.size(); i++) {
                string cur = t;
                swap(cur[i], cur[(i + 1) % t.size()]);
                while (cur[0] == '0') cur = cur.substr(1);
                cout << cur + " second " << endl;
                int newNum = stoi(cur);
                nums[idx] = newNum;
                if (isInc(nums)) return true;
                nums[idx] = org_num;
            }
        }
        return false;
    }

    string mergeStrings(string s1, string s2) {
        unordered_map<char, int> set1, set2;
        for (auto a : s1) set1[a]++;
        for (auto a : s2) set2[a]++;
        int m = s1.size(), n = s2.size(), p = 0, q = 0;
        string res = "";
        while (p < m && q < n) {
            if (set1[s1[p]] == set2[s2[q]]) {
                if (s1[p] < s2[q]) {
                    res += s1[p++];
                }
                else {
                    res += s2[q++];
                }
            } else if (set1[s1[p]] < set2[s2[q]]){
                res += s1[p++];
            } else {
                res += s2[q++];
            }
        }
        while (p < m) res += s1[p++];
        while (q < n) res += s2[q++];
        return res;
    }
};

class SolutionMineSweeper{
    int dirX[8] = {1, 0, -1, 0, 1, 1, -1, -1};
    int dirY[8] = {0, -1, 0, 1, 1, -1, 1, -1};

    bool check(int curX, int curY, int X, int Y) {
        if (curX <0 || curX >= X || curY < 0 || curY >= Y) return false;
        return true;
    }

    void infect(vector<vector<bool>> &field, vector<vector<int>> &res, int x, int y) {
        if (res[x][y] >= 0) return ;
        int m = field.size(), n = field[0].size();
        int near = 0;
        for (int i = 0; i < 8; i++) {
            int curX = x + dirX[i], curY = y + dirY[i];
            if (check(curX, curY, m, n) && field[curX][curY] == true) {
                near++;
            }
        }
        if (near > 0) {
            res[x][y] = near;
            return ;
        } else {
            res[x][y] = 0;
            for (int i = 0; i < 8; i++) {
                int curX = x + dirX[i], curY = y + dirY[i];
                if (check(curX, curY, m, n)){
                    infect(field, res, curX, curY);
                }
            }
        }
    }

    vector<vector<int>> minesweeperClick(vector<vector<bool>> field, int x, int y) {
        int m = field.size(), n = field[0].size();
        vector<vector<int>> res(m, vector<int> (n, -1));
        infect(field, res, x, y);
        return res;
    }
};

class SolutionCS {
    long long subarraysCountBySum(vector<int> arr, int k, long long s) {
        int n = arr.size();
        vector<long long> sum(n, 0);
        unordered_map<long long, vector<int>> map;
        map[0] = {-1};
        for (int i = 0; i < n; i++) {
            sum[i] = arr[i] + (i > 0 ? sum[i-1] : 0);
            map[sum[i]].push_back(i);
        }
        long long res = 0;
        for (int i = 0; i < n; i++) {
            if (i == 0 && sum[i] == s) {
                res++;
                continue;
            }
            long long cur = sum[i];
            if (map.count(cur - s) == 0) continue;
            auto pos = map[cur - s];
            for (auto p : pos) {
                if (p > i) break;
                cout << p<< endl;
                cout << i << endl;
                if (i - p + 1 <= k) res++;
            }
        }
        return res;
    }

//Map + Deque
    long long subarraysCountBySum(vector<int> arr, int k, long long s) {
        unordered_map<long long, deque<int>> map;
        long long sum = 0, res = 0;
        map[0].push_back(-1);
        for (int i = 0; i < arr.size(); i++) {
            sum += arr[i];
            if (map.count(sum - s)) {
                while (i - map[sum - s].front() > k) map[sum - s].pop_front();
                res += map[sum - s].size();
            }
            // 判空 while (i - map[sum].front() > k) map[sum].pop_front();
            map[sum].push_back(i);
        }
        return res;
    }
};