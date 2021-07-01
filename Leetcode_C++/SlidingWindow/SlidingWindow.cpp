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
#include<hash_map>
#include<deque>
using namespace std;

class SolutionT3{
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<int, int> map;
        int left = 0, res = 0;
        for (int i = 0; i < s.size(); i++) {
            if (map.count(s[i]) && map[s[i]] > left) {
                left = map[s[i]];
            }
            map[s[i]] = i;
            res = max(res, i - left);
        }
        return res;
    }
};

class SolutionT159 {
public:
    int lengthOfLongestSubstringTwoDistinct(string s) {
        unordered_map<int, int> map;
        int left = 0, res = 0;
        for (int i = 0; i < s.size(); i++) {
            map[s[i]]++;
            while(map.size() > 2) {
                if (--map[s[left]] == 0) map.erase(s[left]);
                left++;
            }
            res = max(res, i - left + 1);
        }
    }
};

class SolutionT239 {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;
        deque<int> dq;
        for (int i = 0; i < k; i++) {
            while(!dq.empty() && nums[i] >= nums[dq.back()]) {
                dq.pop_back();
            }
            dq.push_back(i);
        }
        res.push_back(nums[dq.front()]);
        for (int i = k; i < nums.size(); i++) {
            while(!dq.empty() && nums[i] >= nums[dq.back()]) dq.pop_back();
            if (!dq.empty() && dq.front() <= i - k) dq.pop_front();
            dq.push_back(i);
            res.push_back(nums[dq.front()]);
        }
        return res;
    }
};

class SolutionT995 {
public:
    int minKBitFlips(vector<int>& A, int K) {
        return helper(A, K) - helper(A, K - 1);
    }
    int helper(vector<int>& A, int K) {
        int n = A.size(), res = 0, left = 0;
        unordered_map<int, int> map;
        for (int i = 0; i < n; i++) {
            if (map[A[i]] == 0) K--;
            map[A[i]]++;
            while(K < 0) {
                if (--map[A[left]] == 0) K++;
                left++;
            }
            res = i - left + 1;
            //此时这个窗口的长度就代表了此时最多有k个不同数字的子数组的个数，将其加入结果 res
            //以i开头 最多到left结尾 最多有k个不同数字的子数组个数
        } 

    }
};

class SolutionT73 {
public:
    string minWindow(string s, string t) {
        int cnt = t.size();
        unordered_map<int, int> map;
        for (auto tmp : t) map[tmp]++;
        int left = 0, right = -1, res = INT_MAX;
        string res_str = "";
        for (int i = 0 ; i < s.size(); i++) {
            if (map.count(s[i])) {
                --map[s[i]];
                if(map[s[i]] == 0) cnt--;
            }
            while(cnt == 0) {
                if (i - left + 1 < res) {
                    res_str = s.substr(left, i - left +1);
                }
                if (map.count(s[left])) {
                    if (++map[s[left]] > 0) cnt++;
                    left++;
                }
            }
        }
        return res_str;
    }
};

class SolutionT239 {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;
        deque<int> dp;
        int max_pos = 0, max_val = INT_MIN;
        for (int i = 0; i < k; i++) {
            while(!dp.empty() && nums[i] >= nums[dp.back()]) dp.pop_back();
            dp.push_back(i);
        }
        res.push_back(nums[dp.front()]);
        for (int i = k; i < nums.size(); i++) {
            while(!dp.empty() && nums[i] >= nums[dp.back()]) dp.pop_back();
            if (!dp.empty() && dp.front() <= i - k) dp.pop_front();
            dp.push_back(i);
            res.push_back(dp.front());
        )
        return res;
    }
};