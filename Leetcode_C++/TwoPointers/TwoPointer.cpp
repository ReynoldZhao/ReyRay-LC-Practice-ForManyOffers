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

class SolutionT11 {
public:
    int maxArea(vector<int>& height) {
        int l = 0, r = height.size()-1;
        int res = 0;
        while(l < r) {
            res = max(res, min(height[l], height[r]) * (r - l));
            if (height[l] < height[r]) l++;
            else r--;
        }
        return res;
    }
};

//接雨水
class Solution {
public:
    int trap(vector<int>& height) {
        int res = 0, l = 0, r = height.size() - 1;
        while (l < r) {
            int mn = min(height[l], height[r]);
            if (mn == height[l]) {
                ++l;
                while (l < r && height[l] < mn) {
                    res += mn - height[l++];
                }
            } else {
                --r;
                while (l < r && height[r] < mn) {
                    res += mn - height[r--];
                }
            }
        }
        return res;
    }
};

//最小覆盖子串 先不要考虑那么多不符合情况，把主干解决办法写出了，不符合情况有更简单的解决办法
//就比如这里不在t里的字符，初始为0，直接--，之后移除窗口后++，永远值不会为正，不影响cnt
class SolutionT76 {
public:
    string minWindow(string s, string t) {
        unordered_map<int, int> bucket;
        for (int i = 0; i < t.size(); i++) bucket[t[i]]++;
        int left = 0, minLen = INT_MAX, n = s.size(), cnt = 0;
        string res = "";
        for (int i = 0; i < n; i++) {
            if (--bucket[s[i]] >= 0) ++cnt;
            while (cnt == t.size()) {
                if (minLen > i - left + 1) {
                    minLen = i - left + 1;
                    res = s.substr(left, minLen);
                }
                if (++bucket[s[left]] > 0) --cnt;
                ++left;
            }
        }
        return res;
    }
};