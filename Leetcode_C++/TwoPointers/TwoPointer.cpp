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

class SolutionT287 {
public:
    int findDuplicate(vector<int>& nums) {
        int temp = nums[0], n = nums.size(), res = nums[0];
        for (int i = 1; i < n; i++) {
            temp = temp ^ nums[i];
            if (!temp) {
                res = nums[i];
                break;
            }
        }
        return res;
    }
};

class SolutionT713 {
public:
//这其实是一种brute-force的双指针
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        int res = 0, left = 0, right = 0, accum = 1, n = nums.size();
        for (int i = 0; i < n; i++) {
            int right = i, accum = 1;
            // 向右
            while (right < n && accum * nums[right] < k) {
                accum = accum * nums[right];
                right++;
                res++;
            }
        }
        return res;
    }

    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        int left = 0, prod = 1, n = nums.size(), res = 0;
        for (int i = 0; i < n; i++) {
            prod = prod * nums[i];
            //向左
            while (left <= i && prod >= k) prod /= nums[left++];
            res += i - left + 1; //left --- i 中的每一个子数组 + 新遍历到i，即为新增子数组个数。
            //这个子数组个数新增真的很巧
            // 4 5 6 7 --- 8    87 876 8765 87654
            //只考虑当前的状态
        }
        return res;
    }
};

class SolutionT253 {
public:
    int minMeetingRooms(vector<vector<int>>& intervals) {
        map<int, int> m;
        for (auto interval : intervals) {
            m[interval[0]]++;
            m[interval[1]]--;
        }
        int res = 0, room = 0;
        for (auto a : m) {
            room = room + a.second;
            res = max(res, room);
        }
        return res;
    }

    int minMeetingRooms(vector<vector<int>>& intervals) {
        vector<int> starts, ends;
        int res = 0, endpos = 0;
        for (auto a : intervals) {
            starts.push_back(a[0]);
            ends.push_back(a[1]);
        }
        sort(starts.begin(), starts.end());
        sort(ends.begin(), ends.end());
        for (int i = 0; i < intervals.size(); i++) {
            if (starts[i] < ends[endpos]) res++;
            else endpos++;
        }
    }
};

class Solution {
public:
    vector<vector<int>> intervalIntersection(vector<vector<int>>& firstList, vector<vector<int>>& secondList) {
        int n = firstList.size();
        vector<int> start, end; 
        vector<vector<int>> res;
        for (int i = 0; i < n; i++) {
            start.push_back(firstList[i][0]); 
            end.push_back(firstList[i][1]); 
        }
        int cur_end = 0;
        for (int i = 0; i < n; i++) {
            if (secondList[i][0] < firstList[cur_end][0]) {
                res.push_back(firstList[cur_end]);
            }
        }
    }
};