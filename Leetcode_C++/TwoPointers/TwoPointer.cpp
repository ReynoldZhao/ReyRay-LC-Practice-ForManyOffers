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