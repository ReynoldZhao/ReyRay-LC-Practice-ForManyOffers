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

class SolutionT658 {
public:
    //反向思维，从数组中去除n-k个元素，肯定是从头尾去除
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        vector<int> res = arr;
        while (res.size() > k) {
            if (x - res.front() <= res.back() - x) {
                res.pop_back();
            } else {
                res.erase(res.begin());
            }
        }
        return res;
    }

    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        auto itr = lower_bound(arr.begin(), arr.end(), x);
        int index = itr - arr.begin();
        vector<int> res({arr[index]});
        int pre = index - 1 >= 0 ? index - 1:-1, next = index+1 < arr.size()?:index+1:arr.size();
        while(k > 0) {
            if (pre < 0 || next >= arr.size()) {
                res.push_back(arr[pre<0?next++:pre--]);
                k--;
            }
            if (abs(x - arr[pre]) <= abs(arr[next] - x)) {
                res.push_back(arr[pre--]);
            } else {
                res.push_back(arr[next++]);
            }
            k--;
        }
        return res;
    }

    //巧妙二分, 这个设计胎牛皮了
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        int l = 0, r = arr.size() - k;
        while (l < r) {
            int mid = l + (r - l)/2;
            if (x - arr[mid] > arr[mid] - x) l = mid + 1;
            else r = mid;
        }
        return vector<int>(arr.begin() + l, arr.begin() + l + k);
    }
};

class SolutionT410 {
public:
    int splitArray(vector<int>& nums, int m) {
        int left = 0, right = 0;
        for (int i = 0; i < nums.size(); i++) {
            left = max(left, nums[i]);
            right += nums[i];
        }
        int res = INT_MAX;
        while (left < right) {
            int mid = left + (right - left)/2;
            if (canSplit(nums, mid, m)) {
                res = min(res, mid);
                right = mid;
            }
            else left = mid + 1;
        }
        return right;
    }

    bool canSplit(vector<int>& nums, int tar, int m) {
        int curSum = 0, cnt = 0;
        for (int i = 0; i < nums.size(); i++) {
            curSum += nums[i];
            if(curSum > tar) {
                cnt++;
                curSum = nums[i];
            }
            if (cnt > m) return false;
        }
        return true;
    }
};