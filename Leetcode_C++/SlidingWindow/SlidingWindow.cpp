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