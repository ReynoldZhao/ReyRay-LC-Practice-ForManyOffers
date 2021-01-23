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