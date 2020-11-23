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
using namespace std;

class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        if(nums.empty()) return 0;
		int slow=0,fast=0;
		while(fast<nums.size()){
			if(nums[slow]==nums[fast]) fast++;
			else {
                nums[slow] = nums[fast];
				slow++;
				fast++;
			}
		}
		return slow+1;
    }
};
