#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<utility>

class Solution {
public:
    vector<vector<int> > threeSum(vector<int>& nums) {
        vector<vector<int> > res;
        sort(nums.begin(),nums.end());
        if(nums.empty()||nums.back()<0||nums.front()>0) return res;
		int left,right;
		for(int i=0;i<nums.size();i++){
			if(nums[i]>0) break;
			if(i>0&&nums[i]==nums[i-1]) continue;
			int target = 0 - nums[i];
			left = i+1,right = nums.size()-1;
			while(left<right){
				if(nums[left]+nums[right]==target){
					vector<int> temp = {nums[i],nums[right],nums[left]};
					res.push_back(temp);
					while(left<right&&nums[left]==nums[left+1]) left++
				}
			}
		}
		return res;
    }
};
