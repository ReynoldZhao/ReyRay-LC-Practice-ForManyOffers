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
    int maxSubArray(vector<int>& nums) {
        int res = INT_MIN;
        int temp = 0;
        for(int i=0;i<nums.size();i++){
            temp = max(temp + nums[i],nums[i]);
            res = max(res,temp)
		}
		return res;
    }
};

class Solution {
public:
    int maxSubArray(vector<int>& nums) {
    	if(nums.empty()) return 0;
		int res = helper(nums,0,nums.size()-1);
		return res;
    }
    int helper(vector<int>& nums,int left,int right){
    	if(left>=right) return nums[left];
    	int mid = left + (right - left)/2;
    	int lmax = helper(nums,left,mid);
    	int rmax = helper(nums,mid+1,right);
    	int mmax = nums[mid],t = mmax;
    	for(int i = mid-1;i>=left;i--){
    		t = t + nums[i];
    		mmax = max(mmax,t);
		}
		int t = mmax;
		for(int i = mid+1;i<=right;i++){
			t = t + num[i];
			mmax = max(t,mmax);
		}
		return max(mmax,max(lmax,rmax));
	}
};
