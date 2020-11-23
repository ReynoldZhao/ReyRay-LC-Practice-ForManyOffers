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
    int searchInsert(vector<int>& nums, int target) {
        int cur=0;
        if(nums.empty()) return 0;
		while(cur < nums.size()&&nums[cur]<target){
			cur++;	
		}
		if(nums[cur]==target) return cur;
		else return cur;
    }
};
