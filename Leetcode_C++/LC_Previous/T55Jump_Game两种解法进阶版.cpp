#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<utility>
#include<cstring>
using namespace std;
// DP
class Solution {
public:
    bool canJump(vector<int>& nums) {
		int n = nums.size();
		int dp[n];//dp表示当前点所能到达的最远距离 
		memset(dp,0,sizeof(dp));
		dp[0] = nums[0];
		for(int i=1;i<n;i++){
			if(i<=dp[i-1]) dp[i] = max(dp[i-1],i+nums[i]);
			else dp[i] = dp[i-1]
		}
		return dp[n-1]> = (n-1)
    }
};
//Greedy
class Solution {
public:
    bool canJump(vector<int>& nums) {
		int n = nums.size();
		int maxstride = 0;//表示整个数组能达到的最长步长 
		for(int i=0;i<n;i++){
			if(i>maxstride||maxstride>=n-1) break;
			maxstride = max(maxstride,i+num[i]);
		}
		return maxstride>=(n-1);
    }
};
