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

int main(){
	while(1){
		int n;
		cin>>n;
		int height[n];
		for(int i=0;i<n;i++){
			cin>>height[i];
		}
		int dp[n][3];
		dp[0][0] = height[0],dp[0][1] = 0, dp[0][2] = 0;
		dp[1][0] = height[1] + min(dp[0][0],dp[0][1]);
		dp[1][1] = dp[0][0];
		dp[1][2] = dp[0][0];
		for(int i=2;i<n;i++){
			dp[i][0] = height[i] + min(dp[i-1][0],min(dp[i-1][1],dp[i-2][2]));
			dp[i][1] = dp[i-1][0];
			dp[i][2] = dp[i-1][0];
		}
		int result1 = min(dp[n-1][0],min(dp[n-1][1],dp[n-1][2]));
		int result2 = min(dp[n-2][2],result1);
		cout<<result2<<endl;
	}
} 
