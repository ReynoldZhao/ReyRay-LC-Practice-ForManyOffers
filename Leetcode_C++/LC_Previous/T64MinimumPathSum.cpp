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
using namespace std;

class Solution {
public:
    int minPathSum(vector<vector<int> >& grid) {
        int M = grid.size();
		int N = grid[0].size();
		int dp[M][N];
		dp[0][0] = grid[0][0];
		for(int i=1;i<N;i++) dp[0][i] += dp[0][i-1];
		for(int i=1;i<M;i++){
			for(int j=0;j<N;j++){
				if(j==0) dp[i][j] = dp[i-1][j] + grid[i][j];
				dp[i][j] = min(dp[i-1][j]+grid[i][j],dp[i][j-1]+grid[i][j]);
			}
		}
		int min=dp[M-1][0];
		for(int i=1;i<N;i++){
			if(dp[M-1][i]<min) min = dp[M-1][i];
		}
		return min;
    }
};
