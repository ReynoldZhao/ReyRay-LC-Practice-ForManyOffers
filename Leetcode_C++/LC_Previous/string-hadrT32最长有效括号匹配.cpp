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
    int longestValidParentheses(string s) {
		vector<int> dp(s.length(),0);
		int m = 0;
		for(int i=1;i<s.length();i++){
			if(s[i]==')'){
				if(s[i-dp[i-1]-1]=='(') dp[i] = dp[i-1]+2;
				dp[i]+=dp[i-dp[i]];
				m = max(m,dp[i]);
			} 
		}
		return m;
    }
};
