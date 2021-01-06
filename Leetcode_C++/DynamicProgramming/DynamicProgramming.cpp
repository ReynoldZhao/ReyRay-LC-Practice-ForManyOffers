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

class SolutionT410 {
public:
    //dp[i][j]dp[i][j] 表示将数组中前j个数字分成i组所能得到的最小的各个子数组中最大值
    int splitArray(vector<int>& nums, int m) {
        int n = nums.size();
        vector<long> sums(n + 1);
        vector<vector<long>> dp(m + 1, vector<long>(n + 1, LONG_MAX));
        dp[0][0] = 0;
        for (int i = 1; i <= n; ++i) {
            sums[i] = sums[i - 1] + nums[i - 1];
        }
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                for (int k = i - 1; k < j; ++k) { //前k个数分成i-1组，k就从i-1开始吧，i-2个数分成i-1组，值为空
                    long val = max(sums[j] - sums[k], dp[i-1][k]);
                    dp[i][j] = min(dp[i][j], val);
                }
            }
        }
        return dp[m][n];
    }
};

class SolutionT32 {
public:
//32. Longest Valid Parentheses 最长有效括号
    int longestValidParentheses(string s) {
        int n = s.size(), res = 0;
        vector<int> dp(n+1);//dp[i] 表示以s[i - 1] 结尾的最长有效括号字符串的长度
        for (int i = 1; i < n; i++) {
            int j = i - dp[i-1] - 2; // [j]  "(" (   dp[i-1]      ) ")"假设这里为i-1; j是前一个有效的开始
            if (s[i-1] == '(' || j < 0 || s[j] == '(') {
                dp[i] = 0;
            } else {
                dp[i] = dp[i-1] + 2 + dp[j];
                res = max(res, dp[i]);
            }
        }
        return res;
    }
};