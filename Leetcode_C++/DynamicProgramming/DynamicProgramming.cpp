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

class SolutionT152 {
public:
    int maxProduct(vector<int>& nums) {
        int n = nums.size();
        vector<int> maxVal(n, 0); //以i结尾的最大乘积子数组
        vector<int> minVal(n, 0); //以i结尾的最小乘积子数组
        maxVal[0] = nums[0] ; minVal[0] = nums[0];
        int res = nums[0];
        //typical of me 过多考虑不符合的情况
        for (int i = 1; i < n; i++) {
            if (nums[i] > 0) {
                maxVal[i] = maxVal[i-1] > 0 ? maxVal[i-1] * nums[i] : nums[i];
                minVal[i] = minVal[i-1] < 0 ? minVal[i-1] * nums[i] : 0; 
            } else if (nums[i] == 0) {
                maxVal[i] = 0;
                minVal[i] = 0;
            } else {
                maxVal[i] = minVal[i-1] < 0 ? minVal[i-1] * nums[i] : 0;
                minVal[i] = maxVal[i-1] > 0 ? maxVal[i-1] * nums[i] : nums[i];
            }
            res = max(res, max(maxVal[i], minVal[i]));
        }
        //这多合理，管你0不0，正不正，负不负，不符合就等于自己
        for (int i = 1; i < n; ++i) {
            f[i] = max(max(f[i - 1] * nums[i], g[i - 1] * nums[i]), nums[i]);
            g[i] = min(min(f[i - 1] * nums[i], g[i - 1] * nums[i]), nums[i]);
            res = max(res, f[i]);
        }
        return res;
    }
};

//最长公共子序列 子串更简单
class SolutionT1143 {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int s1 = text1.size(), s2 = text2.size();
        vector<vector<int> > dp(s1 + 1, vector<int> (s2 + 1, 0));
        for (int i = 1; i <= s1; i++) {
            for (int j = 1; j <= s2; j++) {
                if (text1[i - 1] == text2[j - 1]) dp[i][j] = dp[i-1][j-1] + 1;
                else dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
        return dp[s1][s2];
    }
};

class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        int[][][] dp = new int[][][];
        dp[i][k][0] = max(dp[i-1][k][0], d[i-1][k][1] + prices[i]);
        dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]);
    }

    int maxProfitUnlimited(vector<int>& prices) {
        int res = 0;
        for (int i = 1; i < prices.size(); i++) {
            if (prices[i] > prices[i-1]) {
                res += prices[i] - prices[i-1];
            }
        }
        return res;
    }

    int maxProfit_k_any(int max_k, int[] prices) {
    int n = prices.length;
    if (max_k > n / 2) 
        return maxProfit_k_inf(prices);

    int[][][] dp = new int[n+1][max_k + 1][2];
    for (int i = 0; i <= n; i++) 
        for (int k = max_k; k >= 1; k--) {
            if (i - 1 == -1) { 
                dp[0][k][0] = 0;
                dp[0][k][1] = INT_MIN;
                dp[i][0][0] = 0
                dp[i][0][1] = INT_MIN;
            }
            dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i-1]);
            dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i-1]);     
        }
    return dp[n - 1][max_k][0];
}
};

class SolutionT813 {
public:
    double largestSumOfAverages(vector<int>& A, int K) {
        int n = A.size();
        vector<double> sums(n + 1);
        vector<vector<double>> dp(n, vector<double>(K));
        for (int i = 0; i < n; ++i) {
            sums[i + 1] = sums[i] + A[i];
        }
        for (int i = 0; i < n; ++i) {
            dp[i][0] = (sums[n] - sums[i]) / (n - i);
        }    
        for (int k = 1; k < K; ++k) {
            for (int i = 0; i < n - 1; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    dp[i][k] = max(dp[i][k], (sums[j] - sums[i]) / (j - i) + dp[j][k - 1]);
                }
            }
        }
        return dp[0][K - 1];
    }
};

class SolutionT727 {
public:
    string minWindow(string S, string T) {
        int m = S.size(), n = T.size(), start = -1, minLen = INT_MAX;
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, -1));
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= min(i, n); j++) {
                dp[i][j] = (S[i-1] == T[j-1]) ? dp[i-1][j-1]:dp[i-1][j];
            }
        }

    }
};

class SolutionT978 {
public:
    int maxTurbulenceSize(vector<int>& arr) {
        int res = 1;
        vector<int> dec(arr.size(), 1), inc(arr.size(), 1);
        for (int i = 1; i < arr.size(); i++) {
            if (arr[i-1] > arr[i]) {
                dec[i] = inc[i-1] + 1;
            } else if (arr[i-1] < arr[i]) {
                inc[i] = dec[i-1] + 1;
            }
            res = max(res, max(dec[i], inc[i]));
        }
        return res;
    }

//众所周知一维数组可以压缩空间
    int maxTurbulenceSize(vector<int>& arr) {
        int res = 1, n = arr.size(), inc = 1, dec = 1;
        for (int i = 1; i < n; ++i) {
            if (arr[i] < arr[i - 1]) {
                dec = inc + 1;
                inc = 1;
            } else if (arr[i] > arr[i - 1]) {
                inc = dec + 1;
                dec = 1;
            } else {
                inc = 1;
                dec = 1;
            }
            res = max(res, max(inc, dec));
        }
        return res;
    }
};

//backpack 1
class Solution {
public:
    /**
     * @param m: An integer m denotes the size of a backpack
     * @param A: Given n items with size A[i]
     * @return: The maximum size
     */
    int backPack(int m, vector<int> &A) {
        // write your code here
        sort(A.begin(), A.end());
        int left = 0, sum = 0, i = 0, res = 0;
        while (i < A.size()) {
            sum += A[i];
            i++;
            while (sum > m && left <= i) {
                sum -= A[left];
                left++;
            }
            res = max(res, sum);
        }
        return res;
    }

    int backPack(int m, vector<int> &A) {
        vector<int> dp(m+1, 0);
        int n = A.size();
        if (A.size() == 0 || m == 0) {
            return 0;
        }
        for (int i = 0; i < n; i++) {
            for (int j = m; j >= A[i]; j--) {
                //dp[i][j] = max(dp[i-1][j], dp[i-1][j-A[i]] + A[i]);
                dp[j] = max(dp[j], dp[j - A[i]] + A[i]);
            }
        }
        //return dp[i-1][m];
        return dp[m];
    }
};

//backpack 2
class Solution {
public:
    /**
     * @param m: An integer m denotes the size of a backpack
     * @param A: Given n items with size A[i]
     * @param V: Given n items with value V[i]
     * @return: The maximum value
     */
    int backPackII(int m, vector<int> &A, vector<int> &V) {
        // write your code here
        vector<int> dp(m + 1, 0);
        int n = A.size();
        if (A.size() == 0 || m == 0) {
            return 0;
        }
        for (int i = 0; i < n; i++) {
            for (int j = m; j >= A[i]; --j) {
                dp[j] = max(dp[j], dp[j - A[i]] + V[i]);
            }
        }
        return dp[m];
    }
};
