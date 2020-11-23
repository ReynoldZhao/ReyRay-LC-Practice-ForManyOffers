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
#include<unordered_set>
#include<deque>
#include<multiset>
using namespace std;


class SolutionT123 {
public:
    int maxProfit(int k, vector<int>& prices) {
        if (prices.empty()) return 0;
        int n = prices.size();
        vector<vector<int>> g(n, vector<int>(k+1,0));
        vector<vector<int>> l(n, vector<int>(k+1,0));
        for (int i = 1; i < prices.size(); ++i) {
            int diff = prices[i] - prices[i - 1];
            for (int j = 1; j <= k; ++j) {
                l[i][j] = max(g[i - 1][j - 1] + max(diff, 0), l[i - 1][j] + diff);
                g[i][j] = max(l[i][j], g[i - 1][j]);
            }
        }
        return g[n - 1][k];
    }

    int maxProfit(vector<int> &prices) {
        if (prices.empty()) return 0;
        int n = prices.size();
        vector<int> g(k+1,0);
        vector<int> l(k+1,0);
        for (int i = 1; i < prices.size(); ++i) {
            int diff = prices[i] - prices[i - 1];
            for (int j = 1; j <= k; ++j) {
                l[j] = max(g[j - 1] + max(diff, 0), l[j] + diff);
                g[j] = max(l[j], g[j]);
            }
        }
        return g[k];
    }
};

class SolutionT1450 {
public:
    int busyStudent(vector<int>& startTime, vector<int>& endTime, int queryTime) {
        int res = 0;
        for (int i = 0; i < startTime.size(); i++) {
            if(startTime[i] <= queryTime && endTime[i] >= queryTime) res++;
        }
        return res;
    }
};

class SolutionT1451 {
public:
    string arrangeWords(string text) {
        
    }
};

class SolutionT213 {
public:
    int rob(vector<int>& nums) {
        if(nums.empty()) return 0;
        if(nums.size()<=1) return nums[0];
        if(nums.size()==2) return max(nums[0], nums[1]);
        vector<int> dp(nums.size(),0);
        bool flag = false_type;
        bool flag_bouble = nums[0] == nums[1];
        dp[0] = nums[0], dp[1] = max(nums[0], nums[1]);
        if(dp[1] == dp[0]) flag = true;
        int res = max(nums[0], nums[1]);
        if (!flag_double){
            for(int i=2;i<nums.size();i++){
                if(i == nums.size() - 1 && !){
                    if(flag) dp[i] = dp[i-1];
                    else dp[i] = max(dp[i-1], nums[i]+dp[i-2]);
                }
                else 
                    dp[i] = max(dp[i-1], nums[i]+dp[i-2]);
                res = max(res, dp[i]);
            }
        }
        else {
            for(int i=2;i<nums.size();i++){
                dp[i] = max(dp[i-1], nums[i]+dp[i-2]);
                res = max(res, dp[i]);
            }
        }
        return res; 
    }

    int rob(vector<int>& nums) {
        if(nums.empty()) return 0;
        if(nums.size()<=1) return nums[0];
        if(nums.size()==2) return max(nums[0], nums[1]);
        vector<int> dp(nums.size(),0);
        dp[0] = -nums[0];
        if (nums[0] > nums[1]) {
            dp[1] = -nums[0];
        }
        else {
            dp[1] = nums[1];
        }
        int res = max(nums[0], nums[1]);
        for(int i=2;i<nums.size();i++){
            int temp = 0;
            if(i==nums.size() - 1 && dp[i-2] < 0)) {
                temp = abs(dp[i-1]) > nums[i] ? abs(dp[i-1]):nums[i];
            }
            else {
                int t1 = abs(dp[i-1]);
                int t2 = nums[i] + abs(dp[i-2]);
                temp = max(t1, t2);
                if(temp == t1 && dp[i-1] < 0) dp[i] = -temp;
                else if(temp == t2 && dp[i-2] < 0) dp[i] = -temp;
                else dp[i] = temp;
            }
            res = max(res, temp);
        }
        return res;   
    }
};

class SolutionT95 {
public:
    vector<TreeNode*> generateTrees(int n) {
        if (n == 0) return {};
        vector<vector<vector<TreeNode*>>> memo(n, vector<vector<TreeNode*>>(n));
        return helper(1, n, memo);
    }
    vector<TreeNode*> helper(int start, int end, vector<vector<vector<TreeNode*>>> memo) {
        if (start > end) return {nullptr};
        if (!memo[start - 1][end - 1].empty()) return memo[start - 1][end - 1];
        vector<TreeNode*> res;
        for (int i = start; i <= end; i++) {
            auto left = helper(start, i - 1), right = helper(i + 1, end);
            TreeNode* head = new TreeNode(i);
            for (auto l : left) {
                for (auto r : right) {
                    head->left = l;
                    head->right = r;
                    res.push_back(head);
                }
            }
        }
        return memo[start - 1][end - 1] = res;
    }
};

class SolutionT264 {
public:
    int nthUglyNumber(int n) {
        vector<int> dp(n+1, 0);
        dp[1] = 1;
        i2 = 0， i3 = 0， i5 = 0;
        for (int i = 2; i <= n; i++) {
            dp[i] = min(dp[i-1]*5, min(dp[i-1]*3, dp[i-1]*2));
        }
        return dp[n];
    }
};

class SolutionT1201 {
public:    
   int nthUglyNumber(int k, int A, int B, int C) {
        int lo = 1, hi = 2 * (int) 1e9;
        long a = long(A), b = long(B), c = long(C);
        long ab = a * b / __gcd(a, b);
        long bc = b * c / __gcd(b, c);
        long ac = a * c / __gcd(a, c);
        long abc = a * bc / __gcd(a, bc);
        // a b 的最小公倍数 是 a * b / a和b的最大公因子
        while(lo < hi) {
            int mid = lo + (hi - lo)/2;
            int cnt = mid/a + mid/b + mid/c - mid/ab - mid/bc - mid/ac + mid/abc;
            if(cnt < k) 
                lo = mid + 1;
            else
			   //the condition: F(N) >= k
                hi = mid;
        }
        return lo;
    }
};

class SolutionT44 {
public:
    bool isMatch(string s, string p) {
        int i = 0, j = 0, istar = -1, jstar = -1, m = s.size(), n = p.size();
        while ( i < m ) {
            if( j < n && (s[i] == p[j] || p[j] == '?')) {
                i++;
                j++;
            }
            else if( j < n && p[j] = '*') {
                istar = i;
                jstar = j;
                j++;
            }
            else if(istar > 0){
                ++istar;
                i = istar;
                j = jstar + 1;
            }
            else return false;
        }
        while (j < n && p[j] == '*') j++;
        return j == n - 1;
    }
};
class SolutionT44 {
public:
    bool isMatch(string s, string p) {
        int i = 0, j = 0, istar = -1, jstar = -1, m = s.size(), n = p.size();
        while ( i < m ) {
            if( j < n && (s[i] == p[j] || p[j] == '?')) {
                i++;
                j++;
            }
            else if( j < n && p[j] = '*') {
                istar = i;
                jstar = j;
                j++;
            }
            else if(istar > 0){
                ++istar;
                i = istar;
                j = jstar + 1;
            }
            else return false;
        }
        while (j < n && p[j] == '*') j++;
        return j == n - 1;
    }

    bool isMatch(string s, string p) {
        int m = s.size(), n = p.size();
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true;//都为空
        for (int i = 1; i <= n; i++) {
            if(p[i] == '*') dp[0][i] = dp[0][i-1];//s为空，p为连续的* 
        }
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p[j - 1] == '*') {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                } else {
                    dp[i][j] = ((s[i] == p[j]) || p[j] == '?') && dp[i-1][j-1];
                }
            }
        }
        return dp[m][n];
    }
    //naive 递归
    bool isMatch(string s, string p) {
        if(s.empty()) return p.empty() || (p[0] == '*' && isMatch(s, p.substr(1)));
        if(p.empty()) return false;
        if(s[0] == p[0] && p[0] == '?') return isMatch(s.substr(1), p.substr(1));
        if(p[0] == '*'){
            if (isMatch(s.substr(1), p)) return true;
            if (isMatch(s, p.substr(1))) return true;
        }
        return false;
    }
};

class SolutionT309 {
public:
    int maxProfit(vector<int>& prices) {
        int buy = INT_MIN, pre_buy = 0, sell = 0, pre_sell = 0;
        for (int price : prices) {
            pre_buy = buy;
            buy = max(pre_sell - price, pre_buy);
            pre_sell = sell;
            sell = max(pre_buy + price, pre_sell);
        }
        return sell;
    }

    int maxProfit(vector<int>& prices) {
    int sold = 0, hold = INT_MIN, rest = 0;
    for (int i=0; i<prices.size(); ++i)
    {
        int prvSold = sold;
        sold = hold + prices[i];
        hold = max(hold, rest-prices[i]);
        rest = max(rest, prvSold);
    }
    return max(sold, rest);
};

class SolutionT338 {
public:
    vector<int> countBits(int num) {
        if (num == 0) return {0};
        vector<int> res{0, 1};
        int k = 2, i = 2;
        while (i < num) {
            for (i = pow(2, k-1); i < pow(2, k); i++) {
                if(i > num) break;
                int t = (pow(2, k) - pow(2, k - 1)) / 2;
                if (i < pow(2, k-1) + t) res.push_back(res[i - t]);
                else res.push_back(res[i-t] + 1);
            }
            ++k;
        }
        return res;
    }

    vector<int> countBits(int num) {
        vector<int> res;
        for (int i = 0; i <= num; ++i) {
            res.push_back(bitset<32>(i).count());
        }
        return res;
    }
//从1开始，遇到偶数时，其1的个数和该偶数除以2得到的数字的1的个数相同，遇到奇数时，其1的个数等于该奇数除以2得到的数字的1的个数再加1，
    vector<int> countBits(int num) {
        vector<int> res{0};
        for (int i = 1; i <= num; ++i) {
            if (i % 2 == 0) res.push_back(res[i / 2]);
            else res.push_back(res[i / 2] + 1);
        }
        return res;
    }
};

class SolutionT97 {
public:
//DP
    bool isInterleave(string s1, string s2, string s3) {
        if (s1.size() + s2.size() != s3.size()) return false;
        int n1 = s1.size(), n2 = s2.size();
        vector<vector<bool>> dp(n1 + 1, vector<bool> (n2 + 1)); 
        dp[0][0] = true;
        for (int i = 1; i <= n1; ++i) {
            dp[i][0] = dp[i - 1][0] && (s1[i - 1] == s3[i - 1]);
        }
        for (int i = 1; i <= n2; ++i) {
            dp[0][i] = dp[0][i - 1] && (s2[i - 1] == s3[i - 1]);
        }
        for (int i = 1; i <= n1; ++i) {
            for (int j = 1; j <= n2; ++j) {
                dp[i][j] = (dp[i - 1][j] && s1[i - 1] == s3[i - 1 + j]) || (dp[i][j - 1] && s2[j - 1] == s3[j - 1 + i]);
            }
        }
        return dp[n1][n2];
    }
//DFS
    bool isInterleave(string s1, string s2, string s3) {
        if (s1.size() + s2.size() != s3.size()) return false;
        unordered_set<int> s;
        return helper(s1, 0, s2, 0, s3, 0, s);
    }

    bool helper(string& s1, int i, string& s2, int j, string& s3, int k, unordered_set<int>& s) {
        int key = i * s3.size() + j;
        if (i == s1.size()) return s2.substr(j) == s3.substr(k);
        if (j == s2.size()) return s1.substr(i) == s3.substr(k);
        if ((s1[i] == s3[k] && helper(s1, i + 1, s2, j, s3, k + 1, s)) || 
            (s2[j] == s3[k] && helper(s1, i, s2, j + 1, s3, k + 1, s))) return true;
        s.insert(key);
        return false; 
    }

    bool isInterleave(string s1, string s2, string s3) {
        if (s1.size() + s2.size() != s3.size()) return false;
        int n1 = s1.size(), n2 = s2.size(), n3 = s3.size(), k = 0;
        unordered_set<int> s;
        queue<int> q{0};
        while (!q.empty() && k < s3.size()) {
            int i = q.front()/ n3, j = q.front()%3;q.pop();
            if ( i < n1 && s1[i] == s3[k]) {
                int key = (i + 1) * n3 + j;
                if(!s.count(key)) {
                    if (!s.count(key)) {
                        s.insert(key);
                        q.push(key);
                }
            }
            if (j < n2 && s2[j] == s3[k]) {
                    int key = i * n3 + j + 1;
                    if (!s.count(key)) {
                        s.insert(key);
                        q.push(key);
                    }
                }
            }
            k++;
        }
        return !q.empty() && k == n3;
    }
};

class SolutionT87 {
public:
//递归 简单的说，就是 s1 和 s2 是 scramble 的话，那么必然存在一个在 s1 上的长度 l1，将 s1 分成 s11 和 s12 两段，同样有 s21 和 s22，那么要么 s11 和 s21 是 scramble 的并且 s12 和 s22 是 scramble 的；要么 s11 和 s22 是 scramble 的并且 s12 和 s21 是 scramble 的。
    bool isScramble(string s1, string s2) {
        if (s1.size() != s2.size()) return false;
        if (s1 == s2) return true;
        string str1 = s1, str2 = s2;
        sort(str1.begin(), str1.end());
        sort(str2.begin(), str2.end());
        if (str1 != str2) return false;
        for (int i = 1; i < s1.size(); ++i) {
            string s11 = s1.substr(0, i);
            string s12 = s1.substr(i);
            string s21 = s2.substr(0, i);
            string s22 = s2.substr(i);
            if (isScramble(s11, s21) && isScramble(s12, s22)) return true;
            s21 = s2.substr(s1.size() - i);
            s22 = s2.substr(0, s1.size() - i);
            if (isScramble(s11, s21) && isScramble(s12, s22)) return true;
        }
        return false;
    }

//DP 三维动态规划 

};
