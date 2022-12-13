#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<stack>
#include<list>
#include<utility>
#include<cstring>
#include<string>
#include<unordered_map>
#include<unordered_set>
#include<ext/hash_map>
#include<deque>
using namespace std;

class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = s.size();
        unordered_set<string> set(wordDict.begin(), wordDict.end());
        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        for (int i = 0; i < n; i++) {
            if (dp[i] == 0) continue;
            for (int j = i + 1; j <= n; j++) {
                if (j - i < n && set.count(s.substr(i, j - i))) dp[j] = 1;
            }
            if (dp[n] == 1) return true;
        }
        return dp[n] == 1;
    }
};