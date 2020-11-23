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
https://www.cnblogs.com/grandyang/p/4464476.html
class Solution {
public:
    string longestPalindrome(string s) {
        if (s.size() < 2) return s;
        int n = s.size(), maxLen = 0, start = 0;
        for (int i = 0; i < n - 1; ++i) {
            searchPalindrome(s, i, i, start, maxLen);
            searchPalindrome(s, i, i + 1, start, maxLen);
        }
        return s.substr(start, maxLen);
    }
    void searchPalindrome(string s, int left, int right, int& start, int& maxLen) {
        while (left >= 0 && right < s.size() && s[left] == s[right]) {
            --left; ++right;
        }
        if (maxLen < right - left - 1) {
            start = left + 1;
            maxLen = right - left - 1;
        }
    }
};


class Solution {
public:
    string longestPalindrome(string s) {
    	if(s.size()<2) return s;
    	int n = s.size();
		int left,right;
		int maxlen=0;
		int start=0;
		for(int i=0;i<n;){
			if(n-i<=maxlen/2) break;
			left = right = i;
			while(right<n-1&&s[right+1] = s[right]) ++right;
			i = right+1;
			while(right<n-1&&left>0&&s[right+1]==s[left-1]){
				left--;
				right++;
			}
\			if(right-left+1>maxlen){
				maxlen = right - left+1;
				start = left;
			} 
		}
		return s.substr(start,maxlen);
    }
};

class Solution {
public:
    string longestPalindrome(string s) {
		if(s.empty()) return s;
		int dp[s.size()][s.size()] = {0};
		int left=0,right=0,length = 0;
		for(int i=0;i<s.size();i++){
			for(int j=0;j<i;j++){
				dp[j][i] = (s[i]==s[j])&&(i-j<2||dp[j+1][i-1]);
				if(dp[j][i]&&len<i-j+1){
					len = i-j+1;
					right = i;
					left = j;
				}
			}
			dp[i][i] = 1;
		}
		return s.substr(left,len);
    }
};

