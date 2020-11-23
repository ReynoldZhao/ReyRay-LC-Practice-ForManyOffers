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

https://blog.csdn.net/fuxuemingzhu/article/details/82022530

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
		if(s.size()<=0) return 0;
		vector<bool> m(256,false);
		int l=0,r=0;
		int length = 0,ml=0;
		for(int i=0;i<s.size();i++){
			while(m[s[i]]!=false){
				m[s[l]] = false;
				l++;
			}
			r = i;
			m[s[r]] = true; 
			length = r-l+1;
			ml  = max(ml,length);
		}
		return ml;
    }
};

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        vector<int> m(256,-1);
        int maxlen = 0,start = -1;
        for(int i=0;i<s.size();i++){
        	if(m[s[i]]>start)
        		start = m[s[i]];
        	m[s[i]] = i;
        	maxlen = max(maxlen,i-start);
		}
		return maxlen;
    }
};
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
		int N = s.size();
		if(N<=0) return 0;
		unordered_set<char> set;
		int r,l;
		int length,ml=0;
		while(r<N){
			while(set.count(s[r])){
				set.erase(s[l]);
				l++;
			}
			set.insert(s[r]);
			res = max(res,int(set.size()));
			++r; 
		} 
    }
};
--------------------- 
作者：负雪明烛 
来源：CSDN 
原文：https://blog.csdn.net/fuxuemingzhu/article/details/82022530 
版权声明：本文为博主原创文章，转载请附上博文链接！
