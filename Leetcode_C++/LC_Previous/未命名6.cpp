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

