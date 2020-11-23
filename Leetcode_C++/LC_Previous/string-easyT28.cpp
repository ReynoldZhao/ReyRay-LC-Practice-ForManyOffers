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
    int strStr(string haystack, string needle) {
        if(needle.empty()) return 0;
        int length = needle.size();
        for(int i=0;i<haystack.size();i++){
        	if(haystack[i]==needle[0]){
        		string str = haystack.substr(i,length);
        		if(str==needle) return i;
			}
		}
		return -1;
    }
};
