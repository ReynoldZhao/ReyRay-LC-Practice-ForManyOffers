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
    string longestCommonPrefix(vector<string>& strs) {
        string res = "";
        if(strs.empty()) return res;
		for(int i=0;i<strs[0].size();i++){
			char c= strs[0][i];
			for(int j=1;j<strs.size();j++){
				if(c!=strs[j][i]) return res;
			}
			res+=c;
		}
		return res; 
    }
};
