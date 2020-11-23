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
    int lengthOfLastWord(string s) {
    	int left=0,right=s.size()-1;
    	while(s[left]==' ') left++;
    	while(s[right]==' ') right--;
        int i = 0;
        for(;i<right-left+1;i++){
        	if(s[right-i]==' ') break;
		}
		return i;
    }
};
