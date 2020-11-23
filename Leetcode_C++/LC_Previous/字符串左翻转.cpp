#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<utility>
#include<math.h>
using namespace std;

class Solution {
public:
	void reverse(string &s,int st,int e){
		if(st==e) return s;
		char temp;
		while(s<e){
			temp = s[st];
			s[st] = s[e];
			s[e] = temp;
			s++;
			e--;
		}
	}
    string LeftRotateString(string str, int n) {
        int len = str.size();
        if(len==0||n==0) return str;
        reverse(str.begin(),str.end());
        reverse(str.begin(),len-n-1);
        reverse(str.begin()+len-n,str.end())
        return str
    }
};
