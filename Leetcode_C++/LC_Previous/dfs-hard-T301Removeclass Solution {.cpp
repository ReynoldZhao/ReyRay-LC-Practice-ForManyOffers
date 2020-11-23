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
#include<unordered_set>
using namespace std;

class Solution {
public:
    vector<string> removeInvalidParentheses(string s) {
        vector<string> res;
        queue<string> q;
        q.push(s);
        unordered_set<string> visited{{s}};
        bool found = false;
        while(!q.empty()){
        	string t = q.front();
        	q.pop();
        	if(isValid(t)) 
			{
				res.push_back(t);
				found = true;
			}
			if(found) continue;
			for(int i=0;i<t.size();i++){
				if(t[i]!='('&&t[i]!=')') continue;
				string tm = t.substr(0,i)+t.substr(i+1);
				if(!visited.count(tm)){
					visited.insert(tm);
					q.push(tm);
				}
			}			
		}
		return res;
    }
    
    vector<string> removeInvalidParentheses(string s) {
    	int cnt1=0, cnt2=0;
    	for(char c:s){
    		cnt1 += (c=='(');
    		if(cnt1==0) cnt2+=(c==')');
    		else cnt1-=(c==')');
		}
	}
    
    bool isValid(string s){
    	int cnt=0;
    	for(int i=0;i<s.size();i++){
    		if(s[i]=='(') cnt++;
    		if(s[i]==')') cnt--;
    		if(cnt<0) return false;
		}
		if(cnt==0) return true;
		else return false;
	}
};
