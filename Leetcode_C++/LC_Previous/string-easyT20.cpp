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
    bool isValid(string s) {
    	if(s.empty()) return true;
        stack<char> st;
        char temp;
        for(int i=0;i<s.size();i++){
        	if(s[i]=='{'||s[i]=='['||s[i]=='(') st.push(s[i]);
        	else{
        		switch(s[i]){
        			case '}':
        				if(st.top()!='{') return false;
        				else st.pop();
        				break;
        			case ']':
        				if(st.top()!='[') return false;
        				else st.pop();
        				break;
        			case ')':
        				if(st.top()!='(') return false;
        				else st.pop();
        				break; 
					default:
						break;
                }
			}
		}
		if(st.empty()) return true;
		else return false;
    }
};
class Solution {
public:
    bool isValid(string s) {
        stack<char> parentheses;
        for (int i = 0; i < s.size(); ++i) {
            if (s[i] == '(' || s[i] == '[' || s[i] == '{') parentheses.push(s[i]);
            else {
                if (parentheses.empty()) return false;
                if (s[i] == ')' && parentheses.top() != '(') return false;
                if (s[i] == ']' && parentheses.top() != '[') return false;
                if (s[i] == '}' && parentheses.top() != '{') return false;
                parentheses.pop();
            }
        }
        return parentheses.empty();
    }
};
