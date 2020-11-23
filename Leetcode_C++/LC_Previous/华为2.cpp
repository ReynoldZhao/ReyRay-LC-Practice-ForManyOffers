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
#define pair<int,int> p; 
using namespace std;
bool isAlpha(char c){
	if(c>='A'&&c<='Z') return true;
	else return false;
}
bool isalpha(char c){
	if(c>='a'&&c<='z') return true;
	else return false;
}
bool find(string &s,int tar){
	for(int i=0;i<s.size();i++){
		if(s[i]==tar){
			s[i] = '#';
			return true;
		}
	}
	return false;
}
bool cmp(string s1, string s2){
	if(s1[0]>s2[0]) return true;
	else if(s1[0]==s2[0]){
		if(s1.length()>s2.length()) return true;
		else return false;
	}
	else return false;
}

int main(){
	string s;
	cin>>s;
	vector<string> result;
	string ns="";
	for(int i=0;i<s.size();i++){
		if(isAlpha(s[i])||isalpha(s[i])){
			ns+=s[i];
		}
	}
	for(int i=0;i<ns.size();i++){
		if(isalpha(ns[i])){
			string res = "";
			int current = ns[i];
			int target = ns[i]-32;
//			bool flag = false;
			if(find(ns,target)){
				ns[i] = '#';
//				flag = true
				res+=target;
				res+=current;
				while(current!='z'){
						current = current+1;
						target = target+1;
						if(find(ns,current)&&find(ns,target)){
							res+=target;
							res+=current;
						}
						else break;
				}
			}
//			if(flag==false)
			if(res.empty()){
					ns[i]='#';
					continue;
			}
			else{
				result.push_back(res);
			}
		}
		else if(isAlpha(ns[i])){
			string res = "";
			int current = ns[i];
			int target = ns[i]+32;
//			bool flag = false;
			if(find(ns,target)){
				ns[i] = '#';
//				flag = true
				res+=current;
				res+=target;
				while(current!='z'){
						current = current+1;
						target = target+1;
						if(find(ns,current)&&find(ns,target)){
							res+=current;
							res+=target;
						}
						else break;
				}
			}
//			if(flag==false)
			if(res.empty()){
					ns[i]='#';
					continue;
			}
			else{
				result.push_back(res);
			}			
		}
	}
	if(result.empty()){
		cout<<"Not Found"<<endl;
		return 0;
	}
	sort(result.begin(),result.end(),cmp);
	for(int i=0;i<result.size();i++){
		cout<<result[i]<<endl;
	}
	return 0;	
}
