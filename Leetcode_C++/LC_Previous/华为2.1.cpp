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
bool isAlpha(char c){
	if(c>='A'&&c<='Z') return true;
	else return false;
}
bool isalpha(char c){
	if(c>='a'&&c<='z') return true;
	else return false;
}
bool Find(string s,int tar){
	for(unsigned long int i=0;i<s.size();i++){
		if(s[i]==tar){
			return true;
		}
	}
	return false;	
}
bool find(string &s,int tar){
	for(unsigned long int i=0;i<s.size();i++){
		if(s[i]==tar){
			s[i] = '#';
			return true;
		}
	}
	return false;
}
bool cmp(string s1, string s2){
	if(s1[0]<s2[0]) return true;
	else if(s1[0]==s2[0]){
		if(s1.length()>s2.length()) return true;
		else return false;
	}
	else return false;
}
string substr(string &ns,int current){
	for(unsigned long int i=0;i<ns.size();i++){ 
			string res = "";
			int target = current+32;
			if(find(ns,target)&&find(ns,current)){
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
			if(res.empty()){
					ns[i]='#';
					return "";
			}
			else{
				return res;
			}			
		}
		return "";
}

int main(){
	string s;
	cin>>s;
	vector<string> result;
	string ns="";
	bool exist[26];
	memset(exist,sizeof(exist)*26,0);
	for(unsigned long int i=0;i<s.size();i++){
		if(isAlpha(s[i])||isalpha(s[i])){
			ns+=s[i];
		}
	}
	for(int i=0;i<26;i++){
		int t = 'A'+i;
		int current = t;
		int target = current+32;
		while(Find(ns,current)&&Find(ns,target)){
			string res = substr(ns,current);
			if(!res.empty()) result.push_back(res);
		} 
	}
	if(result.empty()){
		cout<<"Not Found"<<endl;
		return 0;
	}
	sort(result.begin(),result.end(),cmp);
	//Êä³ö 
	for(unsigned long int i=0;i<result.size();i++){
		cout<<result[i]<<endl;
	}
	return 0;	
}
