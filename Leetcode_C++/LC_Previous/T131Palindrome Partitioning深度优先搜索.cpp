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
    vector<vector<string>> partition(string s) {
    	vector< vector<string> > res;
    	vector<string> out;
    	partitionDFS(s,0,out,res);
    	return res;
	}
	void PartitionDFS(string s,int start,vector<string> &out,vector< vector<string> > &res){
		if(s.size()==start){
			res.push_back(out);
			return;
		} 
		for(int i=start;i<s.size();i++){
			if(isPalindrome(s,start,i)){
				out.push_back(s.substr(strat,i-start+1));
				PartitionDFS(s,i+1,out,res);
				out.pop_back();
			}
		}
	}
 	bool isPalindrome(string s,int start,int end){
 		while(start<end){
 			if(s[start]!=s[end]) return false;
 			right--;
 			left++;
		 }
		 return true;
	 }
};
