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
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
      	if(n==1) return {0};
      	vector<int> res;
      	vector<unordered_map<int>> adj;
      	for(auto edge:edges){
      		adj[edge[0]].insert(edge[1]);
      		adj[edge[1]].insert(edge[0]);
		  }
		  queue<int> q;
		  for(int i=0;i<adj.size();i++){
		  	if(adj[i].size()<=1) q.push(i);
		  }
		  while(n>2){
		  	int size = q.size();
		  	n-=size;
		  	for(int i=0;i<q.size();i++){
		  		int t = q.front();
		  		q.pop();
		  		for(auto a:adj[t]){
		  			adj[a].erase(t);
		  			if(adj[a].size()<=1) q.push(a);
				  }
			  }
		  }
		  while(!q.empty()){
		  	res.push_back(q.front());
		  	q.pop();
		  }
		  return res;
    }
};
