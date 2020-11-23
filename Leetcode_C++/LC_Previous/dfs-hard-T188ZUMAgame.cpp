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
    int findMinStep(string board, string hand) {
        int res = INT_MAX;
		unordered_map<char,int> m;
		for(char c:hand) m[c]++;
		res = helper(board,m);
		return res ==INT_MAX? -1:res; 
    }
    int helper(string board,unordered_map<char,int> m){
    	board = remove(board);
    	if(board.empty()) return 0;
    	int cnt = INT_MAX, j=0;
		for(int i=0;i<=board.size();i++){
			if(i<board.size()&&board[i]==board[j]) continue;
			int need =3- (i-j);
			if(m[board[j]]>=need){
				m[board[j]]-=need;
				int t = helper(board.substr(0,j)+board.substr(i),m);
				if(t!=INT_MAX) cnt = min(cnt,t+need);
				m[board[j]]+=need;
			}
			j = i;
		}
		return cnt;    	
	}
	int findMinStep(string board, string hand){
		int res = INT_MAX; 
		unordered_map<char> m;
		for(int i=0;i<hand.size();i++){
			if(m.count(hand[i])) continue;
			for(int j=0;j<board.size();j++){
				if(hand[i]!=board[j]) continue;
				string newboard = board.insert(j,1,hand[i]);
				newboard = remove(newboard);
				if(newboard.size()==0) return 1;
				string newhand = hand.erase(i);
				int cnt = findMinStep(newboard,newhand)
				if(cnt!=-1) cnt = min(res,cnt+1); 
			}
		}
		return res==INT_MA? -1:res
	}
	string remove(string board){
		for(int i=0, j=0;i<=board.size();i++){
			if(i<board.size() && board[i]==board[j]) continue;
			if(i-j>=3) return remove(board.substr(0,j)+board.substr(i));
			else j=i;
		}
		return board;
	} 
    
};
