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
using namespace std;


  struct TreeNode {
      int val;
      TreeNode *left;
      TreeNode *right;
      TreeNode(int x) : val(x), left(NULL), right(NULL) {}
  };
 
class Solution {
public:
	void BFS(int i,int j,vector<vector<char> >& board,bool &flag){
        int m = board.size();
		int n = board[0].size();			
		if(i<0||j<0||i>=m||j>=n){
			return;
		}
		if((i==0||i==m-1||j==0||j==n-1)&&board[i][j]=='O'){
			flag = false;
			return;
		}
		BFS(i+1,j,board,flag);	
		BFS(i-1,j,board,flag);
		BFS(i,j+1,board,flag);
		BFS(i,j-1,board,flag);
	}
	void BFS2(int i,int j,vector<vector<char> >& board){
        int m = board.size();
		int n = board[0].size();			
		if(i<0||j<0||i>=m||j>=n){
			return;
		}
		if(board[i][j]=='O') board[i][j]='X';
		BFS2(i+1,j,board);	
		BFS2(i-1,j,board);
		BFS2(i,j+1,board);
		BFS2(i,j-1,board);		
	}
    void solve(vector<vector<char> >& board) {
        int m = board.size();
		int n = board[0].size();
		bool flag = false;
		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++){
				if(board[i][j]=='X') {
					continue;
				}
				if(board[i][j]=='O'){
					BFS(i,j,board,flag);
					if(flag==true){
						BFS2(i,j,board);
					}
				}
			}
		}
    }
};
