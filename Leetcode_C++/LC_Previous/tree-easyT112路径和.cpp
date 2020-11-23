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



  struct TreeNode {
      int val;
      TreeNode *left;
      TreeNode *right;
      TreeNode(int x) : val(x), left(NULL), right(NULL) {}
  };
class Solution {
public:
	set<int> sumset; 
    bool hasPathSum(TreeNode* root, int sum) {
       if(!root) return false;
       int s = 0;
	    Sum(root,sum);
	    if(sumset.find(sum)!=sumset.end()) return true;
	    else return false;
    }
    void Sum(TreeNode* t,int s){
    	s = s + t->val;
    	if(t->left==NULL&&t->right==NULL){
    		sumset.insert(s);
    		return;
		}
		if(t->left) Sum(t->left,s);
		if(t->right) Sum(t->right,s);
	}
};
