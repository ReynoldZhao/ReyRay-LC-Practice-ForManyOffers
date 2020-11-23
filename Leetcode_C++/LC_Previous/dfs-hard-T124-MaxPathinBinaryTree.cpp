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
    int maxPathSum(TreeNode* root) {
    	int res = INT_MIN; 
    	maxpath(root,res);
		return res; 
    }
    int maxpath(TreeNode* root, int &value){
		if(!root) return 0;
		int left = max(maxpath(root->left,0))
		int right = max(maxpath(root->right,0))
		res = max(res,left+right+root->val);
		return max(right,left)+root->val;
		 
	}
};
