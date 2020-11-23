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
	vector<int> temp;
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        vector<vector<int> > res;
        DFS(root,sum,res);
        return res;
    }
    void DFS(TreeNode* root, int sum,vector<vector<int> > res){
        if(!root) return;
        temp.push(root->val);
    	if(root->val==sum&&!root->right&&!root->left) {
			res.push_back(temp);
		}
		DFS(root->left, sum-root->val,res);
		DFS(root->right, sum-root->val,left);
		temp.pop_back();
	}
};

