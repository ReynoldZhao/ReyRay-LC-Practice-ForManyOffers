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
    int maxDepth(TreeNode* root) {
        int depth=0;
        if(root==NULL) return depth;
        vector<vector<int> > res;
        levelorder(root,0,res);
        return res.size();
    }
    void levelorder(TreeNode *root,int level,vector<vector<int> >& res){
    	if(!root) return;
    	if(level == res.size()){
    		res.push_back({});
		}
		res[level].push_back(root->val);
		if(root->left) levelorder(root->left,level+1,res);
		if(root->right) levelorder(root->right,level+1,res);
	}
};

class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;
        return 1 + max(maxDepth(root->left), maxDepth(root->right));
    }
};

class Solution {
public:
    int maxDepth(TreeNode* root) {
		if(!root) return 0;
		return 1+max(maxDepth(root->left),maxDepth(root->right));
    }
    int minDepth(TreeNode* root){
    	if(!root) return 0;
    	return 1+min(minDepth(root->left),minDepth(root->right));
	}
};

class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;
        int res = 0;
        queue<TreeNode*> q{{root}};
        while (!q.empty()) {
            ++res;
            for (int i = q.size(); i > 0; --i) {
                TreeNode *t = q.front(); q.pop();
                if (t->left) q.push(t->left);
                if (t->right) q.push(t->right);
            }
        }
        return res;
    }
};
