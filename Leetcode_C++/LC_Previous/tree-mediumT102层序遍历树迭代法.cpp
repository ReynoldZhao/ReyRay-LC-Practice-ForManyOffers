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
    vector<vector<int> > levelOrderBottom(TreeNode* root) {
		vector<vector<int> > res;
		if(root==NULL) return res;
		queue<TreeNode*> q;
		q.push(root);
		TreeNode *t;
		while(!q.empty()){
			int length = q.size();
			for(int i=0;i<length;i++){
				vector<int> vt;
				t = q.front();
				vt.push_back(t->val);
				if(t->left!=NULL) q.push(t->left);
				if(t->right!=NULL) q.push(t->right);
				q.pop(); 				
			}
		}
    }
};

class Solution {
public:
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
        vector<vector<int>> res;
        levelorder(root, 0, res);
        return vector<vector<int>> (res.rbegin(), res.rend());
    }
    void levelorder(TreeNode* node, int level, vector<vector<int>>& res) {
        if (!node) return;
        if (res.size() == level) res.push_back({});
        res[level].push_back(node->val);
        if (node->left) levelorder(node->left, level + 1, res);
        if (node->right) levelorder(node->right, level + 1, res);
    }
};

