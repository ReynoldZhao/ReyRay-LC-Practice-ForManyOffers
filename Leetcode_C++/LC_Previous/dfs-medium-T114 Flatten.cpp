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
    void flatten(TreeNode* root) {
        if(!root) return;
        if(root->left) flatten(root->left);
        if(root->right) flatten(root->right);
        TreeNode* temp = root->right;
        root->right = root->left;
        root->left = NULL;
        while(root->right) root = root->right;
        root->right = temp;
        
    }
};

class Solution {
public:
    void flatten(TreeNode* root) {
		TreeNode* cur = root;
		while(cur){
			if(cur->right){
				TreeNode* p = cur->left;
				while(p->right) p = p->right;
				p->right = cur->right;
				cur->right = cur->left;
				cur->left = NULL; 
			}
			cur = cur->right;
		}   
    }
};

