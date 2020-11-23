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
    void recoverTree(TreeNode* root) {
		vector<int> v;
		vector<TreeNode*> list;
		inorder(root,list,v);
		sort(v.begin(),v.end());
		for(int i=0;i<v.size();i++){
			list[i]->val = v[i];
		}
    }
    void inorder(TreeNode* root, vector<TreeNode*> list, vector<int> v){
    	if(!root) return;
    	inorder(root->left,list,v);
    	v.push_back(root->val);
    	list.push_back(root);
    	inorder(root->right,list,v);
	}
};
