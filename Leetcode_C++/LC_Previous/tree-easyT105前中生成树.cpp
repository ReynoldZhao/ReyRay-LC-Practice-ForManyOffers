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


 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };

class Solution {
public:
    TreeNode* node(vector<int>& vp,int p,vector<int>& vi,int i,int n){
		if(n==0) return NULL;
		if(n==1){
			TreeNode *root = new TreeNode(vp[0]);
			root->right = NULL;
			root->left = NULL;
		}
		TreeNode *root = new TreeNode(vp[p]);
		int value = root->val;
		for(int j=0;j<vi.size();j++){
			if(vi[i+j]==value) break;
		}
		int L = j;
		int R = n-i-1-j;
		root->left = node(vp,p+1,vi,i,L)
		roo->right = node(vp,p+L+1,vi,i+L+1,R)
    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
		int n = preorder.size();
		TreeNode* root;
		root = node(preorder,0,inorder,0,n);
		return root; 
    }
};
