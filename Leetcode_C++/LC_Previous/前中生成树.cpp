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
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    	if(preorder.size()==1) {
    		TreeNode *root;
    		root->val = preorder.begin();
    		return root;
		}
        TreeNode *root;
		root->val = preorder.begin(); 
		vector<int> vp;
		vector<int> vi;
		int v = root->val;
		vector<int>::iterator i = find(inorder.begin(),inorder.end(),v);
		pos = i-preorder.begin(); 
		root->left = buildTree(vp,vi)
    }
};
