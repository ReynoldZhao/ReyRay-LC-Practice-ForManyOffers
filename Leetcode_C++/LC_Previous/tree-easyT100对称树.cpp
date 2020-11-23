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
    bool isSym(TreeNode* p,TreeNode* q){
		if(p==NULL&&q->NULL) return true;
		if(p==NULL||q->NULL) return false;
		return (p->val==q->val)&&isSym(p->left,q->right)&&isSym(p->right,q->left);
		
    }
    bool isSymmetric(TreeNode* root) {
		if(root==NULL) return true;
		if(root->left==NULL&&root->right==NULL) return true;
		if(root->left==NULL||root->right==NULL) return false;
		return isSym(root->left,root->right);
    }
};
