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
#include<unordered_set>
using namespace std;

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> res;
		string temp = "";
		if(!root) return res;
		helper(root,temp,res);
		return res; 
    }
    void helper(TreeNode* root, string &temp, vector<string> &res){
    	string t = to_string(root->val);
		temp+=t;
    	if(!root->right&&!root->left){
    		res.push_back(temp);
    		return;
		}
		temp+="->";
		if(root->right) helper(root->right,temp,res);
		if(root->left) helper(root->left,temp,res);
		return;	
	}
};
