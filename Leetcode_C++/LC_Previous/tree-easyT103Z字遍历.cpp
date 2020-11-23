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
    vector<vector<int> > zigzagLevelOrder(TreeNode* root) {
        vector<vector<int> > result;
        if (root == NULL) return result;
        stack<TreeNode*> s1;
        stack<TreeNode*> s2;
		s1.push(root); 
        int dep = 0;
        while (!s1.empty()||!s2.empty()) {
            vector<int> out;
            TreeNode* temp;
            while(!s1.empty()){
            	temp = s1.top();
            	out.push_back(temp->val);
            	if(temp->left) s2.push(temp->left);
				if(temp->right) s2.push(temp->right);
				s1.pop(); 
			}
			if(!out.empty()) result.push_back(out);
			out.clear();
            while(!s2.empty()){
            	temp = s2.top();
            	out.push_back(temp->val);
            	if(temp->right) s1.push(temp->right);
				if(temp->left) s1.push(temp->left);
				s2.pop(); 
			}
			if(!out.empty()) result.push_back(out);
			out.clear();
        }
        return result;      
    }
};
