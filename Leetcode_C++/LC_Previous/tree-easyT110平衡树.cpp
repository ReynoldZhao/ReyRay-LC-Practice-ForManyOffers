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

class Solution {
public:    
    bool isBalanced(TreeNode *root) {
		if(checkDepth(root)==-1) return false;
		else return true;
    }
    int checkDepth(TreeNode *root) {
		if(!root) return 0;
		int left = checkDepth(root->left);
		if(leff==-1) return -1;
		int right = checkDepth(root->right);
		if(right==-1) return -1;
		int dif = abs(left-right);
		if(dif>1) return -1; 
		else return 1+max(left,right);
    }
};
