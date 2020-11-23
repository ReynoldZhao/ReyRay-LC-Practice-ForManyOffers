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
    TreeNode* node(vector<int>& vp,int p,vector<int>& vi,int i,int n){
    	if(n==0) return NULL;
    	if(n==1)
        	{   
           TreeNode *root = new TreeNode(vp[p]);
				root->left = NULL;
				root->right = NULL;
				return root;
			}
        TreeNode *root = new TreeNode(vp[p]);
			int j;
			for(j=0;j<n;j++){
				if(vi[i+j]==root->val)
					break;
			}
			int L,R;
			L = j;
			R = n-j-1;
			root->left = node(vp,p+1,vi,i,L);
			root->right = node(vp,p+L+1,vi,i+L+1,R);
			return root;
    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n = preorder.size();
		TreeNode* root;
		root = node(preorder,0,inorder,0,n);
		 return root;
    }
};
