struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
 
class Solution {
public:
    TreeNode* node(vector<int>& vi,int i,vector<int>& vpo,int po,int n){
    	if(n==0) return NULL;
    	if(n==1)
        	{   
           TreeNode *root = new TreeNode(vpo[po+n-1]);
				root->left = NULL;
				root->right = NULL;
				return root;
			}
        TreeNode *root = new TreeNode(vpo[po+n-1]);
			int j;
			for(j=0;j<n;j++){
				if(vi[i+j]==root->val)
					break;
			}
			int L,R;
			L = j;
			R = n-j-1;
			root->left = node(vi,i,vpo,po,L);
			root->right = node(vi,i+L+1,vpo,po+L,R);
			return root;
    }
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        int n = inorder.size();
		TreeNode* root;
		root = node(inorder,0,postorder,0,n);
		 return root;
    }
};

