
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 
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
	vector<int> v1;
	vector<int> v2;
	void PreOrderRetrieval(TreeNode* p,vector<int> &v){
		if(p==NULL){
			v.push_back(INT_MAX);
			return;
		}
		else v.push_back(p->val);
		PreOrderRetrieval(p->left,v);
		PreOrderRetrieval(p->right,v);
	}
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if(p==NULL&&q==NULL) return true;
        if(p==NULL||q==NULL) return false;
        PreOrderRetrieval(p,v1);
        PreOrderRetrieval(q,v2);
        if(v1.size()!=v2.size()) return false;
        for(int i=0;i<v1.size();i++){
        	if(v1[i]!=v2[i]) return false;
		}
		return true;
    }
};
