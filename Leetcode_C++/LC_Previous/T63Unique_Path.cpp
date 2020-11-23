#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<utility>
#include<cstring>
using namespace std;


 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 
class Solution {
public:
    bool isSym(TreeNode* p,TreeNode* q){
    	if(p->left==q->right&&p->right==q->left){
    		if(p->left!=NULL) return isSym(p->left,q->right);
    		if(p->right!=NULL) return isSym(p->right,q->left);
    		else return true;
		}
		else return false;
    
    }
    bool isSym(TreeNode* p,TreeNode *q){
    	queue<TreeNode*> pp;
    	queue<TreeNode*> qp;
    	pp.push(p);
    	qp.push(q);
    	
    	while(!pp.empty()&&!qp.empty()){
    		TreeNode* tempp = pp.front();
    		TreeNode* temqq = qp.front();
    		pp.pop();
    		qp.pop();
    		
    		if(tempp==NULL&&temqp==NULL) return true;
    		if(tempp==NULL||temqp==NULL) return false;
    		if(tempp->val!=temqp->val) return false;
    		if(tempp->val==temqp->val)
    		{
    			pp.push(tempp->left);
    			qp.push(temqp->right);
    			
    			pp.push(tempp->right);
    			qp.push(temqp->left);
			}
    		
		}

	}
    bool isSymmetric(TreeNode* root) {
        if(root->left==NULL&&root->right==NULL) return true;
        else if(root->left==root->right) return isSym(root->left,root->right);
        else return false;
    }
};
