    TreeNode* KthNode(TreeNode* pRoot, int k)
    {
        if(!pRoot) return NULL;
        stack<TreeNode*> s;
        int index = 0;
        TreeNode* temp = pRoot
        while(!s.empty()&&temp!=NULL){
            if(temp!=NULL){
                s.push(temp);
                temp = temp->left;
            }
            else{
                temp = s.pop();
                index++;
                if(index==k) return temp;
                temp = temp->right;
                
            }
        }
    }
//无法理解中序遍历的非递归写法