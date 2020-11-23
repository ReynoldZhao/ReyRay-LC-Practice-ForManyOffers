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

  struct TreeNode {
      int val;
      TreeNode *left;
      TreeNode *right;
      TreeNode(int x) : val(x), left(NULL), right(NULL) {}
  };
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
       if(nums.empty()) return NULL;
       int left = 0,right = nums.size()-1;
       TreeNode* root;
       root = insert(0,right,nums);
       return root;
    }
    TreeNode* insert(int left,int right,vector<int>& nums){
    	if(left>=right){
    		TreeNode* t = new TreeNode(nums[left]);
    		return t;
		}
		int mid = (left+right)/2;
    	TreeNode* t = new TreeNode(nums[mid]);
		t->left = insert(left,mid-1,nums);
		t->right = insert(mid+1,right,nums);
		return t;
	}
};
