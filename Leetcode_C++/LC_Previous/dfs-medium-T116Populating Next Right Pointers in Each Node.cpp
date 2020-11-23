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

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}

class Solution {
public:
    Node* connect(Node* root) {
        if(!root) return NULL;
        if(root->left) root->left-next = root->right;
		if(root->right) root->right->next = root->next? root->next->left:NULL;
		connect(root->left);
		connect(root->right);
    }
    Node* connect(Node* root){
    	queue<Node*> q;
    	q.push(root);
    	while(!q.empty()){
    		int size = q.size();
    		Node* temp = q.front();
    		q.pop();
    		for(int i=0;i<size;i++){
    			Node* temp = q.front();
    			q.pop();
    			if(i<size-1) temp->next = q.front();
				if(temp->left) q.push(temp->left);
				if(temp->right) q.push(temp->right); 
			}
		}
		return root;
	}
	Node* connect(Node* root){
		Node* start = root, *cur = NULL;
		while(start->left){
			cur = start;
			while(cur){
				if(cur->left) cur->left->next = cur->right;
				if(cur->right) cur->right->next = cur->next? cur->next:cur->next->left;
				cur = cur->next; 
			}
			start = start->left;
		}
		return root;
	}
};
