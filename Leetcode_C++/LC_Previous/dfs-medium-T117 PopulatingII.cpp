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
		queue<Node*> q;
		q.push(root);
		while(!q.empty()){
			int size = q.size();
			for(int i=0;i<size;i++){
				Node* temp = q.front();
				q.pop();
				if(i<size-1) temp->next = q.front();
				else temp->next = NULL;
				if(temp->left) q.push(temp->left);
				if(temp->right) q.push(temp->right);
			}
		}
		return root;
    }
};
