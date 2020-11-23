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

class Node {
public:
    int val;
    vector<Node*> neighbors;

    Node() {}

    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};

class Solution {
public:
    Node* cloneGraph(Node* node) {
        unordered_map<Node*,Node*> m;
        queue<Node*> q;
        q.push(node);
        Node* clone = new Node(node->val);
        while(!q.empty()){
        	Node* temp = q.front();
        	q.pop();
        	for(Node* neighbor:temp->neighbors){
        		if(!m.count(neighbor)){
        			m[neighbor] = new Node(neighbor->val);
        			q.push(neighbor);
				}
				m[temp]->neighbors.push_back(neighbor);
			}
		}
		return clone;
    }
	Node* cloneGraph(Node* node){
		unordered_map<Node*,Node*> m;
		return helper(node,m);
	}
    Node* helper(Node* node,unordered_map<Node*, Node*>& m){
    	if(!node) return NULL;
    	if(m.count(node)) return m[node];
    	Node* clone = new Node(clone->val);
    	m[node] = clone;
    	for(Node* neighbor:node->neighbors){
    		clone->neighbors.push_back(helper(neighbor,m)); 
		}
		return clone;	
	}
};
