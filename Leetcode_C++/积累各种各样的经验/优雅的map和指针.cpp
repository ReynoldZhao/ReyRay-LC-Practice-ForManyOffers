/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
*/

class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(!head) return NULL;
        unordered_map<Node*, Node*> m1;
        Node* temp = head;
        Node* res = new Node(0, nullptr, nullptr);
        Node* cur = res;
        while(temp!=NULL){
            Node* t = new Node(temp->val, nullptr, nullptr);
            m1.emplace(temp, t);
            cur->next = t;
            cur = cur->next;
            temp = temp->next;
        }
        temp = head;
        cur = res->next;
        while(temp!=NULL){
            cur->random = m1[temp->random];
            temp = temp->next;
            cur = cur->next;
        }
        return res->next;      
    }
};