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



  struct ListNode {
      int val;
      ListNode *next;
      ListNode(int x) : val(x), next(NULL) {}
  };

class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
    	if(head==NULL) return NULL;
        set<int> s;
		ListNode* l;
        l = head;
		s.insert(l->val);
		while(l->next!=NULL){
			if(s.find(l->next->val)!=s.end()){
				l->next = l->next->next;
			}
			else{
				s.insert(l->next->val);
				l = l->next;
			} 
		} 
    }
};

class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
    	ListNode* cur = head;
    	while(cur&&cur->next){
    		if(cur->val&&cur->next->val){
    			cur->next = cur->next->next;
			}
			else cur = cur->next;
		}
		return head;
    }
};

class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
    	ListNode* cur = head;
    	while(cur&&cur->next){
    		if(cur->val&&cur->next->val){
    			cur->next = cur->next->next;
			}
			else cur = cur->next;
		}
		return head;
    }
};

class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
		if(!head||!head->next) return head;
		head->next = deleteDuplicates(head->next);
		return (head->val==head->next->val)?head->next:head;
    }
};
