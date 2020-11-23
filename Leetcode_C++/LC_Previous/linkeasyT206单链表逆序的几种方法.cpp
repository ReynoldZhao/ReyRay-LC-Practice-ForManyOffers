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
    ListNode* reverseList(ListNode* head) {
        if(!head) return NULL;
        ListNode *dummy = NULL;
        while(head){
			ListNode *t = head->next;
			head->next = dummy;
			dummy = head;
			head = t;
		}
		return dummy;
    }
    ListNode* reverseList(ListNode* head){
    	ListNode *newNode = new ListNode(0);
    	ListNode *temp;
    	newNode->next = head;
    	ListNode *cur = head;
		while(cur&cur->next){
			temp = newNode->next;
			newNode->next = cur->next;
			cur->next = cur->next->next;
			newNode->next->next = temp;
		} 
		return newNode;
	}
	ListNode* reverseList(ListNode* head){
		ListNode *temp = NULL;
		ListNode *cur = NULL;
		if(!head) return NULL;
		cur = head->next;
		while(cur){
			temp = cur->next;
			cur->next = temp->next;
			temp->next = head->next;
			head->next = temp; 
		}
	} 
};
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode *newHead = reverseList(head->next);
        head->next->next = head;
        head->next = NULL;
        return newHead;
    }
};
