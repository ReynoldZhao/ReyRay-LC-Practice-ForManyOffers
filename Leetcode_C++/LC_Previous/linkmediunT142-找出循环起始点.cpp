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
    ListNode *detectCycle(ListNode *head) {
       ListNode *temp;
       ListNode *res; 
       if(!hasCycle(head)) return NULL;
	   while(head&&head->next){
	   		temp = head->next;
			res = head;
			head->next = NULL;
			head = temp;
			if(!isCycle(head)) return res; 	
	   } 
    }
    bool hasCycle(ListNode *head) {
        if(head==NULL) return false;
        ListNode *fast, *slow;
        fast = head, slow = head;
		while(fast->next&&fast->next->next){ 
                fast = fast->next->next;
                slow = slow->next;
			if(fast == slow) return true;
		}
		return false; 
    }
};
