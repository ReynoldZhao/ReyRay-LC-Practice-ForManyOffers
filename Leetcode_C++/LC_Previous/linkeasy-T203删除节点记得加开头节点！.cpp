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
    ListNode* removeElements(ListNode* head, int val) {
		if(!head) return NULL;
		while(head->val==val) head = head->next;
		ListNode *cur = head;
		while(cur->next){
			if(cur->next->val==val){
				cur->next = cur->next->next;
			}
			else cur = cur->next;
		}
		return head;  
    }
};
