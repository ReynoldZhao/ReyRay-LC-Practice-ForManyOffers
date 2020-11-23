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
class Solution{
	public:
		void reorderList(ListNode *head) {
			if(!head||!head->next||!head->next->next) return false;
			ListNode *fast = head, *slow = head;
			while(fast->next||fast->next->next){
				slow = slow->next;
				fast = fast->next->next;
			}
			ListNode *mid = slow->next;
			ListNode *last = mid,*pre = NULL;
			while(last){
				ListNode *next = last->next;
				last->next = pre;
				pre = last;
				last = next;
			}
			while(head&&pre){
				ListNode *next = head->next;
				head->next = pre;
				pre = pre->next;
				head->next->next = next;
				head = next;
			}
		}
};

class Solution{
	public:
		void reorderList(ListNode *head) {
			if(!head||!head->next||!head->next->next) return false;
			ListNode *fast = head, *slow = head;
			while(fast->next||fast->next->next){
				slow = slow->next;
				fast = fast->next->next;
			}
			ListNode *mid = slow->next;
			ListNode *last = mid,*pre = NULL;
			while(last){
				ListNode *next = last->next;
				last->next = pre;
				pre = last;
				last = next;
			}
			while(head&&pre){
				ListNode *next = head->next;
				head->next = pre;
				pre = pre->next;
				head->next->next = next;
				head = next;
			}
		}
};

class Solution {
public:
    void reorderList(ListNode *head) {
		if(!head||!head->next||!head->next->next) return;
		stack<ListNode*> st;
	 	ListNode* cur = head;
	 	while(cur){
	 		st.push(cur);
	 		cur = cur->next;
		 }
		 int cnt = (st.size()-1)/2;
		 cur = head;
		 while(cnt--){
		 	auto t = st.top();
		 	st.pop();
		 	ListNode* next = cur->next;
		 	cur->next = t;
		 	t->next = next;
		 	cur = next;
		 	
		 }
		 st.top()->next = NULL;
    }
};
