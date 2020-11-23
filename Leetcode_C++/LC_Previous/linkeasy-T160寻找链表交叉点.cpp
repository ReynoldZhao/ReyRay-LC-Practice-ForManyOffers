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
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(!headA||!headB) return NULL;
        int la = getlength(headA),lb = getlength(headB);
        if(la>lb){
        	for(int i=0;i<la-lb;i++){
        		headA = headA->next;
			}
		}
		if(lb>la){
        	for(int i=0;i<lb-la;i++){
        		headB = headB->next;
			}			
		}
		while(headA&&headB&&headA!=headB){
			headA = headA->next;
			headB = headB->next;
			if(headA==headB) return headA;
		}
		return (headA&&headB)?headA:NULL;
    }
    int getlength(ListNode *l){
    	int cur = 0;
    	if(!l) return cur;
    	while(l){
    		cur++;
    		l = l->next;
		}
		return cur;
	} 
};
