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

class Solution {
public:
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
