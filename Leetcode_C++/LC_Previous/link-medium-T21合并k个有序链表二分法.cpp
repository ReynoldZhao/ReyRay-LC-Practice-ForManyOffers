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
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if(lists.empty()) return NULL;
        int n = lists.size();
        while(n>1){
            int k = (n+1)/2;
            for(int i=0;i<n/2;i++){
                lists[i] = merge(lists[i],lists[i+k]);
            }
            n = k;
        }
        return lists[0];
    }
    ListNode* merge(ListNode* l1, ListNode* l2){
        ListNode *head = new ListNode(-1), *cur = head;
        while(l1&&l2){
            if(l1->val<l2->val){
                cur->next = l1;
                l1 = l1->next;
              }
            else{
                cur->next = l2;
                l2 = l2->next;
              }
             cur = cur->next;		  
          }
          cur->next = l1?l1:l2;
          return head->next;
  }
};
