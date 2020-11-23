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
    bool isPalindrome(ListNode* head) {
    	if(!head) return NULL;
    	int len = getlength(head);
        int half = len/2;
        queue<int> q;
    	int t;
        for(int i=0;i<half;i++){
        	q.push(head->val);
        	head = head->next;
		}
		if(len%2==0){
			for(int i=0;i<half;i++){
				t = q.front();
				if(head->val!=t) return false;
				head = head->next;
				q.pop();
			}
			return true;
		}
		else{
			head = head->next;
			for(int i=0;i<half;i++){
				t = q.front();
				if(head->val!=t) return false;
				head = head->next;
				q.pop();
			}
			return true;			
		}
    }
    int getlength(ListNode *l){
    	if(!l) return 0;
    	int cnt = 0;
    	while(l){
    		cnt++;
    		l = l->next;
		}
		return cnt;
	}
};
