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

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> res;
        helper(res, "", 0, 0, n);
        return res;
    }
    void helper(vector<string> res, string s, int left, int right, int max){
    	if(left < right) return;
    	if(left == right == 3) res.push_back(s);
    	else{
    		if(left <= max) helper(res, s + '(', left + 1, right, max);
    		if(right < left) helper(res, s + ')', left, right + 1, max);
		}
	}
};

class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
		if(lists.size() == 0) return null;
		
		auto cmp = [](ListNode*& a, ListNode*& b){
			return a->val > b->val;
		}
		
    }
};

class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
  		vector<vector<int>> res;
		vector<int> out, visited(nums.size(), 0);
		DFS();
		      
    }
    void DFS1(vector<vector<int>>& res, int level, vector<int>& visited, vector<int>& out, vector<int>& nums){
    	if( level == nums.size()) {
			res.push_back(out);
			return;
		}
    	for (int i = 0; i < nums.size(); i++){
    		if (visited[i]!=0) continue;
    		visited[i] = 1;
    		out.push_back(nums[i]);
    		DFS()
    		out.pop_back();
    		visited[i] = 0;
		}
	}
	void DFS2(vector<vector<int>>& res, int start, vector<int>& nums){
		if (start >= nums.size()) {
			res.push_back(nums);
			return;
		}
		for (int i = start; i < nums.size(); ++i){
			swap(nums[start], nums[i]);
			DFS(res, start + 1, nums);
			swap(nums[start], nums[i]);
		}
	}
};

class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int size = nums.size();
        int i = j = 0;
        for (i = nums.size() - 2; i >= 0; i--){
        	if (nums[i] >= nums[i - 1]) continue;
        	else {
        		for (j = nums.size() - 1; j > i; j--){
        			if (nums[j] > nums[i]) swap(nums[i], nums[j]);
        			break;
				}
				reverse(nums.begin() + i, nums.end());
				return;
			}
		}
		return;
    }
};

class Solution {
public:
    int longestValidParentheses(string s) {
		stack<int> st;
		for (int i = 0; i < s.size(); i++){
			if (s[i] == '(') st.push(i);
			else if (s[i] == ')'){
				if (!s.empty()){
					if (s[st.top()] == '(') st.pop();
					else st.push(i);
				}
				else st.push(i);
			}
		}
		int longest = 0;
		if (st.empty()) longest = s.length();
		else {
			int a = n, b = 0;
			while(!st.empty()){
				b = st.top();
				st.pop();
				longest = max(longest, a - b -1);
				a = b;
			}
		}
		return longest;
		
    }
};

class Solution {
    public int divide(int dividend, int divisor) {
        if (dividend == INT_MIN && divisor == -1) return INT_MAX;
        int mark =  ((dividend < 0) ^ (divisor < 0)) ? -1 : 1;
        long int dividend = labs(dividend), divisor = labs(divisor);
    	long int t = divisor, p = 0;
    	while (dividend > divisor){
    		long int temp = divisor, p = 1;
    		while(dividend >= (temp << 1)){
    			temp << 1;
    			p << 1;
			}
			dividend = dividend - temp;
			ans + = p;
			
    		
		}
		long int remain = dividend - (divisor<<t) 
    }
}
