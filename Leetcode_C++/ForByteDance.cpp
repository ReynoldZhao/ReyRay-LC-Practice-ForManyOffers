#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <map>
#include <set>
#include <stack>
#include <utility>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <hash_map>
#include <deque>
using namespace std;

//反转链表
class SolutionT206 {
public:
//递归
    ListNode* reverseList(ListNode* head) {
        if (!head||!head->next) return head;
        ListNode* dummy = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return dummy;
    }
//迭代
    //你自己的插入
    //or
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr;
        ListNode* cur = head;
        while(head!=nullptr) {
            ListNode* nextTemp = cur->next;
            cur->next = pre;
            pre = cur;
            cur = nextTemp;
        }
        return pre;
    }
};

//K个翻转链表
class SolutionT25 {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        if (!head) return nullptr;
        if (k==1) return head;
        ListNode *cur = nullptr, *pre = head;
        for (int i = 1; i <= k; i++) {
            ListNode* temp = pre->next;
            pre->next = cur;
            cur = pre;
            pre = temp;
            if(!pre) break;
        }
        head->next = reverseKGroup(pre, k);
        return cur;
    }

    // public ListNode reverseKGroup(ListNode head, int k) {
    //     if (head == null || head.next == null) {
    //         return head;
    //     }
    //     ListNode tail = head;
    //     for (int i = 0; i < k; i++) {
    //         //剩余数量小于k的话，则不需要反转。
    //         if (tail == null) {
    //             return head;
    //         }
    //         tail = tail.next;
    //     }
    //     // 反转前 k 个元素
    //     ListNode newHead = reverse(head, tail);
    //     //下一轮的开始的地方就是tail
    //     head.next = reverseKGroup(tail, k);

    //     return newHead;
    // }

    // /*
    // 左闭又开区间
    //  */
    // private ListNode reverse(ListNode head, ListNode tail) {
    //     ListNode pre = null;
    //     ListNode next = null;
    //     while (head != tail) {
    //         next = head.next;
    //         head.next = pre;
    //         pre = head;
    //         head = next;
    //     }
    //     return pre;

    // }
};

class LRUCache{
public:
    LRUCache(int capacity) {
        cap = capacity;
    }
    
    int get(int key) {
        auto it = m.find(key);
        if (it != m.end()) {
            l.splice(l.begin(), l, it->second);
            return it->second->second;
        } else {
            return -1;
        }
    }
    
    void put(int key, int value) {
        auto it = m.find(key);
        if (it != m.end()) {
            l.erase(it->second);
        }
        l.push_frond(make_pair(key, value));
        m[key] = l.begin();
        if (l.size() > cap) {
            int k = l.rbegin()->first;
            m.erase(k);
            l.pop_back();
        }
    }
    
private:
    int cap;
    list<pair<int, int>> l;
    unordered_map<int, pair<int,int>> m;
};

//最长无重复子串
class SolutionT3 {
public:
    int lengthOfLongestSubstring(string s) {
        int res = 0, left = -1, n = s.size();
        unordered_map<int, int> m;
        for (int i = 0; i < n; ++i) {
            if (m.count(s[i]) && m[s[i]] > left) {
                left = m[s[i]];  
            }
            m[s[i]] = i;
            res = max(res, i - left);            
        }
        return res;
    }
};

//完全二叉树
class SolutionT958 {
public:
//具体到写法就是先把根结点放入到队列中，然后进行循环，条件是队首结点不为空。在循环中取出队首结点，
//然后将其左右子结点加入队列中，这里不必判断子结点是否为空，为空照样加入队列，因为一旦取出空结点，
//循环就会停止。然后再用个循环将队首所有的空结点都移除，这样若是完全二叉树的话，队列中所有还剩的
//结点都应该是空结点，且都会被移除，若队列中存在非空结点，说明不是完全二叉树，最后只要判断队列是否为空即可，
    bool isCompleteTree(TreeNode* root) {
        queue<TreeNode*> q{{root}};
        while (q.front() != NULL) {
            TreeNode *cur = q.front(); q.pop();
            q.push(cur->left);
            q.push(cur->right);
        }
        while (!q.empty() && q.front() == NULL) {
            q.pop();
        }
        return q.empty();
    }
};

class SolutionT34 {
public:
    void reverseWords(string &s) {
        int storeIndex = 0, n = s.size();
        reverse(s.begin(), s.end());
        for (int i = 0; i < n; ++i) {
            if (s[i] != ' ') {
                if (storeIndex != 0) s[storeIndex++] = ' ';
                int j = i;
                while (j < n && s[j] != ' ') s[storeIndex++] = s[j++];
                reverse(s.begin() + storeIndex - (j - i), s.begin() + storeIndex);
                i = j;
            }
        }
        s.resize(storeIndex);
    }
};

//字符串相加 大数相加
class Solution {
public:
    string addStrings(string num1, string num2) {
        string res = "";
        int m = num1.size(), n = num2.size(), i = m - 1, j = n - 1;
        int sum = 0, carry = 0;
        while (i >= 0 || j >= 0) {
            int a = i >= 0 ? num1[i] - '0' : 0;
            int b = j >= 0 ? num2[j] - '0' : 0;
            sum = a + b + carry;
            carry = sum/10;
            res.insert(res.begin(), sum%10 + '0');

        }
        return carry == 1? "1" + res:res;
    }
};

class SolutionT4 {
public:
//若 m+n 为奇数的话，那么其实 (m+n+1) / 2 和 (m+n+2) / 2 的值相等，相当于两个相同的数字相加再除以2，还是其本身
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size(), left = (m + n + 1)/2, right = (m + n + 2)/2;
    }
};