//反转链表，我最喜欢用dummy,然后头插法，每次插到dummy后面

//递归
class SolutionT206 {
public:
    ListNode* reverseList(ListNode* head) {
        if (!head||!head->next) return head;
        ListNode* dummy = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return dummy;
    }

    //cur指向已翻转成功的头部节点，pre永远指向下一个要反转的节点
    ListNode* reverseList(ListNode* head) {
        ListNode* cur = NULL, *pre = head;
        while (pre != NULL) {
            ListNode* t = pre->next;
            pre->next = cur;
            cur = pre;
            pre = t;
        }
        return cur;
    }
};
//reverse K
class SolutionT25 {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        if (!head) return nullptr;
        ListNode* cur = head;
        int count = 0;
        while(cur){
            cur = cur->next;
            count++;
        }
        if (count < k) return head;
        ListNode* pre = head, cur = nullptr;
        for (int i = 1; i <= k; i++) {
            ListNode* t = pre->next;
            pre->next = cur;
            cur = pre;
            pre = t;
        }
        head->next = reverseKGroup(pre, k);
        return cur;
    }

    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode *dummy = new ListNode(-1), *pre = dummy, *cur = pre;
        dummy->next = head;
        int num = 0;
        while (cur = cur->next) ++num;
        //我喜欢的头插法，因为cur直接就是反转前的第一个，在pre的后面，用t插入到pre的后面的时候，cur直接
        //成了最后一个节点，并且cur后面连接下一个要反转的节点
        while (num >= k) {
            cur = pre->next;
            for (int i = 1; i < k; ++i) {
                ListNode *t = cur->next;
                cur->next = t->next;
                t->next = pre->next;
                pre->next = t;
            }
            pre = cur;
            num -= k;
        }
        return dummy->next;
    }
};