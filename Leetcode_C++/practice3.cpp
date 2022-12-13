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
#include<unordered_set>
#include<hash_map>
#include<deque>
using namespace std;

class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node() {
        val = 0;
        neighbors = vector<Node*>();
    }
    Node(int _val) {
        val = _val;
        neighbors = vector<Node*>();
    }
    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};


class Solution {
public:
    Node* cloneGraph(Node* node) {
        if (!node) return NULL;
        unordered_map<Node*, Node*> map;
        queue<Node*> q({node});
        Node *clone = new Node(node->val);
        Node* temp = new Node(node->val);
        map[node] = clone;
        while(!q.empty()) {
            auto t = q.front(); q.pop();
            for (auto neighbor : t->neighbors) {
                if (!map.count(neighbor)) {
                    map[neighbor] = new Node(neighbor->val);
                    q.push(neighbor);
                }
                map[t]->neighbors.push_back(map[neighbor]);
            }
        }
        return clone;
    }
};

class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> set(wordList.begin(), wordList.end());
        if (!set.count(endWord)) return 0;
        queue<string> q({beginWord});
        int res = 0;
        while (!q.empty())
        {
            for (int i = 0; i < q.size(); i++) {
                auto word = q.front(); q.pop();
                if (word == endWord) return res + 1;
                for (int j = 0; j < word.size(); j++) {
                    string newWord = word;
                    for (char ch = 'a'; ch <= 'z'; ch++) {
                        newWord[j] = ch;
                        if (set.count(newWord) && newWord != word) {
                            q.push(newWord);
                            set.erase(newWord);
                        }
                    }
                }
            }
            ++res;
        }
        return 0;
    }
};

class SolutionT261 {
public:
    bool validTree(int n, vector<pair<int, int>>& edges) {
        vector<vector<int>> graph(n, vector<int>());
        vector<bool> visited(n, false);
        for (auto edge : edges) {
            graph[edge.first].push_back(edge.second);
            graph[edge.second].push_back(edge.first);
        }
        if (!dfs(graph, visited, 0, -1)) return false;
        for (auto a : v) {
            if (!a) return false;
        }
        return true;
    }

    bool dfs(vector<vector<int>> &g, vector<bool> &v, int cur, int pre) {
        if (v[cur]) return false;
        v[cur] = true;
        for (auto a : g[cur]) {
            if (a != pre) {
                if (!dfs(g, v, a, cur)) return false;
            }
        }
        return true;
    }

    bool validTree(int n, vector<pair<int, int>>& edges) {
        vector<int> root(n, 0);
        for (int i = 0; i < n; i++) root[i] = i;
        for (auto a : edges) {
            int x = findRoot(root, a.first), y = findRoot(root, a.second);
            if (x == y) return false;
            roots[x] = y;
        }
        return edges.size() == n - 1;
    }

    int findRoot(vector<int> &roots, int i) {
        return i == roots[i] ? i : findRoot(roots, roots[i]);
    }
};

class Solution {
public:
    /*
     * @param s: a string
     * @param dict: a set of n substrings
     * @return: the minimum length
     */
    int minLength(string &s, unordered_set<string> &dict) {
        int N = s.size();
        if (N == 0) return 0;
        queue<string> q({s});
        unordered_set<string> visited;
        int minLen = N;
        while(!q.empty()) {
            int t_size = q.size();
            for (int i = 0; i < t_size; i++) {
                string temp = q.front(); q.pop();
                for (auto sub_str : dict) {
                    int pos = temp.find(sub_str);
                    while(pos != -1) {
                        string new_str = s.substr(0, pos) + s.substr(pos + sub_str.size());
                        if (visited.find(new_str) == visited.end()) {
                            q.push(new_str);
                            visited.insert(new_str);
                            minLen = min(minLen, new_str.size());
                        }
                        pos = temp.find(sub_str, pos + 1); //这一步保证遍历到当前状态string中，
                        //当前遍历到的字典字符出现的所有位置！
                    }
                }
            }
        }
        return minLen;
    }
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class SolutionT94 {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        if (!root) return res;
        stack<TreeNode*> st;
        TreeNode* p = root;
        while(p || !st.empty()) {
            while(p) {
                st.push(p);
                p = p->left;
            }
            p = st.top(); st.pop();
            res.push_back(p->val);
            p = p->right;
        }
    }
};

class SolutionT144 {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        if (!root) return res;
        stack<TreeNode*> st;
        TreeNode* p = root;
        while(p || !st.empty()) {
            if (p) {
                st.push(p);
                res.push_back(p->val);
                p = p->left;
            } else {
                p = st.top(); st.pop();
                p = p->right;
            }
        }        
    }
};

class SolutionT145 {
public:

    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> s;
        TreeNode *p = root;
        while (!s.empty() || p) {
            if (p) {
                s.push(p);
                res.insert(res.begin(), p->val);
                p = p->right;
            } else {
                TreeNode *t = s.top(); s.pop();
                p = t->left;
            }
        }
        return res;       
    }
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> res;
        if (!root) return res;
        stack<TreeNode*> st({root});
        while(!st.empty()) {
            TreeNode* temp = st.top(); st.pop();
            res.insert(res.begin(), temp->val);
            if (temp->right) st.push(temp->right);
            if (temp->left) st.push(temp->left);
        }
    }
};

class Solution {
public:
    TreeNode* constructFromPrePost(vector<int>& pre, vector<int>& post) {
        return buildTree(pre, 0, pre.size() - 1, post, 0, post.size()-1);
    }
    TreeNode *buildTree(vector<int> &preorder, int preLeft, int preRight, vector<int> &postorder, int pLeft, int pRight){
        if (preLeft > preRight || pLeft > pRight) return NULL;
        TreeNode* root = new TreeNode(preorder[preLeft]);
        if (preLeft + 1 <= preRight) {
            int leftRoot = preorder[preLeft + 1];
            int i = 0;
            for (i = pLeft; i <= pRight; i++) {
                if (postorder[i] == leftRoot) break;
            }
            int leftLen = i - pLeft + 1;
            root->left = buildTree(preorder, preLeft + 1, preLeft + leftLen, postorder, pLeft, pLeft + leftLen - 1);
            root->right = buildTree(preorder, preLeft + leftLen + 1, preRight, postorder, pLeft + leftLen, pRight);
        }
    }
};

//相当于中序遍历的非递归
class BSTIterator {
public:
    BSTIterator(TreeNode* root) {
        while(root) {
            st.push(root);
            root = root->left;
        }
    }
    
    int next() {
        auto temp = st.top();
        st.pop();
        if (temp->right) {
            temp = temp->right;
            while (temp) {
                st.push(temp);
                temp = temp->left;
            }
        }
        return temp->val;
    }
    
    bool hasNext() {
        return !st.empty();
    }
private:
    stack<TreeNode*> st;
};

class SolutionT230 {
public:
    struct MyTreeNode {
        int val;
        int count;
        MyTreeNode *left;
        MyTreeNode *right;
        MyTreeNode(int x) : val(x), count(1), left(NULL), right(NULL) {}
    };

    MyTreeNode* build(TreeNode* root) {
        if (!root) return NULL;
        MyTreeNode* myRoot = new MyTreeNode(root->val);
        myRoot->left = build(root->left);
        myRoot->right = build(root->right);
        if (myRoot->left) myRoot->count += myRoot->left->count;
        if (myRoot->right) myRoot->count += myRoot->right->count;
        return myRoot;
    }

    int kthSmallest(TreeNode* root, int k) {
        MyTreeNode *myRoot = build(root);
        return helper(myRoot, k);
    }

    int helper(MyTreeNode* root, int k) {
        if (root->left) {
            int cnt = root->left->count;
            if (k <= cnt) {
                return helper(root->left, k);
            } else if (k > cnt + 1) {
                return helper(root->right, k - cnt - 1);
            }
            return root->val;
        } else {
            if (k == 1) return root->val;
            return helper(root->right, k - 1);

        }
    }
};

//找到比p大的第一个节点，不要老是想着找到p的后继节点
class Solution285 {
public:
    TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p) {
        TreeNode* res = NULL;
        while(root) {
            if (root->val > p->val) {
                res = root;
                root = root->left;
            } else root = root->right;
        }
    }
};

class SolutionT270 {
public:
    int closestValue(TreeNode* root, double target) {
        int res = root->val;
        while(root) {
            if (abs(res - target) >= abs(root->val - target)) {
                res = root->val;
            }
            root = target > root->val? root->right : root->left;
        }
    }

    int closestValue(TreeNode* root, double target) {
        int a = root->val;
        TreeNode *t = target < a ? root->left : root->right;
        if (!t) return a;
        int b = closestValue(t, target);
        return abs(a - target) < abs(b - target) ? a : b;
    }
};

class Solution {
public:
    vector<int> closestKValues(TreeNode* root, double target, int k) {
        vector<int> res;
        queue<int> q;
        inorder(root, target, k, q);
        return res;
    }
    void inorder(TreeNode *root, double target, int k, queue<int> &res) {
        if (!root) return ;
        inorder(root->left, target, k, res);
        int cur_val = root->val;
        if (res.size() < k) res.push(cur_val);
        else if (abs(double(target - cur_val)) < abs(double(target - res.front()))) {
            res.pop();
            res.push(cur_val);
        }
        inorder(root->right, target, k, res);
    }

    vector<int> closestKValues(TreeNode* root, double target, int k) {
        vector<int> res;
        stack<TreeNode*> s;
        TreeNode *p = root;
        while (p || !s.empty()) {
            while (p) {
                s.push(p);
                p = p->left;
            }
            p = s.top(); s.pop();
            if (res.size() < k) res.push_back(p->val);
            else if (abs(p->val - target) < abs(res[0] - target)) {
                res.erase(res.begin());
                res.push_back(p->val);
            } else break;
            p = p->right;
        }
        return res;
    }
};

class SolutionT510 {
public:
    Node* inorderSuccessor(Node* node) {
        if (!node) return nullptr;
        if (node->right) {
            Node* res = node->right;
            while(res->left) res = res->left;
            return res;
        }
        else {
            Node* res = node->parent;
            Node* cur = node;
            while(res && res->left != cur) {
                cur = res;
                res = res->parent;
            }
            return res;
        }
    }
};

class SolutionT915 {
public:
    /**
     * @param root: the given BST
     * @param p: the given node
     * @return: the in-order predecessor of the given node in the BST
     */
    TreeNode * inorderPredecessor(TreeNode * root, TreeNode * p) {
        // write your code here
        if (!p) return nullptr;
        if (p->left) {
            TreeNode* temp = root->left;
            while(temp->right) {
                temp = temp->right;
            }
            return temp;
        }
        TreeNode* pre = nullptr;
        while(root) {
            if (p->val <= root->val) {
                root = root->left;
            } else {
                if (!pre || root->val > pre->val) {
                    pre = root;
                }
                root = root->right;
            }
        }
        return pre;
    }
};

class SolutionT110 {
public:
    bool isBalanced(TreeNode* root) {
        return helper(root) != -1;
    }

    int helper(TreeNode* root) {
        if (!root) return 0;
        int lheight = helper(root->left);
        if (lheight == -1) return -1;
        int rheight = helper(root->right);
        if (rheight == -1 || abs(lheight - rheight) > 1) return -1;
        return max(lheight, rheight) + 1;
    }
};


//这个写法的错误在于，默认了root就在BST里，递归进去的max_val, min_val会一直在里面
class SolutionT333 {
public:
    int largestBSTSubtree(TreeNode* root) {
        int res = 0, mn = INT_MIN, mx = INT_MAX;
        helper(root, mn, mx, res);
        return res;
    }

    void helper(TreeNode* root, int& mn, int& mx, int& res) {
        if (!root) return ;
        int left_cnt = 0, right_cnt = 0;
        int left_mn = INT_MIN, right_mn = INT_MIN;
        int left_mx = INT_MAX, right_mx = INT_MAX;
        helper(root->left, left_mn, left_mx, res);
        helper(root->right, right_mn, right_mx, res);
        if ((!root->left || root->val > left_mx) && (!root->right || root->val < right_mn)) {
            res = left_cnt + right_cnt + 1;
            mn = root->left ? left_mn : root->val;
            mx = root->right ? right_mx : root->val;
        } else {
            res = max(left_cnt, right_cnt);
        }
    }
};

class SolutionT113 {
public:
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        vector<vector<int>> res;
        vector<TreeNode*> st;
        TreeNode *cur = root, *pre = nullptr;
        int val = 0;
        while(cur || !st.empty()) {
            while(cur) {
                st.push_back(cur);
                val += cur->val;
                cur = cur->left;
            }
            cur = st.back();
            if (!cur->left && !cur->right && val == sum) {
                vector<int> v;
                for (auto &a : st) v.push_back(a->val);
                res.push_back(v);
            }
            if (cur->right && cur->right != pre) {
                cur = cur->right;
            } else {
                pre = cur;
                val -= cur->val;
                st.pop_back();
                cur = nullptr;
            }
        }
    }
};

class SolutionT298 {
public:
    int longestConsecutive(TreeNode* root) {
        int res = 1;
        helper(root, nullptr, 0, res);
        return res;
    }

    void helper(TreeNode* root, TreeNode* pre, int cur_len, int &res) {
        if (!root) return ;
        if (pre && root->val = pre->val + 1) {
            cur_len += 1;
            if (cur_len > res) res = cur_len;
            helper(root->left, root, cur_len, res);
            helper(root->right, root, cur_len, res);
        } else {
            helper(root->left, root, 1, res);
            helper(root->right, root, 1, res);
        }
    }
};

class SolutionT549 {
public:
    int longestConsecutive(TreeNode* root) {
        int res = 0;
        helper(root, root, res);
        return res;
    }

    pair<int, int> helper(TreeNode* root, TreeNode* parent, int &res) {
        if (!root) return {0, 0};
        auto left = helper(root->left, root, res);
        auto right = helper(root->right, root, res);
        res = max(res, left.first + right.second + 1);
        res = max(res, left.second + right.first + 1);
        int inc = 0, dec = 0;
        if (root->val == parent->val + 1) {
            inc = max(left.first, right.first) + 1;
        } else if (root->val == parent->val - 1) {
            dec = max(left.second, right.second) + 1;
        }
        return {inc, dec};
    }
};

struct MultiTreeNode {
    int val;
    vector<MultiTreeNode*> children;
    MultiTreeNode(int x) : val(x) {}
};

class SolutionT548 {
public:
    /**
     * @param root the root of k-ary tree
     * @return the length of the longest consecutive sequence path
     */
    int longestConsecutive3(MultiTreeNode* root) {
        int res = 0;
        helper(root, res);
        return res;
    }

// incLen存从当前节点向下的最长上升路径的长度；decLen存从当前节点向下的最长下降路径的长度
//  只存最值 
    pair<int, int> helper(MultiTreeNode* cur, int &res) {
        if (!cur) return {0, 0};
        int inc = 1, dec = 1;
        for (auto child : cur->children) {
            if (child) {
                auto temp = helper(child, res);
                if (cur->val = child->val + 1) {
                    inc = max(inc, temp.first + 1);
                }
                if (cur->val = child->val - 1) {
                    dec = max(inc, temp.second + 1);
                }
            }
        }
        res = max(res, inc + dec - 1);
        return {inc, dec};
    }
};

class SolutionT442 {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        vector<int> res;
        for (int i = 0; i < nums.size(); i++) {
            while(nums[i] != i + 1) {
                //由于重复必然陷入死循环
                int temp = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = temp;
            }
        }
    }
};

class SolutionT48 {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for (int i = 0; i < n/2; i++) {
            for (int j = i; j < n - 1 - i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - j][i];
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j];
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i];
                matrix[j][n - 1 - i] = temp;
            }
        }
    }
};

class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        int up = 0, left = 0, down = m - 1, right = n - 1;
        vector<int> res;
        while(true) {
            for (int i = left ; i <= right; i++) {
                res.push_back(matrix[up][i]);
            }
            if (++up > down) break;
            for (int j = up; j <= down; j++) {
                res.push_back(matrix[j][right]);
            }
            if (--right < left) break;

        }
    }
};

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}     ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class SolutionT25 {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode *dummy = new ListNode(-1), *pre = dummy, *cur = head;
        dummy->next = head;
        int num = 0;
        while (cur = cur ->next) ++num;
        while (num >= k) {
            cur = pre->next;
            for (int i = 0; i < k; ++i) {
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

class SolutionT82 {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode* dummy = new ListNode(-1), *pre = dummy;
        dummy->next = head;
        while(pre->next) {
            ListNode* cur = pre->next, *temp = cur->next;
            if (!temp) break;
            if (cur->val != temp->val) {
                pre = cur;
                continue;
            }
            while (temp && cur->val == temp->val) {
                temp = temp->next;
            }
            if (!temp) {
                pre->next = nullptr;
                break;
            } else {
                pre->next = temp;
            }
        }
        return dummy->next;
    }

    ListNode* deleteDuplicates(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode* dummy = new ListNode(-1), *pre = dummy;
        dummy->next = head;
        while(pre->next) {
            ListNode *cur = pre->next;
            while(cur->next && cur->next->val = cur->val) {
                cur = cur->next;
            }
            if (cur != pre->next) {
                pre->next = cur->next;
            } else pre = pre->next;
        }
        return dummy->next;
    }
};

class SolutionT86 {
public:
    ListNode* partition(ListNode* head, int x) {
        
    }
};


class SolutionT92 {
public:
    ListNode *reverseBetween(ListNode *head, int m, int n) {
        ListNode* dummy = new ListNode(-1), *pre = dummy;
        dummy->next = head;
        for (int i = 0; i < m - 1; i++) pre = pre->next;
        ListNode* cur = pre->next;
        for (int i = m; i < n; i++) {
            ListNode* t = cur->next;
            cur->next = t->next;
            t->next = pre->next;
            pre->next = t;
        }
        return dummy->next;
    }
};

class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};

class SolutionT138 {
public:
    Node* copyRandomList(Node* head) {
        unordered_map<Node*, Node*> map;
        return helper(head, map);
    }

    Node* helper(Node* node, unordered_map<Node*, Node*>& m) {
        if (!node) return nullptr;
        if (m.count(node)) return m[node];
        Node *res = new Node(node->val);
        m[node] = res;
        res->next = helper(node->next, m);
        res->random = helper(node->random, m);
        return res;
    }
};

class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (!head) return nullptr;
        Node *cur = head;
        while (cur) {
            Node *t = new Node(cur->val);
            t->next = cur->next;
            cur->next = t;
            cur = t->next;
        }
        cur = head;
        while (cur) {
            if (cur->random) cur->next->random = cur->random->next;
            cur = cur->next->next;
        }
        cur = head;
        Node* res = head->next;
        while(cur) {
            Node* t = cur->next;
            cur->next = t->next;
            if (t->next) t->next = t->next->next;
            cur = cur->next;
        }
        return res;
    }
};

class SolutionT142 {
public:
    ListNode* sortList(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode* slow = head, *fast = head, *pre = head;
        while(fast && fast->next) {
            pre = slow;
            slow = slow->next;
            fast = fast->next->next;
        }
        pre->next = nullptr;
        return merge(sortList(head), sortList(slow));
    }

    ListNode* merge(ListNode* l1, ListNode* l2) {
        ListNode *dummy = new ListNode(-1);
        ListNode *cur = dummy;
        while (l1 && l2) {
            if (l1->val < l2->val) {
                cur->next = l1;
                l1 = l1->next;
            } else {
                cur->next = l2;
                l2 = l2->next;
            }
            cur = cur->next;
        }
        if (l1) cur->next = l1;
        if (l2) cur->next = l2;
        return dummy->next;
    }
};

class SolutionT206 {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* dummy = new ListNode(-1);
        while (head) {
            ListNode* t = head->next;
            head->next = dummy->next;
            dummy->next = head;
            head = t;
        }
        return dummy->next;
    }

    ListNode* reverseList(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode* newHead = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return newHead;
    }
};

class SolutionT328 {
public:
    ListNode* oddEvenList(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode *odd = head, *even = head->next, *even_head = even;
        while (even && even->next) {
            odd->next = even->next;
            odd = odd->next;
            even->next = odd->next;
            even = even->next;
        }
        odd->next = even_head;
        return head;
    }
};

class Solution {
public:
    vector<int> buildArray(vector<int>& nums) {
        int q = nums.size();
        for (int i = 0 ; i < q; i++) {
            int r = nums[i];
            int b = nums[nums[i]] % q;
            nums[i] = b * q + r;
        }
        for (int i = 0; i < q; i++) {
            nums[i] /= q;
        }
        return nums;
    }
};

class Solution {
public:
    int eliminateMaximum(vector<int>& dist, vector<int>& speed) {
        vector<double> time(dist.size(), 1.0);
        for (int i = 0; i < dist.size(); i++) {
            time[i] = (double)dist[i] / (double)speed[i] ;
        }
        int res = 1;
        sort(time.begin(), time.end());
        for (int i = 1; i < dist.size(); i++) {
            if (time[i] > time[i-1] + 1 || time[i] > i) res++;
            else break;
        }
        return res;
    }
};

class Solution {
public:
    const int mod = 1e7;
    const int base = 113;

    long long qPow(long long x, long long n) {
        long long ret = 1;
        while(n) {
            if (n & 1) {
                res = res * x %mod;
            }
            x = x * x % mod;
            n >>= 1;
        }
        return ret;
    }

    bool check(vector<int>& A, vector<int>& B, int len) {
        long long hashA = 0;
        for (int i = 0 l i < len; i++) {
            hashA = (hashA* base + A[i]) % mod;
        }
        unordered_set<long, long> bucketA;
        bucketA.insert(hashA);
        long long mult = qPow(base, len - 1);
        for (int i = len; i < A.size(); i++) {
            hashA = ((hashA - A[i-len]*mult %mod + mod) % mod *base +A[i]) %mod;
            bucketA.insert(hashA);
        }
        long long hashB = 0;
        for (int i = 0; i < len; i++) {
            hashB = (hashB * base + B[i]) % mod;
        }
        if (bucketA.count(hashB)) {
            return true;
        }
        for (int i = len; i < B.size(); i++) {
            hashB = ((hashB - B[i - len] * mult % mod + mod) % mod * base + B[i]) % mod;
            if (bucketA.count(hashB)) {
                return true;
            }
        }
        return false;
    }

    int findLength(vector<int>& A, vector<int>& B) {
        int left = 1, right = min(A.size(), B.size()) + 1;
        while (left < right) {
            int mid = (left + right) >> 1;
            if (check(A, B, mid)) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left - 1;
    }

    int findLength(vector<int>& A, vector<int>& B) {
        int n = A.size(), m = B.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
        int ans = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (A[i] == B[j]) {
                    if (i == 0 || j == 0) dp[i][j] = 1;
                    else dp[i][j] = dp[i - 1][j - 1] + 1;
                } else dp[i][j] = 0;
                ans = max(ans, dp[i][j]);
            }
        }
        return ans;
    }
};

class SolutionT1925 {
public:
    int countTriples(int n) {
        //枚举法
    }
};

class SolutionT1926 {
public:
    int nearestExit(vector<vector<char>>& maze, vector<int>& entrance) {
        int m = maze.size();
        int n = maze[0].size();
        vector<int> directX = {0, 1, 0, -1};
        vector<int> directY = {1, 0, -1, 0};
        queue<vector<int>> q({entrance});
        int res = INT_MAX, step = 0;
        while (!q.empty()) {
            step++;
            int size = q.size();
            for (int i = 0; i < size; i++) {
                auto t = q.front(); q.pop();
                //if (checkExit(maze, t)) res = min(res, step - 1);
                for (int k = 0; k < 4; k++) {
                    int temp_row = t[0] + directX[k];
                    int temp_col = t[1] + directY[k];
                    if (0 <= temp_row && temp_row < m && 0 <= temp_col && temp_col < n && maze[temp_row][temp_col] == '.') {
                        if (checkExit(maze, {temp_row, temp_col})) res = min(res, step);
                        q.push({temp_row, temp_col});
                    }
                }
                if (res < INT_MAX) break;
            }
        }
        return res;
    }

    bool checkExit(vector<vector<char>>& maze, vector<int> pos) {
        int m = maze.size();
        int n = maze[0].size();
        if (pos[0] == 0 || pos[0] == m-1 || pos[1] == 0 || pos[1] == n - 1) return true;
        else return false;
    }
};

class Solution {
public:
    TreeNode* Convert(TreeNode* pRootOfTree) {
        if (!pRootOfTree) return nullptr;
        stack<TreeNode*> st;
        TreeNode* p = pRootOfTree, *head;
        TreeNode* pre = nullptr;
        while (p || !st.empty()) {
            if (p) {
                st.push(p);
                p = p->left;
            } else {
                auto t = st.top(); st.pop();
                if (pre) {
                    t->left = pre;
                    pre->right = t;
                } else {
                    head = t;
                }
                pre = t;
                p = t->right;     
            }
        }
        return head;
    }
};

class Solution {
public:
    vector<string> rec;
    vector<int> vis;

    void backstrack(const string& s, int i, int n, string& perm) {
        if (i == n) {
            rec.push_back(perm);
        }
        for (int j = 0; j < n; j++) {
            if (vis[j] || (j > 0 && !vis[j-1] && s[j-1] == s[j])) {
                continue;
            }
            vis[j] = true;
            perm.push_back(s[j]);
            backstrack(s, i+1, n, perm);
            perm.pop_back();
            vis[j] = false;
        }
    }

    vector<string> permutation(string s) {
        int n = s.size();
        vis.resize(n);
        sort(s.begin(), s.end());
        string perm;
        backstrack(s, 0, n, perm);
        return rec;
    }
};


class Solution { 
    int partition(vector<int> &nums, int l, int r) {
        int pivot = nums[r];
        int i = l - 1;
        for (int i = l; i < r; i++) {
            if (nums[j] < pivot) {
                i = i + 1;
                swap(nums[i], nums[j]);
            }
        }
        swap(nums[i+1], nums[r]);
        return i+1;
    }

    int randomized_partition(vector<int>& nums, int l, int r) {
        int i = rand() % (r - l + 1) + l;
        swap(nums[r], nums[i]);
        return partition(nums, l, r);
    }

    void randomized_selected(vector<int>& arr, int l, int r, int k) {
        if (l >= r) return ;
        int pos = randomized_partition(arr, l, r);
        int num = pos - l + 1;
        if (k == num) {
            return ;
        } else if (k < num) {
            randomized_selected(arr, l, pos - 1, k);
        } else {
            randomized_selected(arr, pos + 1, r, k - num);
        }
    }
public: 
    vector<int> getLeastNumbers(vector<int>& arr, int k) { 
        srand((unsigned)time(NULL)); 
        randomized_selected(arr, 0, (int)arr.size() - 1, k); 
        vector<int> vec; 
        for (int i = 0; i < k; ++i) { 
            vec.push_back(arr[i]); 
        } 
        return vec; 
    }    
};

class Solution { 
public: 
    vector<string> rec; 
    vector<int> vis; 

    vector<string> permutation(string s) {
        int n = s.size();
        vis.resize
    }
};

class Solution {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        vector<int> out, visited(nums.size(), 0);
        backtracking(res, out, nums, 0, visited);
        return res;
    }

    void backtracking(vector<vector<int>> &res, vector<int>& out, vector<int>& nums, int k, vector<int> &visited) {
        if (k == nums.size()) {
            res.push_back(out);
            return ;
        }
        for (int i = 0; i < nums.size(); i++) {
            if (visited[i] == 1 || ( i > 0 && nums[i] == nums[i-1] && visited[i-1] == 0)) continue;
            visited[i] = 1;
            out.push_back(nums[i]);
            backtracking(res, out, nums, k + 1, visited);
            out.pop_back();
            visited[i] = 0;
        }
        return ;
    }
};

class SolutionT1927 {
public:
    int minCost(int maxTime, vector<vector<int>>& edges, vector<int>& passingFees) {
        int n = passingFees.size();
        int res = INT_MAX;
        vector<vector<int>> graph(n);
        vector<unordered_map<int, int>> cost(n);
        for (auto e : edges) {
            graph[e[0]].push_back(e[1]);
            graph[e[1]].push_back(e[0]);
            cost[e[0]][e[1]] = e[2];
            cost[e[1]][e[0]] = e[2];
        }
        queue<vector<int>> q({{0, 0, 0}});
        while(!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                auto temp = q.front(); q.pop();
                if (temp[1] > maxTime) continue;
                if (temp[0] == n-1 && temp[1] <= maxTime) {
                    res = min(res, temp[2]);
                }
                int cur_pos = temp[0], cur_cost = temp[1], cur_fee = temp[2];
                for (int j = 0; j < graph[cur_pos].size(); j++) {
                    int next_pos = graph[cur_pos][j];
                    int temp_cost = cur_cost + cost[cur_pos][next_pos];
                    int temp_fee = cur_fee + passingFees[next_pos];
                    q.push({graph[cur_pos][j], temp_cost});
                }
            }
        }
        return res == INT_MAX ? -1 : res;
    }
};

class Solution {
private:
    // 极大值
    static constexpr int INFTY = INT_MAX / 2;

public:
    int minCost(int maxTime, vector<vector<int>>& edges, vector<int>& passingFees) {
        int n = passingFees.size();
        vector<vector<int>> f(maxTime + 1, vector<int>(n, INFTY));
        f[0][0] = passingFees[0];
        for (int t = 1; t <= maxTime; t++) {
            for (const auto& edge: edges) {
                int i = edge[0], j = edge[1], cost = edge[2];
                if (cost <= t) {
                    f[t][i] = min(f[t][i], f[t - cost][j] + passingFees[i]);
                    f[t][j] = min(f[t][j], f[t - cost][i] + passingFees[j]);
                }
            }
        }
        int ans = INFTY;
        for (int t = 1; t <= maxTime; ++t) {
            ans = min(ans, f[t][n - 1]);
        }
        return ans == INFTY ? -1 : ans;
    }    
};

class SolutionT1946 {
public:
    string maximumNumber(string num, vector<int>& change) {
        bool isChanged = false;
        for (int i = 0; i < num.size(); i++) {
            if (isChanged) break;
            if (change[num[i] - '0'] > num[i]) {
                isChanged = true;
                while(i < num.size() && change[num[i] - '0'] > num[i]) {
                    num[i] = change[num[i] - '0'];
                    i++;
                }
            }
        }
        return num;
    }
};

class MyHashMapT706 {
public:
    /** Initialize your data structure here. */
    MyHashMap() {
        data.resize(1000000, -1);
    }
    
    /** value will always be non-negative. */
    void put(int key, int value) {
        data[key] = value;
    }
    
    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    int get(int key) {
        return data[key];
    }
    
    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    void remove(int key) {
        data[key]  = -1;
    }
    
private:
    vector<int> data;
};

class SolutionT49 {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<string>> res;
        unordered_map<string, int> map;
        for (int i = 0; i < strs.size(); i++) {
            string t = strs[i];
            sort(t.begin(), t.end());
            if (!map.count(t)) {
                map[t] = res.size();
                res.push_back({});
            }
            res[map[t]].push_back(strs[i]);
        }
        return res;
    }
};

class SolutionT128 {
public:
    int longestConsecutive(vector<int>& nums) {
        int res = 0;
        unordered_set<int> set(nums.begin(), nums.end());
        for (int i = 0; i < nums.size(); i++) {
            int t = nums[i];
            if (!set.count(t)) continue;
            int pre = t - 1, next = t + 1;
            while(set.count(pre)) set.erase(pre--);
            while(set.count(next)) set.erase(next++);
            res = max(res, next - pre - 1);
        }
        return res;
    }

    int longestConsecutive(vector<int>& nums) {
        unordered_map<int, int> map;
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int val = nums[i];
            int left = map.count(val - 1) ? map[val - 1] : 0;
            int right = map.count(val + 1) ? map[val + 1] : 0;
            int len = right + left + 1;
            res = max(res, len);
            map[val] = len;
            map[val - left] = len;
            map[val + right] = len;
        }
    }
};

class SolutionT290 {
public:
    bool wordPattern(string pattern, string s) {
        unordered_map<char, int> m1;
        unordered_map<string, int> m2;
        istringstream in(s);
        int i = 0, n = pattern.size();
        for (string word; in >> word; ++i) {
            if (i == n || m1[pattern[i]] != m2[word]) return false;
            m1[pattern[i]] = m2[word] = i + 1;
        }
        return i == n;
    }

    bool wordPattern(string pattern, string s) {
        unordered_map<char, string> map;
        istringstream in(s);
        int i = 0, n = pattern.size();
        for (string word; in >> word; i++) {
            if (i >= n) continue;
            if (!map.count(pattern[i])) {
                map[pattern[i]] = word;
            } else {
                if (word != map[pattern[i]]) return false;
                for (auto a : map) {
                    if (word == a.second) return false;
                }
            }
        }
        return i == n;
    }
};

class SolutionT23 {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        int n = lists.size();
        while (n > 1) {
            int k = (n + 1) / 2;
            for (int i = 0; i < n /2; i++) {
                list[i] = mergeTwoLists(lists[i], lists[i + k]);
            }
            n = k;
        }
        return lists[0];
    }

    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {

    }

     ListNode* mergeKLists(vector<ListNode*>& lists) {
         auto cmp = [](ListNode* a, ListNode* b) {return a->val > b->val;};
         priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> q(cmp);
         for (auto l : lists) {
             if (l) q.push(l);
        }
        ListNode *dummy = new ListNode(-1), *cur = dummy;
        while(!q.empty()) {
            auto temp = q.top(); q.pop();
            cur->next = temp;
            cur = cur->next;
            if (temp->next) q.push(temp->next);
        }
        return dummy->next;
    }
};

class SolutionT767 {
public:
    string reorganizeString(string S) {
        string res = "";
        unordered_map<char, int> map;
        for (auto c : S ) map[c]++;
        priority_queue<pair<int, char>> pq;
        for (auto m : map) {
            if (m.second > (S.size() + 1) / 2) return "";
            pq.push({m.second, m.first});
        }
        while (pq.size() >= 2) {
            auto t1 = pq.top(); pq.pop();
            auto t2 = pq.top(); pq.pop();
            res.push_back(t1.second);
            res.push_back(t2.second);
            if (--t1.first > 0) pq.push(t1);
            if (--t2.first > 0) pq.push(t2);
        }
        if (pq.size() > 0) res.push_back(q.top().second);
        return res;
    }
};

class SolutionT480 {
public:
    //small.size() >= large.size()
vector<double> medianSlidingWindow(vector<int>& nums, int k) { 
        vector<double> res; 
        multiset<int> small, large; 
        for (int i = 0; i < nums.size(); ++i) { 
            if (i >= k) { 
                if (small.count(nums[i - k])) small.erase(small.find(nums[i - k])); 
                else if (large.count(nums[i - k])) large.erase(large.find(nums[i - k])); 
            } 
            //其实根本不会小于
            if (small.size() <= large.size()) { 
                if (large.empty() || nums[i] <= *large.begin()) small.insert(nums[i]); 
                else { 
                    small.insert(*large.begin()); 
                    large.erase(large.begin()); 
                    large.insert(nums[i]); 
                } 
            } else { 
                if (nums[i] >= *small.rbegin()) large.insert(nums[i]); 
                else { 
                    large.insert(*small.rbegin()); 
                    small.erase(--small.end()); 
                    small.insert(nums[i]); 
                } 
            } 
            if (i >= (k - 1)) { 
                if (k % 2) res.push_back(*small.rbegin()); 
                else res.push_back(((double)*small.rbegin() + *large.begin()) / 2); 
            } 
        } 
        return res; 
    }
    
    vector<double> medianSlidingWindow(vector<int>& nums, int k) {
        vector<double> res;
        multiset<double> ms(nums.begin(), nums.begin() + k);
        auto mid = next(ms.begin(), k / 2);
        for (int i = k; ; ++i) {
            res.push_back( (*mid + *prev(mid, 1 - k%2))/2);
            if (i == nums.size()) return res;
            ms.insert(nums[i]);
            if (nums[i] <= *mid) --mid;
            if (nums[i - k] <= *mid) ++mid;
            ms.erase(ms.lower_bound(nums[i - k]));
        } 
    }
};


class SolutionT84 {
public:
    int largestRectangleArea(vector<int>& heights) {
        stack<int> st;
        int res = 0;
        heights.insert(heights.begin(), 0);
        heights.push_back(0);
        for (int i = 0; i < heights.size(); i++) {
            while (!st.empty() && heights[i] <= heights[st.top()]) {
                int cur_pos = st.top(); st.pop();
                int cur_height = heights[cur_pos];
                int cur_area = cur_height * (st.empty() ? i : i - st.top() - 1);
                res = max(res, cur_area);
            }
            st.push(heights[i]);
        }
        return res;
    }
};

class SolutionT85 {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        vector<int> height;
        int res = 0;
        for (int i = 0; i < matrix.size(); i++) {
            height.resize(matrix[i].size(), 0);
            for (int j = 0; j < matrix[i].size(); j++) {
                height[j] = matrix[i][j] == '0' ? 0 : height[j] + 1;
            }
            res = max(res, largestRectangleArea(height));
        }
        return res;
    }

    int largestRectangleArea(vector<int>& heights) {
        stack<int> st;
        int res = 0;
        heights.insert(heights.begin(), 0);
        heights.push_back(0);
        for (int i = 0; i < heights.size(); i++) {
            while (!st.empty() && heights[i] <= heights[st.top()]) {
                int cur_pos = st.top(); st.pop();
                int cur_height = heights[cur_pos];
                int cur_area = cur_height * (st.empty() ? i : i - st.top() - 1);
                res = max(res, cur_area);
            }
            st.push(heights[i]);
        }
        return res;
    }
};

class Solution {
public:
    int calculate(string s) {
        int res = 0, n = s.size();
        stack<int> st;
        char op = '+';
        for (int i = 0; i < n; i++) {
            char c = s[i];
            if (c >= '0') {
                int num = 0;
                while (i < n && s[i] > '0') {
                    num = num * 10 + s[i] - '0';
                }
                if (op == '+') st.push(num);
                if (op == '-') st.push(-num);
                if (op == '*' || op == '/') {
                    int tmp = (op == '*') ? st.top() * num : st.top() / num;
                    st.pop();
                    st.push(tmp);
                }
            } else if ((c < '0' && c != ' ') || i == n - 1) {
                op = c;
            }
        }
        while (!st.empty()) {
            res += st.top();
            st.pop();
        }
        return res;
    }
};

class SolutionT394 {
public:
    string decodeString(string s) {
        int i = 0;
        return decode(s, i);
    }

    string decode(string s, int &i) {
        string res = "";
        int num = 0;
        int n = s.size();
        while (i < n) {
            if (isAlpha(s[i])) res += s[i];
            if (isNum(s[i])) num = num*10 + s[i] - '0';
            if (s[i] == '[') {
                i++;
                string temp_str = decode(s, i);
                i++;
                for (int j = 0; j < num; j++) {
                    res += temp_str;
                }
            }
            if (s[i] == ']') break;
        }
        return res;
    }


    bool isAlpha(char c) {
        return c >= 'a' && c <= 'z';
    }

    bool isNum(char c) {
        return c >= '0' && c <= '9';
    }

    string decodeString(string s) {
        int n = s.size(), i = 0;
        string res = "";
        stack<int> st_num;
        stack<string> st_str;
        while (i < n) {
            if (isAlpha(s[i])) res+=s[i++];
            else if (isNum(s[i])) {
                int temp = 0;
                while (i < n && isNum(s[i])) {
                    temp = temp*10 + s[i++] - '0';
                }
                st_num.push(temp);
            }
            else if (s[i] == '[') {
                i++;
                st_str.push(res);
                res.clear();
            }
            else if (s[i] == ']') {
               int temp_num = st_num.top(); st_num.pop();
               string temp_str = st_str.top(); st_str.pop(); 
               for (int j = 0; j < temp_num; j++) temp_str+=res;
               res = temp_str;
               i++;
            }
        }
        return st_str.empty() ? res : st_str.top();
    }
};

class SolutionT1249 {
public:
    string minRemoveToMakeValid(string s) {
        int left = 0;
        string t = "";
        for (int i = 0; i < s.size(); i++) {
            if (isAlpha(s[i])) {
                t+=s[i];
                continue;
            }
            if (s[i] == '(') left++;
            else if (s[i] == ')') left--;
            if (left >=0) t+=s[i];
        }
        string str = t;
        string res = "";
        int right = 0;
        for (int j = str.size() - 1; j >= 0; --j) {
            if (isAlpha(str[j])) {
                res+=str[j];
                continue;
            }
            if (str[j] == ')') right++;
            else if (str[j] == '(') right--;
            if (right >= 0) res += str[j];       
        }
        reverse(res.begin(), res.end());
        return res;
    }

    bool isAlpha(char c) {
        return c >= 'a' && c <= 'z';
    }

    bool isAlpha(char c) {
        return c >= 'a' && c <= 'z';
    }
};

class SolutionT300 {
public:
    int lengthOfLIS(vector<int>& nums) {
    }

    bool isPrime(int n){
        if(n<=1) return false;//特判
        int sqr=(int)sqrt(n);//根号n
        for(int i=2;i<=sqr;i++){//遍历2~根号n 
            if(n%i==0) return false;//n是i的倍数，则n不是素数 
        } 
        return true;//n是素数 
    }
};


class SolutionT1952 {
public:
    bool isThree(int n) {
        int divisor = (int)sqrt(n);
        if (divisor * divisor != n) return false;
        else {
            return isPrime(divisor);
        }
    }

    bool isPrime(int n){
        if(n<=1) return false;//特判
        int sqr=(int)sqrt(n);//根号n
        for(int i=2;i<=sqr;i++){//遍历2~根号n 
            if(n%i==0) return false;//n是i的倍数，则n不是素数 
        } 
        return true;//n是素数 
    }
};

class Solution {
public:
    long long numberOfWeeks(vector<int>& milestones) {
        priority_queue<int, vector<int>, less<int> > pq;
        long long res = 0;
        for (auto& mile : milestones) {
            pq.push(mile);
        }
        while (pq.size() >= 2) {
            int temp1 = pq.top(); pq.pop();
            int temp2 = pq.top(); pq.pop();
            temp1--;
            temp2--;
            res+=2;
            if (temp1 > 0) pq.push(temp1);
            if (temp2 > 0) pq.push(temp2);
        }
        if (!pq.empty()) return pq.top() == 1 ? res + 1 : res;
        return res;
    }

    long long numberOfWeeks(vector<int>& milestones) {
        int64_t sum = 0, max = 0;
        for (auto d : milestones) {
            sum += d;
            max = (max >d ? max : d);
        }
        sum - max;
        if (max <= sum + 1) {
            return sum + max;
        }
        return 2 * sum + 1;
    }
};

class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> dq;
        for (int i = 0; i < k; i++) {
            while (!dq.empty() && nums[i] >= nums[dq.back()]) dq.pop_back();
            dq.push_back(i);
        }
        vector<int> res;
        for (int i = k; i < nums.size(); i++) {
            res.push_back(nums[dq.front()]);
            if (dq.front() <= i - k) dq.pop_front();
            while (!dq.empty() && nums[i] >= nums[dq.back()]) dq.pop_back();
            dq.push_back(i);
        }
        res.push_back(nums[dq.front()]);
        return res;
    }     
};

class SolutionT1019 {
public:
    vector<int> nextLargerNodes(ListNode* head) {
        vector<int> st;
        ListNode* t = head;
        int index = 0;
        while (t) {
            int temp_val = t->val;
            if (st.empty()) st.push_back(index);
            
        }
    }
};

class SolutionT1019 {
public:
    vector<int> nextLargerNodes(ListNode* head) {
        vector<int> res, nums;
        stack<int> st;
        int cnt = 0;
        ListNode* t = head;
        while (t) {
            nums.push_back(t->val);
            while (!st.empty() && t->val > nums[st.top()]) {
                int temp_index = st.top();
                res[temp_index] = t->val;
                st.pop();
            }
            st.push(cnt);
            res.resize(++cnt);
            t = t->next;
        }
        return res;
    }
};

class SolutionT496 {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int, int> map;
        stack<int> st;
        for (int i = 0; i < nums2.size(); i++) {
            while (!st.empty() && nums2[i] > nums2[st.top()]) {
                int small_pos = st.top();
                map[nums2[small_pos]] = nums2[i];
                st.pop();
            }
            st.push(i);
        }
        vector<int> res(nums1.size(), -1);
        for (int j = 0; j < nums1.size(); j++) {
            if (map.count(nums1[j])) res[j] = map[nums1[j]];
        }
        return res;
    }
};

class SolutionT503 {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        vector<int> inc;
        vector<int> dec;
        vector<int> res(nums.size(), -1);
        for (int i = 0; i < nums.size(); i++) {
            while (!dec.empty() && nums[i] > nums[dec.back()]) {
                int temp_index = dec.back();
                res[temp_index] = nums[i];
                dec.pop_back();
            }
            dec.push_back(i);
        }
        while (dec.size() > 1) {
            if (dec.size() >= 2) {
                res[dec.back()] = nums[dec[0]]; 
            }
            dec.pop_back(); 
        }
        return res;
    }
    // 1,1,1,1,1, 修正
    vector<int> nextGreaterElements(vector<int>& nums) {
        vector<int> dec;
        vector<int> res(nums.size(), -1);
        for (int i = 0; i < nums.size(); i++) {
            while (!dec.empty() && nums[i] > nums[dec.back()]) {
                int temp_index = dec.back();
                res[temp_index] = nums[i];
                dec.pop_back();
            }
            dec.push_back(i);
        }
        while (dec.size() > 1) {
            if (dec.size() >= 2 && nums[dec.back()] < nums[dec[0]]) {
                res[dec.back()] = nums[dec[0]]; 
            }
            dec.pop_back(); 
        }
        return res;
    }

    //[1,2,3,2,1]
    vector<int> nextGreaterElements(vector<int>& nums) {
        int n = nums.size();
        stack<int> st;
        vector<int> res(n, -1);
        for (int i = 0; i < 2 * nums.size(); i++) {
            int cur_pos = i % n;
            while (!st.empty() && nums[cur_pos] > nums[st.top()]) {
                int temp_pos = st.top();
                res[temp_pos] = nums[cur_pos];
                st.pop();
            }
            if (i < n) st.push(i);
        }
        return res;
    }
};

class TrieNode {
public:
    TrieNode *child[26];
    bool isWord;
    TrieNode(): isWord(false) {
        for (auto &a : child) a = nullptr;
    }
};

class Trie {
public:
    /** Initialize your data structure here. */
    Trie() {
        root = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        TrieNode *t = root;
        for (int i = 0; i < word.size(); i++) {
            char cur_c = word[i];
            int index = cur_c - 'a';
            if (!t->child[index]) t->child[index] = new TrieNode();
            t = t->child[index];
        }
        t->isWord = true;
    }
    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        TrieNode *t = root;
        for (int i = 0; i < word.size(); i++) {
            int cur_c = word[i];
            int index = cur_c - 'a';
            if (!t->child[index]) return false;
            t = t->child[index];
        }
        return t->isWord
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        TrieNode *p = root;
        for (auto &a : prefix) {
            int i = a - 'a';
            if (!p->child[i]) return false;
            p = p->child[i];
        }
        return true;       
    }
private:
    TrieNode* root;
};

class WordDictionary {
public:
    struct TrieNode {
    public:
        TrieNode *child[26];
        bool isWord;
        TrieNode() : isWord(false) {
            for (auto &a : child) a = NULL;
        }
    };
    
    WordDictionary() {
        root = new TrieNode();
    }
    
    // Adds a word into the data structure.
    void addWord(string word) {
        TrieNode *p = root;
        for (auto &a : word) {
            int i = a - 'a';
            if (!p->child[i]) p->child[i] = new TrieNode();
            p = p->child[i];
        }
        p->isWord = true;
    }

    // Returns if the word is in the data structure. A word could
    // contain the dot character '.' to represent any one letter.
    bool search(string word) {
        return searchWord(word, root, 0);
    }
    
    bool searchWord(string &word, TrieNode *p, int i) {
        if (i == word.size()) return p->isWord;
        TrieNode* t = p;
        char cur_c = word[i];
        bool isFound = false;
        if (cur_c == '.') {
            for (int i = 0; i < 26; i++) {
                if (t->child[i] && searchWord(word, t->child[i], i+1)) return true;
            }
        } else {
            if (!t->child[cur_c - 'a']) return false;
            return searchWord(word, t->child[cur_c - 'a'], i + 1);
        }
    }

    bool searchWord(string &word, TrieNode *p, int i) {
        if (i == word.size()) return p->isWord;
        if (word[i] == '.') {
            for (auto &a : p->child) {
                if (a && searchWord(word, a, i + 1)) return true;
            }
            return false;
        } else {
            return p->child[word[i] - 'a'] && searchWord(word, p->child[word[i] - 'a'], i + 1);
        }
    }
    
private:
    TrieNode *root;
};

class StreamChecker {
public:
    StreamChecker(vector<string>& words) {
        root = new TrieNode();
        for (auto& word : words) {
            TrieNode *t = root;
            for (int i = word.size() - 1; i >= 0; i--) {
                int cur_c = word[i];
                if (!t->child[cur_c - 'a']) t->child[cur_c - 'a'] = new TrieNode();
                t = t->child[cur_c - 'a'];
            }
            t->isWord = true;
        }
    }
    
    bool query(char letter) {
        queryString.push_back(letter);
        TrieNode* t = root;
        for (int i = queryString.size() - 1; i >= 0; --i) {
            char cur_c = queryString[i];
            if (t->child[cur_c - 'a']) t = t->child[cur_c - 'a'];
            else return false;
            if (t->isWord) return true;
        }
    }
private:
    string queryString;
    TrieNode *root;
};

class Interval {
    int start, end;
    Interval(int start, int end) {
        this->start = start;
        this->end = end;
    }
};

class Solution {
public:
    /**
     * @param airplanes: An interval array
     * @return: Count of airplanes are in the sky.
     */
    int countOfAirplanes(vector<Interval> &airplanes) {
        // write your code here
        vector<int> start_t, end_t;
        for (auto& i : airplanes) {
            start_t.push_back(*i->start);
            end_t.push_back(*i->end);
        }
        for (int i = 0; i < nums.size(); ++i) {
            update(i, nums[i]);
        }
    }
};

class NumArray {
public:
    NumArray(vector<int>& nums) {
        data.resize(nums.size());
        bit.resize(nums.size() + 1);
        for (int i = 0; i < nums.size(); ++i) {
            update(i, nums[i]);
        }
    }
    
    void update(int index, int val) {
        int diff = val - data[index];
        for (int j = index + 1; j < bit.size(); j += (j & -j)) {
            bit[j] += diff;
        }
        data[index] = val;
    }
    
    int sumRange(int left, int right) {
        return getSum(left+1) - getSum(right);
    }
    int getSum(int i) {
        int res = 0;

    }

    int getSum(int i) {
        int res = 0;
        for (int j = i; j > 0; j -= (j&-j)) {
            res += bit[j];
        }
        return res;
    }

private:
    vector<int> data, bit;
};

class SolutionT327 {
public:
    int countRangeSum(vector<int>& nums, int lower, int upper) {
        int sum = 0, res = 0;
        multiset<int> set;
        for (int i = 0; i < nums.size(); i++) {
            sum += nums[i];
            res += distance(set.lower_bound(sum - upper), set.upper_bound(sum - lower));
            set.insert(sum);
        }
        return res;
    }
};

class RangeModule {
public:
    RangeModule() {}

    void addRangde(int left, int right) {
        vector<pair<int, int>> res;
        int n = v.size(), cur = 0;
        for (int i = 0; i < n; ++i) {
            if (v[i].second < left) {
                res.push_back(v[i]);
                ++cur;
            } else if (v[i].first > right) {
                res.push_back(v[i]);
            } else {
                left = min(left, v[i].first);
                right = max(right, v[i].second);
            }
        }
        res.insert(res.begin() + cur, {left, right});
        v = res;
    }

    bool queryRange(int left, int right) {
        for (auto a : v) {
            if (a.first <= left && a.second >= right) return true;
        }
        return false;
    }

    void removeRange(int left, int right) {
        vector<pair<int, int>> res, t;
        int n = v.size(), cur = 0;
        for (int i = 0; i < n; i++) {
            if (v[i].second <= left) {
                res.push_back(v[i]);
                cur++;
            } else if (v[i].first >= right) {
                res.push_back(v[i]);
            } else {
                if (v[i].first < left) {
                    t.push_back({v[i].first, left});
                }
                if (v[i].second > right) {
                    t.push_back({right, v[i].second});
                }
            }
        }
        res.insert(res.begin() + cur, t.begin(), t.end());
        v = res;
    }

private:
    vector<pair<int, int>> v;
};

class RangeModule {
public:
    RangeModule() {}

    void addRange(int left, int right) {
        auto x = find(left, right);
        //利用查询，已经把left，right之间清空了，
        m[x.first] = x.second;
    }

    bool queryRange(int left, int right) {
        auto it = m.upper_bound(left);
        //前面一个的起始位置一定比left小，然后结束位置有比right大
        return it != m.begin() && (--it)->second >= right;
    }

    void removeRange(int left, int right) {
        auto x = find(left, right);
        if (left > x.first) m[x.first] = left;
        if (x.second > right) m[right] = x.second;
    }
private:
    map<int, int> m;

    pair<int, int> find(int left, int right) {
        //这个函数的作用就是查询该范围，在树中，所处的位置
        auto l = m.upper_bound(left), r = m.upper_bound(right);

        --l;
        //由于事upper_bound，去前一个区间看有无重叠，

        if ( l != m.begin() && l->second < left) ++l; 
        //这一步表明，前一个和查询范围无重叠

        if (l == r) return {left, right};
        //这一步表明，没有查询到任何重叠，可以直接操作

        --r;
        //由于是upper_bound，去前一个区间看有无重叠，

        //左部有重叠，l必定在{_s, left, _e}的这个interval里，取min
        //右部有重叠，r必定在{_s, right, _e}的这个interval里，取max

        //之前的e一定小于left了，之后的start也一定大于right了，由于upper_bound
        int i = min(left, l->first), j = max(right, r->second);

        //如果有重叠，重叠的都在l，++r之间
        m.erase(l, ++r);
        return {i, j};
    }
};

class SolutionT315 {
public:
    vector<int> countSmaller(vector<int>& nums) {
        vector<int> t, res(nums.size());
        for (int i = nums.size() - 1; i >= 0; i--) {
            int cur_val = nums[i];
            int left = 0, right = t.size();
            while (left < right) {
                int mid = (right - left) / 2 + left;
                if (t[mid] >= cur_val) right = mid;
                else left = mid + 1;
            }
            t.insert(t.begin() + right, cur_val);
            res[i] = right;
        }
        return res;
    }
};

class Solution {
public:
    int lowbit(int x) {
        return x & (-x);
    }

    void update(int i, vector<int>&C) {
        while (i < C.size()) {
            C[i]++;
            i += lowbit(i);
        }
    }

    void query(int i, int j, vector<int>&C, vector<int>&counts) {
        while (i >= 1) {
            counts[j] += C[i];
            i -= lowbit(i);
        }
    }

    vector<int> countSmaller(vector<int>& nums) {
        vector<int> counts(nums.size(), 0);
        if (nums.size() < 1) {
            return counts;
        }

        vector<int> N(nums);
        sort(N.begin(), N.end());
        int slow = 1, fast = 1;
        while (fast < N.size()) {
            if (N[fast] != N[slow - 1]) {
                N[slow] = N[fast];
                slow++;
                fast++;
            } else {
                fast++;
            }
        }
        N.resize(slow);

        map<int, int> m;
        for (int j = 1; j < 1 + N.size(); ++j) {
            m[N[j-1]] = j;
        }
        //数字与他们在树状数组中的index

        vector<int>C(N.size() + 1, 0);
        int index;
        for (int j = nums.size() - 1; j >= 0 ; --j) {
            index = m[nums[j]];
            update(index, C);
            if (index != 1) {
                //左边都是比当前小的，都要加
                query(index- 1, j, C, counts);
            } else {
                counts[j] = 0;
            }
        }
        return counts;
    }
};