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
#include<deque>
using namespace std;

struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

// T94 Binary Tree Inorder Traversal 
class SolutionT94 {
public:
    vector<int> inorderTraversal(TreeNode* root) { 
        vector<int> res; 
        stack<TreeNode*> s; 
        TreeNode *p = root; 
        while (!s.empty() || p) { 
            if (p) { 
                s.push(p); 
                p = p->left; 
            } else { 
                p = s.top(); s.pop(); 
                res.push_back(p->val); 
                p = p->right; 
            } 
        } 
        return res; 
    } 
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> s;
        TreeNode* p = root;
        while(p || !s.empty()) {
            while(p) {
                s.push(p);
                p = p->left;
            }
            p = s.top(); s.pop();
            res.push_back(p->val);
            p = p->right;
        }
        return res;
    }

    //Morris
    vector<int> inorderTraversalMorris(TreeNode* root) {
        if (!root) return {};
        vector<int> res;
        TreeNode *cur = root, *pre = NULL;
        while ( cur!=NULL ) {
            if (!cur->left) {
                res.push_back(cur->val);
                cur = cur->right;
            }
            else {
                pre = cur->left;
                while (pre->right && pre->right != cur) pre = pre->right;
                if (!pre->right) {
                    pre->right = cur;
                    res.push_back(cur->val);
                    cur = cur->left;
                }
                else {
                    pre->right = NULL;
                    cur = cur->right;
                }
            }
        }
    }
};

// T144 Binary Tree Preorder Traversal
class SolutionT144 {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> s;
        TreeNode* p = root;
        while (p || !s.empty()) {
            if (p) {
                res.push_back(p->val);
                if (p->right) s.push(p->right);
                p = p->left;
            }
            else {
                p = s.top(); s.pop();
            }
        }
        return res;
    }

//我更喜欢这种
    vector<int> preorderTraversal1(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> st;
        TreeNode *p = root;
        while (!st.empty() || p) {
            if (p) {
                res.push_back(p->val);
                st.push(p);
                p = p->left;
            }
            else {
                p = st.top(); st.pop();
                p = p->right;
            }
        }
        return res;
    }

    vector<int> preorderTraversal2(TreeNode* root) {
        if (!root) return { };
        vector<int> res;
        stack<TreeNode*> s{{root}};
        while (!s.empty()) {
            TreeNode* t = s.top();  s.pop();
            res.push_back(t->val);
            if (t->right) s.push(t->right);
            if (t->left) s.push(t->left);
        }
    }

    //Morris
    vector<int> preorderTraversalMorris(TreeNode* root) {
        if (!root) return {};
        vector<int> res;
        TreeNode *cur = root, *pre = NULL;
        while ( cur!=NULL ) {
            if (!cur->left) {
                res.push_back(cur->val);
                cur = cur->right;
            }
            else {
                pre = cur->left;
                while (pre->right && pre->right != cur) pre = pre->right;
                if (!pre->right) {
                    pre->right = cur;
                    cur = cur->left;
                }
                else {
                    pre->right = NULL;
                    res.push_back(cur->val);
                    cur = cur->right;
                }
            }
        }
    }
};

// T144 Binary Tree Postorder Traversal
class SolutionT145 {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        if (!root) return {};
        vector<int> res;
        stack<TreeNode*> s{{root}};
        while (!s.empty()) {
            TreeNode *t = s.top(); s.pop();
            res.insert(res.begin(), t->val);
            if (t->left) s.push(t->left);
            if (t->right) s.push(t->right);
        }
        return res;
    }

    vector<int> postorderTraversal2(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> s;
        TreeNode* p = root;
        while (p || !s.empty()) {
            if (p) {
                s.push(p);
                res.insert(res.begin(), p->val);
                p = p->right;
            }
            else {
                p = s.top(); s.pop();
                p = p->left;
            }
        }
        return res;
    }

    vector<int> postorderTraversal3(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> s{{root}};
        TreeNode* head = root;
        while (!s.empty()) {
            TreeNode* t = s.top();
            if ((!t->left && !t->right) || t->left == head || t->right == head) {
                res.push_back(t->val);
                s.pop();
                head = t;
            }
            else {
                if (t->right) s.push(t->right);
                if (t->left) s.push(t->left);
            }
        }
        return res;
    }
};

// Unique Binary Search Trees
class SolutionT95 {
public:
    vector<TreeNode*> generateTrees(int n) {
        if (n == 0) return {};
        vector<vector<vector<TreeNode*>>> memo(n, vector<vector<TreeNode*>>(n));
        return helper(1, n, memo);
    }
    vector<TreeNode*> helper(int start, int end, vector<vector<vector<TreeNode*>>>& memo) {
        if (start > end) return {nullptr};
        if (!memo[start - 1][end - 1].empty()) return memo[start - 1][end - 1];
        vector<TreeNode*> res;
        for (int i = start; i <= end; ++i) {
            auto left = helper(start, i - 1, memo), right = helper(i + 1, end, memo);
            for (auto a : left) {
                for (auto b : right) {
                    TreeNode *node = new TreeNode(i);
                    node->left = a;
                    node->right = b;
                    res.push_back(node);
                }
            }
        }
        return memo[start - 1][end - 1] = res;
    }
};

// 105. 从前序与中序遍历序列构造二叉树
class SolutionT105 {
public:
    TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder) {
        return buildTree(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
    }
    TreeNode *buildTree(vector<int> &preorder, int pLeft, int pRight, vector<int> &inorder, int iLeft, int iRight) {
        if (pLeft > pRight || iLeft > iRight) return NULL;
        int i = 0;
        for (i = iLeft; i <= iRight; ++i) {
            if (preorder[pLeft] == inorder[i]) break;
        }
        TreeNode *cur = new TreeNode(preorder[pLeft]);
        cur->left = buildTree(preorder, pLeft+1, pLeft + i - iLeft, inorder, iLeft, i-1);
        cur->right = buildTree(preorder, pLeft + i - iLeft +1, pRight, inorder, i+1, iRight);
        return cur;
    }
};

// 106. Construct Binary Tree from Inorder and Postorder Traversal
class SolutionT106 {
public:
    TreeNode *buildTree(vector<int> &inorder, vector<int> &postorder) {
        return buildTree(inorder, 0, inorder.size() - 1, postorder, 0, postorder.size() - 1);
    }
    TreeNode *buildTree(vector<int> &inorder, int iLeft, int iRight, vector<int> &postorder, int pLeft, int pRight) {
        if (iLeft > iRight || pLeft > pRight) return NULL;
        TreeNode *cur = new TreeNode(postorder[pRight]);
        int i = 0;
        for (i = iLeft; i < inorder.size(); ++i) {
            if (inorder[i] == cur->val) break;
        }
        cur->left = buildTree(inorder, iLeft, i - 1, postorder, pLeft, pLeft + i - iLeft - 1);
        cur->right = buildTree(inorder, i + 1, iRight, postorder, pLeft + i - iLeft, pRight - 1);
        return cur;
    }
};

// 108. 将有序数组转换为二叉搜索树
class SolutionT108 {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        int mid = nums.size() / 2;
        TreeNode* root = new TreeNode(nums[mid]);
        vector<int> left(nums.begin(), nums.begin() + mid), right(nums.begin() + mid + 1, nums.end());
        root->left = sortedArrayToBST(left);
        root->right = sortedArrayToBST(right);
        return root;
    }
};

// 112. 路径总和
class SolutionT112 {
public:
    bool hasPathSum(TreeNode* root, int sum) {
        int accumulate = 0;
        return help(root, sum);
    }

    bool help(TreeNode* node, int sum) {
        if (sum == 0 && !node) return true;
        if (!node) return false;
        return help(node->left, sum - node->val) ||  help(node->right, sum - node->val); 
    }
};

// 113. 路径总和 II
class SolutionT113 {
public:
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        vector<vector<int>> res;
        vector<int> temp
        helper(root, sum, temp, res);
        return res;
    }

    void helper(TreeNode* root, int sum, vector<int> &temp, vector<vector<int>> res) {
        if (!root) return;
        temp.push_back(root->val);
        if (!root->left && !root->right && sum-root->val == 0) {
            res.push_back(temp);
            // 不要return; 因为后面有可能是 1 -1 
        }
        helper
        helper
        temp.pop_back();
    }
};

class SolutionT114 {
public:
    void flatten(TreeNode* root) {
        if (!root) return;
        if (root->left) flatten(root->left);
        if (root->right) flatten(root->right);
        TreeNode* temp = root->right;
        root->right = root->left;
        root->left = NULL;
        while(root->right) root = root->right;
        root->right = temp;
    }

    void flatten(TreeNode* root) {
        TreeNode* cur = root;
        while(cur) {
            if (cur->left) {
                TreeNode* p = cur->left;
                while (p->right) p = p->right;
                p->right = cur->right;
                cur->right = cur->left;
                cur->left = NULL;
            }
            cur = cur->right;
        }
    }

    void flatten(TreeNode* root) {
        if (!root) return;
        stack<TreeNode*> s;
        s.push(root);
        while (!s.empty()) {
            TreeNode *t = s.top(); s.pop();
            if (t->left) {
                TreeNode *r = t->left;
                while (r->right) r = r->right;
                r->right = t->right;
                t->right = t->left;
                t->left = NULL;
            }
            if (t->right) s.push(t->right);
        }
    }
};

// 填充每个节点的下一个右侧节点指针
class SolutionT116 {
public:
    Node* connect(Node* root) {
        if (!root) return NULL;
        queue<Node*> q;
        q.push(root);
        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                Node *t = q.front(); q.pop();
                if (i < size - 1) {
                    t->next = q.front();
                }
                if (t->left) q.push(t->left);
                if (t->right) q.push(t->right);
            }
        }
        return root;
    }

    Node* connect(Node* root) {
        if (!root) return nullptr;
        if (root->left) root->left->next = root->right;
        if (root->right) root->right->next = root->next?root->next->left:nullptr;
        connect(root->left);
        connect(root->right); 
        return root;
    }
};

// 填充每个节点的下一个右侧节点指针 II
class SolutionT117 {
public:
    Node* connect(Node* root) {
        if (!root) return nullptr;
        TreeNode* p = root->next;
        while(p) {
            if (p->left) {
                p = p->left;
                break;
            }
            if (p->right) {
                p = p->right;
                break;
            }
            p = p->next;
        }
        if (root->right) root->right->next = p;
        if (root->left) root->left->next = root->right?root->right:p;
        connect(root->right);
        connect(root->left);
        return root;
    }

    Node* connect(TreeNode* root) {

        // cur 和 dummy， cur指向dummy，cur->next指向，则dummy->next指向，但是cur移动，dummy不移动
        Node *dummy = new Node(0, NULL, NULL, NULL), *cur = dummy, *head = root;
        while (root) {
            if (root->left) {
                cur->next = root->left;
                cur = cur->next;
            }
            if (root->right) {
                cur->next = root->right;
                cur = cur->next;
            }
            root = root->next;
        }
        if (!root) {
            root = dummy->next;
            cur = dummy;
            dummy->next = nullptr;
        }
        return head;
    }
};

//124. 二叉树中的最大路径和 
class SolutionT124 {
public:
    int maxPathSum(TreeNode* root) {
        int res = INT_MIN;
        helper(root, res);
        return res;
    }

    int helper(TreeNode* root, int &res) {
        if (!root) return 0;
        int left = max(helper(root->left, res), 0);
        int right = max(helper(root->right, res), 0);
        res = max(res, root->val + left + right);
        return max(root->val + left, root->val + right);
    }
};

// 129. 求根到叶子节点数字之和
class SolutionT129 {
public:
    int sumNumbers(TreeNode* root) {
        int sum = 0;
        if (!root) return sum;
        helper(root, 0, sum);
        return sum;
    }

    void helper(TreeNode* node, int cur, int &sum) {
        cur = cur*10 + node->val;
        if (!node->left && !node->right) {
            sum += cur;
            return ;
        }
        if (node->left) helper(node->left, cur, sum);
        if (node->right) helper(node->left, cur, sum);
        return ;
    }

    int sumNumbers(TreeNode* root) {
        if (!root) return 0;
        int sum = 0;
        stack<TreeNode*> s{{root}};
        while (!s.empty()) {
            TreeNode* p = s.top(); s.pop();
            if (!p->left && !p->right) {
                sum += p->val;
            }
            if (p->left) {
                p->left->val += p->val * 10;
                s.push(p->left); 
            }
            if (p->right) {
                p->right->val += p->val * 10;
                s.push(p->right); 
            }
        }
        return sum;
    }
};

// 226. 翻转二叉树
class SolutionT226 {
public:
    TreeNode* invertTree(TreeNode* root) {

    }
};

// 222. 完全二叉树的节点个数
class SolutionT222 {
public:
    int countNodes(TreeNode* root) {

    }
};

// 230. 二叉搜索树中第K小的元素
class SolutionT230 {
public:
    int kthSmallest(TreeNode* root, int k) {
        TreeNode* p = root;
        stack<TreeNode*> s;
        int cnt = 0;
        while (p || !q.empty()) {
            while (p->left) {
                s.push(p);
                p = p->left;
            }
            p = s.top(); s.pop();
            cnt++;
            if (cnt == k) return p->val;
            p = p->right;
        }
    }

    int kthSmallest(TreeNode* root, int k) {
        return DFS(root, k);
    }

    int DFS(TreeNode* root, int &k) {
        if (!root) return -1;
        int cnt = DFS(root->left, k);
        if (k == 0) return cnt;
        // 到自己才开始减
        if (--k == 0) return root->val;     
        return DFS(root->right, k);

    }

    int kthSmallest(TreeNode* root, int k) {
        int cnt = count(root->left);
        if (k < cnt) {
            return kthSmallest(root->left, k);
        }
        else if (k > cnt + 1) {
            return kthSmallest(root->right, k - cnt - 1);
        }
        else {
            return root->val;
        }
    }

    int count(TreeNode* node) {
        if (!node) return 0;
        return 1 + count(node->left) + count(node->right);
    }

    struct MyTreeNode {
        int val;
        int count;
        MyTreeNode *left;
        MyTreeNode *right;
        MyTreeNode(int x) : val(x), count(1), left(NULL), right(NULL) {}
    };

    MyTreeNode* build(TreeNode* root) {
        if (!root) return NULL;
        MyTreeNode* p = new MyTreeNode(root->val);
        p->left = build(root->left);
        p->right = build(root->right);
        if (p->left) p->count += p->left->count;
        if (p->right) p->count += p->right->count;
        return node;
    }

    int kthSmallest(TreeNode* root, int k) {
        MyTreeNode* node = build(root);
        return helper(node, k);
    }
};
// 如果二叉搜索树经常被修改（插入/删除操作）并且你需要频繁地查找第 k 小的值，你将如何优化 kthSmallest 函数？

// 235. 二叉搜索树的最近公共祖先
class SolutionT235 {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root) return NULL;
        if (root->val > max(p->val, q->val)) 
            return lowestCommonAncestor(root->left, p, q);
        else if (root->val < min(p->val, q->val)) 
            return lowestCommonAncestor(root->right, p, q);
        else return root;
    }
};

// 236. 二叉树的最近公共祖先
class SolutionT236 {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root || p==root || q==root ) return root;
        TreeNode* left = lowestCommonAncestor(root->left, p, q);
        TreeNode* right = lowestCommonAncestor(root->right, p ,q);
        if (left && right) return root;
        return left?left:right;
    }
};

class Codec {
public:
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        ostringstream out;
        serialize(root, out);
        return out.str();
    }
    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        istringstream in(data);
        return deserialize(in);
    }
private:
    void serialize(TreeNode *root, ostringstream &out) {
        if (root) {
            out << root->val << ' ';
            serialize(root->left, out);
            serialize(root->right, out);
        } else {
            out << "# ";
        }
    }
    TreeNode* deserialize(istringstream &in) {
        string val;
        in >> val;
        if (val == "#") return nullptr;
        TreeNode* node = new TreeNode(stoi(val));
        node->left = deserialize(in);
        node->right = deserialize(in);
        return node;
    }
};

class CodecLevelOrder {
public:
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        istringstream out;
        queue<TreeNode*> q{{root}};
        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < q.size(); i++) {
                TreeNode* p = q.front();
                if (!p) out << "# ";
                else {
                    out << p->val << ' ';
                    q.push(p->left);
                    q.push(p->right);
                }
            }
        }
        return out.str();
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        if (data.empty()) return nullptr;
        istringstream in(data);
        string val;
        in >> val;
        TreeNode* res = new TreeNode(stoi(val)), *cur = res;
        queue<TreeNode*> q{{cur}};
        while (q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode* p = q.front(); q.pop();
                if (!(in>>val)) break;
                if (val != "#") {
                    cur = new TreeNode(stoi(val));
                    q.push(cur);
                    p->left = cur;
                }
            if (!(in >> val)) break;
            if (val != "#") {
                cur = new TreeNode(stoi(val));
                q.push(cur);
                t->right = cur;
            }
        }
        return res;
        }
    }
};

// 337. 打家劫舍 III
class SolutionT337 {
public:
    int rob(TreeNode* root) {
        unordered_map<TreeNode*, int> map;
        return dfs(root, map);
    }

    int dfs(TreeNode* root, unordered_map<TreeNode*, int> map;) {
        if (!root) return 0;
        if (map.count(root)) return map[root];
        int val = 0;
        if (root->left) {
            val = val + dfs(root->left->left, map) + dfs(root->left->right, map);
        }
        if (root->right) {
            val = val + dfs(root->right->left, map) + dfs(root->right->right, map);
        }
        val = max(root->val + val, dfs(root->left,map) + dfs(root->right,map))
        m[root] = val;
        return val;
    }
};


class Solution {
    int pathSum(TreeNode* root, int sum) {
        int res = 0;
        vector<TreeNode*> out;
        helper(root, sum, 0, out, res);
        return res;
    }
    void helper(TreeNode* node, int sum, int curSum, vector<TreeNode*>& out, int& res) {
        if (!node) return 0;
        int val = node->val;
        curSum += val;
        if (curSum + val == sum) res++;
        out.push_back(node);
        for (int i = 0; i < out.size() - 1; i++) {
            if (curSum - out[i]->val == sum) res++;
        }
        helper(node->left, sum, curSum, out, res);
        helper(node->right, sum, curSum, out, res);
        out.pop_back();
    }
}

class CodecT449 {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        ostringstream out;
        serial(root, out);
        return out.str();
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        istringstream in(data);
        return deserial(in);
    }
private:
    string serial(TreeNode* root, istringstream &out) {
        if (!root) {
            out << "# "; 
        }
        else {
            out << root->val << " ";
            serial(root->left, out);
            serial(root->right, out);
        }
    }

    TreeNode* deserial(istringstream &in) {
        string val;
        in >> val;
        if (val == "#") return nullptr;
        TreeNode* root = new TreeNode(stoi(val));
        root->left = deserial(in);
        root->right = deserial(in);
        return root;
    }
};

class CodecT449 {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {

    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {

    }
}

class SolutionT501 {
public:
    vector<int> findMode(TreeNode* root) {
        vector<int> res;
        queue<TreeNode*> q;
        TreeNode* p = root;
        unordered_map<int,int> map;
        while (p || q.empty()) {
            while (p) {
                q.push(p);
                p = p->left;
            }
            p = q.front(); q.pop();
            map[p->val]++;
            p = p->right;
        }

    }
};

class Solution {
public:
    vector<int> findFrequentTreeSum(TreeNode* root) {
        unordered_map<int, int> map;
        Traverse(root, map);
        //helper(root, map);
        int mx = 0;
        for (auto m:map) {
            if(m.second > mx) mx = m.second;
        }
        vector<int> res;
        for (auto m:map) {
            if (m.second == mx) {
                res.push_back(m.first);
            }
        }
        return res;
    }

    int Traverse(TreeNode* root, unordered_map<int, int> &map) {
        if (!root) return 0;
        root->val += Traverse(root->left) + Traverse(root->right);
        map[root->val]++;
        return root->val;
    }

    void helper(TreeNode* root, unordered_map<int, int> &map) {
        if (!root) return;
        map[root->val]++;
        helper(root->left, map);
        helper(root->right, map);
        return ;
    }

public:
    vector<int> findFrequentTreeSum(TreeNode* root) {
        Traverse(root);
        return res;
    }

    int Traverse(TreeNode* root) {
        if (!root) return 0;
        root->val += Traverse(root->left) + Traverse(root->right);
        int sum = root->val;
        map[sum]++;
        if (map[sum] >= cnt) {
            if (map[sum] > cnt) res.clear();
            res.push_back(sum);
            cnt = map[sum];
        }
        return root->val;
    }

private:
    unordered_map<int, int> map;
    int cnt = 0;
    vector<int> res;
};

class Solution {
public:
    int res = INT_MIN;
    int diameterOfBinaryTree(TreeNode* root) {
        helper(root);
        return res;
    }

    int helper(TreeNode* root) {
        if (!root) return 0;
        int left = helper(root->left);
        int right = helper(root->right);
        int temp = root->val + left + right;
        res = max(res, temp);
        return max(root->val, max(root->val + left, root->val + right));
    }

    int res = 0;
    int diameterOfBinaryTree(TreeNode* root) {
        if (!root) return 0;
        helper(root);
        return res - 1;
    }

    int helper(TreeNode* root) {
        if (!root) return 0;
        if (m.count(root)) return m[root];
        int left = helper(root->left);
        int right = helper(root->right);
        int temp = 1 + left + right;
        res = max(res, temp);
        return m[root] = max(1 + left, 1 + right);
    }

private:
    unordered_map<TreeNode*, int> m;

};

class SolutionT450 {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        if (!root) return NULL;
        if (root->val == key) {
            if (!root->right) return root->left;
            else {
                TreeNode *cur = root->right;
                while (cur->left) cur = cur->left;
                swap(root->val, cur->val);
            }
        }
        root->left = deleteNode(root->left, key);
        root->right = deleteNode(root->right, key);
        return root;
    }
};

class SolutionT513 {
public:
    int findBottomLeftValue(TreeNode* root) {
        queue<TreeNode*> q{{root}};
        int result = 0;
        while (!q.empty()) {
            int size = q.size();
            queue<int> res;
            for (int i = 0; i < size; i++) {
                TreeNode* temp = q.front();q.pop();
                res.push(temp->val);
                if (temp->left) q.push(temp->left);
                if (temp->right) q.push(temp->right);
            }
            if (q.empty()) {
                result = res.front();
                break;
            }
        }
        return result;
    }

    // 层序遍历从右往左
    int findBottomLeftValue(TreeNode* root) {
        queue<TreeNode*> q{{root}};
        while (!q.empty()) {
            root = q.front(); q.pop();
            if (root->right) q.push(root->right);
            if (root->left) q.push(root->left);
        }
        return root->val;
    }

    // 先序遍历
    int findBottomLeftValue(TreeNode* root) {
        int max_depth = 1;
        int res = root->val;
        helper(root, max_depth, res);
        return res;
    }

    void helper(TreeNode* root, int depth, int &res) {
        if (depth > max_depth) {
            max_depth = depth;
            res = root->val;
        }
        if (root->left) helper(root->left, depth+1, res);
        if (root->right) helper(root->right, depth+1, res);
    }
};

//前中生成树
class SolutionT105 {
public:
    TreeNode* node(vector<int>& vp,int p,vector<int>& vi,int i,int n){
    	if(n==0) return NULL;
    	if(n==1)
        	{   
				TreeNode *root = new TreeNode(vp[p]);
						root->left = NULL;
						root->right = NULL;
						return root;
			}
        TreeNode *root = new TreeNode(vp[p]);
			int j;
			for(j=0;j<n;j++){
				if(vi[i+j]==root->val)
					break;
			}
			int L,R;
			L = j;
			R = n-j-1;
			root->left = node(vp,p+1,vi,i,L);
			root->right = node(vp,p+L+1,vi,i+L+1,R);
			return root;
    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n = preorder.size();
		TreeNode* root;
		root = node(preorder,0,inorder,0,n);
		 return root;
    }
};

//后中生成树
class Solution {
public:
    TreeNode* node(vector<int>& vi,int i,vector<int>& vpo,int po,int n){
    	if(n==0) return NULL;
    	if(n==1)
        	{   
           TreeNode *root = new TreeNode(vpo[po+n-1]);
				root->left = NULL;
				root->right = NULL;
				return root;
			}
        TreeNode *root = new TreeNode(vpo[po+n-1]);
			int j;
			for(j=0;j<n;j++){
				if(vi[i+j]==root->val)
					break;
			}
			int L,R;
			L = j;
			R = n-j-1;
			root->left = node(vi,i,vpo,po,L);
			root->right = node(vi,i+L+1,vpo,po+L,R);
			return root;
    }
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        int n = inorder.size();
		TreeNode* root;
		root = node(inorder,0,postorder,0,n);
		 return root;
    }
};
