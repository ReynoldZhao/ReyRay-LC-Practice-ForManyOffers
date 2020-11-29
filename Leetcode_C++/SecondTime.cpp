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

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class SolutionT25 {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if (!l1 || !l2) return l1?l1:l2;
        ListNode* cur = new ListNode(0);
        ListNode* head = cur;
        while(l1 && l2) {
            if(l1->val < l2->val) {
                head->next = l1;
                l1 = l1->next;
            }
            else {
                head->next = l2;
                l2 = l2->next;
            }
            head = head->next;
        }
        head->next = l1?l1:l2;
        return cur->next;
    }

    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if (!l1 || !l2) return l1?l1:l2;
        if (l1->val < l2->val) {
            l1->next = mergeTwoLists(l1->next, l2);
            return l1;
        }
        else {

        }
    }
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class SolutionT26 {
public:
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if (A==nullptr && B==nullptr) return true;
        if (!A || !B) return false;
        if (A->val == B->val) return isSubStructure(A->left,B) || isSubStructure(A->right,B) || equal(A, B);
        else return isSubStructure(A->left,B) || isSubStructure(A->right,B);
    }

    bool equal(TreeNode* A, TreeNode* B){
        if (!B) return true;
        if (!A || A->val != B->val) return false;
        return equal(A->left, B->left) && equal(A->right, B->right);
    }
};

class SolutionT29 {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector <int> res;
        if(matrix.empty()) return res;
        int rl = 0, rh = matrix.size()-1; //记录待打印的矩阵上下边缘
        int cl = 0, ch = matrix[0].size()-1; //记录待打印的矩阵左右边缘
        while(1){
            for(int i=cl; i<=ch; i++) res.push_back(matrix[rl][i]);//从左往右
            if(++rl > rh) break; //若超出边界，退出
            for(int i=rl; i<=rh; i++) res.push_back(matrix[i][ch]);//从上往下
            if(--ch < cl) break;
            for(int i=ch; i>=cl; i--) res.push_back(matrix[rh][i]);//从右往左
            if(--rh < rl) break;
            for(int i=rh; i>=rl; i--) res.push_back(matrix[i][cl]);//从下往上
            if(++cl > ch) break;
        }
        return res;
    }
};

class MinStackT30 {
public:
    /** initialize your data structure here. */
    MinStack() {

    }
    
    void push(int x) {
        s.push(x);
        if (minVal.empty()) {
            minVal.push(x);
        }
        else if (x < minVal.top()) {
            while (!minVal.empty()) {
                minVal.pop();
            }
            minVal.push(x);
        }
        else if (x == minVal.top()) {
            minVal.push(x);
        }
    }
    
    void pop() {
        int temp = s.top();
        s.pop();
        if (temp == minVal.top()) {
            minVal.pop();
        }
    }
    
    int top() {
        return s.top();
    }
    
    int min() {
        return minVal.top();
    }

private:
    stack<int> s;
    stack<int> minVal;
};

class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        stack<int> s;
        int j = 0;
        for (int i = 0; i < pushed.size(); i++) {
            s.push(pushed[i]);
            while(s.top() == popped[j]) {
                s.pop();
                j++;
            }
        }
        if (!s.empty() || j != popped.size()) return false;
        return true;
    }
};

class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        int flag = false;
        queue<TreeNode*> q{root};
        vector<vector<int>> res;
        while(!q.empty()) {
            vector<int> out;
            int size = q.size();
            if(!flag) {
                for (int i = 0; i < size; i++) {
                    TreeNode* temp = q.front();
                    if(temp->right) q.push(temp->right);
                    if(temp->left) q.push(temp->left);
                    out.push_back(temp->val);
                }
            }
            else{
                for (int i = 0; i < size; i++) {
                    TreeNode* temp = q.front();
                    if(temp->left) q.push(temp->left);
                    if(temp->right) q.push(temp->right);
                    out.push_back(temp->val);
                }
            }
            res.push_back(out);
            flag = !flag;
        }
        return res;
    }
};


class Solution {
public:
    bool verifyPostorder(vector<int>& postorder) {
        return helper(postorder, 0, postorder.size() - 1);
    }

    bool helper(vector<int>& post, int i, int j) {
        if (i >= j) return true;
        int root = post[j];
        int k = i;
        while(post[k] < root) k++;
        int m = k;
        while(post[k] > root) k++;
        return (k==j) && helper(post, i, k-1) && helper(post, k, j-1);
    }
};

class Solution {
public:
    vector<int> out;
    vector<vector<int>> res;
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        return helper
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

class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(!head) return nullptr;
        unordered_map<Node*, Node*> map;
        Node* temp = head;
        while(temp) {
            Node* newnode = new Node(temp->val);
            map[temp] = newnode;
            temp = temp->next;
        }
        temp = head;
        while(temp) {
            map[temp]->next = map[temp->next];
            map[temp]->random = map[temp->random];
            temp = temp->next;
        }
        return map[head];
    }
};

class Solution {
public:
    Node* head = NULL, pre = NULL;
    Node* treeToDoublyList(Node* root) {
        if (!root) return NULL;
        dfs(root)
        head->left = pre;
        pre->right = head;
        return head;
    }

    void dfs(Node* cur) {
        if (!cur) return ;
        dfs(cur->left);
        if (pre){
            cur->left = pre;
            pre->right = cur;
        }
        else head = cur;
        pre = cur;
        dfs(cur->right);
        return ;
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
    void serialize(TreeNode* root, ostringstream out) {
        if(root) {
            out << root->val << ' ';
            serialize(root->left, out);
            serialize(root->right, out);
        }
        else {
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

class Solution {
public:
    int majorityElement(vector<int>& nums) {
        // sort(nums.begin(), nums.end());
        int count = 1;
        int cur = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] != cur) {
                count--;
                if (count == 0) {
                    cur = nums[i];
                }
            }
            else count ++;
        }
        return cur;
    }

    // 摩尔投票法
    int majorityElement(vector<int>& nums) {
        int x = 0, votes = 0;
        for (auto num:nums) {
            if(votes == 0) x = num;
            votes += (num == x)?1:-1;
        }
        return x;
    }
};

class Solution {
public:

    int randomized_partition(vector<int>& arr, int l, int r){
        int i = rand() % (r - l + 1) + l;
        swap(nums[r], nums[i]);
        return partition(nums, l, r);
    };

    void random_selected(vector<int>& arr, int l, int r, int k) {
        if (l >= r) return ;
        int pos = randomized_partition(arr, l, r);
        int num = pos - l + 1;
        if (num == k) {
            return ;
        }
        else if (num < k) {
            random_selected(arr, pos + 1, r, k - num);
        }
        else {
            random_selected(arr, l, pos, k);
        }

    }
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        vector<int> res;
        if (k==0 || arr.size() == 0) return res;
        random_selected(arr, 0, arr.size()-1, k);
        for (int i = 0; i < k; ++i) {
            res.push_back(arr[i]);
        }
        return res;
    }
};

class SolutionT50 {
public:
    char firstUniqChar(string s) {

    }
};

class SolutionT49 {
public:
    int nthUglyNumber(int n) {
        vector<int> dp(n, 0);
        dp[0] = 1;
        int a = 0, b = 0, c = 0;
        for (int i = 0; i < n; i++) {
            int n2 = dp[a] * 2, n3 = dp[b] * 3, n5 = dp[c] * 5;
            dp[i] = min(min(n2, n3), n5);
            if(dp[i] == n2) a++;
            if(dp[i] == n3) b++;
            if(dp[i] == n5) c++;
        }
        return dp[n - 1];
    }
}

class SolutionT48 {
public:
    int lengthOfLongestSubstring(string s) {
        int maxLen = 0;
        unordered_map<int, int> m;
        int start = -1;
        for (int i = 0; i < s.size(); i++) {
            if (m.count(s[i]) && m[s[i]] > start) {
                start = m[s[i]];
            }
            m[s[i]] = i;
            maxLen = max(maxLen, i - start);
        }
        return maxLen;
    }

    int lengthOfLongestSubstring(string s) {
        int maxLen = 0;
        vector<int> v(128, -1);
        int start = -1;
        for (int i = 0; i < s.size(); i++) {
            left = max(left, v[s[i]]);
            v[s[i]] = i;
            maxLen = max(maxLen, i - left);

        }
        return maxLen;
    }
};

class SolutionT47 {
public:
    int maxValue(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        vector<int> dp(n,0);
        for (int i = 0; i < n ; i++) {
            dp[i] = (i == 0)? grid[0][0]:dp[i] + grid[0][i];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j==0) dp[j] = grid[i][j] + dp[j];
                else {
                    dp[j] = max(dp[j-1], dp[j]) + grid[i][j];
                }
            }
        }
        return dp[n-1];
    }
};

class Solution {
public:
    int res = 0;
    int translateNum(int num) {
        string src = to_string(num);
        return dfs(0, src);
    }

    int dfs(int index, string src) {
        int n = src.size();
        if (index >= src.size()) {
            res++;
            return 1;
        }
        if (index == n-1 || src.substr(index, 2) < "10" || src.substr(index, 2) > "25") {
            return dfs(index+1, src);
        }
        return dfs(index+1, src) + dfs(index+2, src);
    }

    int translateNum(int num) {
        string src = to_string(num);
        int a = 1, b = 1;
        int r = 0;
        for (int i = 2; i <= src.size(); i++) {
            string temp = src.substr(i-2, 2);
            if (temp >= "10" && temp <= "25") {
                r = a + b;
            }
            else {
                r = b;
            }
            a = b;
            b = r;
        }
        return r;
    }

    int translateNum(int num) {
        int a = 1, b = 1, x, y = num % 10;
        while (num != 0) {
            num = num/10;
            x = num % 10;
            int temp = 10*x + y;
            int c = (tmp >= 10 && tmp <= 25) ? a + b : a;
            b = a;
            a = c;
            y = x;
        }
        return a;
};

class Solution {
public:
    string minNumber(vector<int>& nums) {
        sort(nums.begin(), nums.end(),[](int a, int b){ return to_string(a) + to_string(b) < to_string(b) + to_string(a)})
        string res = "";
        for (int i = 0; i < nums.size(); i++) {
            res += to_string(nums[i]);
        }
        return res;
    }
};

class Solution {
public:
    int findNthDigit(int n) {
        long long len = 1, cnt = 9, start = 1;
        while (n > len * cnt) {
            n -= len * cnt;
            ++len;
            cnt *= 10;
            start *= 10;
        }
        start += (n-1)/len;
        string t = to_string(start);
        return t[(n-1) % len] - '0';
    }
};

class SolutionT43 {
public:
    int countDigitOne(int n) {

    }
};

class MedianFinder {
public:
    /** initialize your data structure here. */
    priority_queue< long > small, large;
    MedianFinder() {

    }
    
    void addNum(int num) {
        small.push(num);
        large.push(-small.top());
        small.pop();
        if (small.size() < large.size()) {
            small.push(-large.top());
            large.pop();
        }
    }
    
    double findMedian() {
        return small.size() > large.size() ? small.top() : 0.5 *(small.top() - large.top());
    }
};

class SolutionT55 {
public:
    bool isBalanced(TreeNode* root) {
        if (!root) return true;
        int leftD = getDepth(root->left);
        int rightD = getDepth(root->right);
        if (leftD == -1 || rightD == -1 || abs(leftD - rightD) >= 2) return false;
        return isBalanced(root->left) && isBalanced(root->right);
    }

    int getDepth(TreeNode* root) {
        if (!root) return 0;
        int leftD = getDepth(root->left);
        int rightD = getDepth(root->right);
        if (leftD == -1 || rightD == -1 || abs(rightD - leftD) >= 2) return -1;
        return 1 + max(leftD, rightD);
    }

};

class Solution {
public:
    int res, k;
    int kthLargest(TreeNode* root, int k) {
        this->res = 0, this->k = k;
        dfs(root);
        return res;
    }
    
    void dfs(TreeNode* root) {
        if (!root) return ;
        dfs(root->right);
        if (k==0) return ;
        if (--k == 0) res = root->val;
        dfs(root->left);
        return 
    }
};

class SolutionT53 {
public:
    int missingNumber(vector<int>& nums) {
        int i = 0, j = nums.size();
        while (i < j) {
            int m = i + (j - i)/2;
            if (nums[m] == m) i = m + 1;
            else j = m - 1;
        }
        return i;
    }
};

class SolutionT5100 {
public:
    int search(vector<int>& nums, int target) {
        return helper(nums, target) - helper(nums, target - 1);
    }

    int helper(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] <= target) l = mid + 1;
            else r = mid - 1;
        }
        return l;
    }
};

class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int count[32] = {0};
        for (int i = 0; i < nums.size(); i++) {
            for (int j = 0; j < 32; j++) {
                if (nums[i] & 1) count[j]++;
                nums[i] >>>= 1;
            }
        }
        int res = 0, m = 3;
        for (int i = 0; i < 32; i++) {
            res <<= 1;
            if (count[31-i] % m !=0) {
                res |= 1;
            }
        }
        return res;
    }
};

class SolutionT58 {
public:

    int mergesort(vector<int>& nums, int l, int r, vector<int>& temp) {
        if (l >= r) {
            temp[l] = nums[l];
            return 0;
        }
        int mid = l + (r - l) / 2;
        int lreverse = mergesort(nums, l, mid, temp);
        int rreverse = mergesort(nums, mid+1, r, temp);
        int lptr = mid, rptr = r;
        int res = 0;
        int index = rptr;
        while (lptr >= l && rptr >= mid + 1)
        {
            if (nums[lptr] > nums[rptr]) {
                temp[index--] = nums[lptr--];
                res += rptr - mid;
            }
            else {
                temp[index--] = nums[rptr--];
            }
        }
        for (;lptr >= l; lptr--) {
            temp[index--] = nums[lptr];
        }
        for (;rptr >= mid + 1; rptr--) {
            temp[index--] = nums[rptr];
        }
        return res + lreverse + rreverse;
    }

    int reversePairs(vector<int>& nums) {
        int n = nums.size();
        vector<int> temp(n, 0);
        return mergesort(nums, 0, n - 1, temp);
    }
};

class SolutionT57 {
public:
    vector<vector<int>> findContinuousSequence(int target) {
        vector<vector<int>> res;
        int i = 1, j = 1, sum = 0;
        while (j <= target/2 + 1) {
            if (sum < target) j++;
            else if (sum > target) i++;
            else if (sum == target) {
                vector<int> temp;
                for (int k = i; k <= j; k++) {
                    temp.push_back(k);
                }
                res.push_back(temp);
                sum-=i;
                i++;
            }
        }
        return res;
    }
};

class SolutionT59 {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;
        if(nums.size() <= 0 || nums.size() < k || k <= 0) return res;
        deque<int> dq;
        for (int i = 0; i < k; i++) {
            while (!dq.empty() && nums[i] >= nums[dq.back()]) dq.pop_back();
            dq.push_back(i);
        }
        res.push_back(nums[dq.front()]);
        for (int i = k; i < nums.size() - k; i++) {
            if (dq.front() < i - k) dq.pop_front();
            while (!dq.empty() && nums[i] >= nums[dq.back()]) dq.pop_back();
            dq.push_back(i);
            res.push_back(nums[dq.front()]);
        }
        return res;
    }
};

class SolutionT62 {
public:
    int lastRemaining(int n, int m) {
        vector<int> vec(n,0);
        for (int i = 0; i < n; i ++) {
            vec[i] = i;
        }
        int count = n;
        int base = -1;
        int step = 0;
        while (count > 0) {
            base++;
            if (base >= n) base = 0;
            if (vec[base] == -1) continue;
            step++;
            if (step = m) {
                vec[base] = -1;
                count--;
                step = 0;
            }
        }
        return i;
    }
};

class SolutionT60 {
public:
    vector<double> twoSum(int n) {
        int dp[15][70]
        memset(dp, 0, sizeof(dp));
        for (int i = 1; i <= 6; i++) {
            dp[1][i] = 1;
        }
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <=6 * i; j++) {
                for (int cur = 1; cur <= 6; cur ++) {
                    if (j - cur <=0) break;
                    dp[i][j] += dp[i-1][j - cur];
                }
            }
        }
        int all = pow(6, n);
        vector<double> res;
        for (int i = n; i<= 6 * n; i++) {
            res.push_back( dp[n][i] * 1.0 / all);
        }
        return res;
    }
};

class MaxQueue {
public:
    queue<int> q;
    queue<int> maxq;
    MaxQueue() {

    }
    
    int max_value() {
        if (q.empty()) return -1;
    }
    
    void push_back(int value) {

    }
    
    int pop_front() {

    }
};

class SolutionT63 {
public:
    int maxProfit(vector<int>& prices) {
        int maxProfit = 0;
        int start = prices[0];
        for (int i = 0; i < prices.size() - 1; i++) {
            if (prices[i] > start)
        }
    }
};

class SolutionT605 {
public:
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
        int i = 0, count = 0;
        while (i < flowerbed.size()) {
            if (flowerbed[i] == 0 && (i == 0 || flowerbed[i+1] == 0) && (i == flowerbed.size() - 1 || flowerbed[i+1] == 0) {
                flowerbed[i] = 1;
                count ++;
            }
        }
        return count >= n;
    }
};

class SolutionT452 {
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        sort(points.begin(), points.end(),[](vector<int> a, vector<int> b) {
            return a[1] > b[1];
        });
        int n = points.size() - 1;
        int pend = points[n][1], count = 1, pstart = points[n][0];
        for (int i = n - 1; i >= 0; i--) {
            if (points[i][1] < pstart) {
                pstart = points[i][0];
                count++;
            }
            else if (points[i][1] >= pstart) {
                pstart = points[i][0];
            }
        }
        return count;
    }
};

class SolutionT763 {
public:
    vector<int> partitionLabels(string S) {
        //char set[26] = {0};
        char lastpos[26] = {0};
        vector<int> res;
        int n = S.size() - 1;
        for (int i = 0; i <= n; i++) {
            //set[S[i] - 'a']++;
            lastpos[S[i] - 'a'] = i;
        }
        int LastPos = 0, pre = -1;
        for (int i = 0; i <= n; i++) {
            LastPos = max(LastPos, lastpos[S[i] - 'a']);
            if (LastPos == i) {
                res.push_back(i - pre);
                pre = i;
            }
        }
        return res;
    }   
};

class SolutionT122 {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.empty()) return 0;
        int in = prices[0], sum = 0;
        for (int i = 1; i < prices.size(); i++) {
            if (prices[i] > in){
                sum += prices[i] - in;
            }
            in = prices[i];
        }
        return sum;
    }
};

class SolutionT406 {
public:
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        sort(people.begin(), people.end(), [](vector<int>& a, vector<int>& b) {
            return a[0] > b[0] || (a[0] == b[0] && a[1] < b[1]);
        });
        int cnt = 0;
        for (int i = 1; i < people.size(); i++) {
            for (int j = 0; j < i; j++) {
                if (cnt == people[i][1]) {
                    auto temp = people[i];
                    for (int k = i-1; k >= j; k--){
                        people[k+1] = people[k];
                    }
                    people[j] = temp;
                }
                cnt++;
            }
        }
    }
};

class SolutionT665 {
public:
    bool checkPossibility(vector<int>& nums) {
        int cnt = 0, pre = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] < nums[i - 1]){
                cnt++;
                pre = max(pre, nums[i-1]);
            }
        }
        return cnt < 2;
    }
};

class SolutionT633 {
public:
    bool judgeSquareSum(int c) {
        long i = 0, j = sqrt(c);
        while (i <= j) 
            if (i * i + j * j == c) return true;
            else if (i * i + j * j > c) j --;
            else i ++;
        return false;
    }

    bool judgeSquareSum(int c) {
        for (int a = 0; a*a <= c; a++) {
            int b = c - a*a;
            if(binary_search(0, b, b)) return true;
        }
        return false;
    }

    bool binary_search(long s, long e, long n) {
        long mid = (s + e) / 2;
        long sum = mid * mid;
        if (sum == n) return true;
        else if (sum < n) return binary_search(mid + 1, e, n);
        else binary_search(s, mid - 1, n);
    }
};

class SolutionT680 {
public:
    bool validPalindrome(string s) {
        int l = 0, r = s.size() - 1;
        int count = 0;
        while (l <= r) {
            if(s[l] == s[r]) {
                l++;
                r--;
            }
            else return isPalindrome(s, l+1, r) || isPalindrome(s, l, r-1);
        }
    }

    bool isPalindrome(string s, int l, int r) {
        while (l <= r) {
            if(s[l] == s[r]) {
                l++;
                r--;
            }
            else return false;
        }
        return true;
    }
};

class SolutionT524 {
public:
    string findLongestWord(string s, vector<string>& d) {
        sort(d.begin(), d.end(), [](string a, string b){
            return a.length() != b.length() ? a.size() - b.size():a > b;
        });

    }
};

class SolutionT159 {
public:
    int lengthOfLongestSubstringTwoDistinct(string s) {
        int res = 0, left = 0;
        unordered_map<char, int> m;
        for (int i = 0; i < s.size(); ++i) {
            ++m[s[i]];
            while (m.size() > 2) {
                if (--m[s[left]] == 0) m.erase(s[left]);
                ++left;
            }
            res = max(res, i - left + 1);
        }
        return res;
    }
};

class SolutionT340 {
public:
    int lengthOfLongestSubstringKDistinct(string s, int k) {
        int start = 0, end = 0;
        int res = -1;
        unordered_map<char, int> map;
        while (end < s.size()) {
            if (map.count(s[end]) == 0) {
                ++map[s[end]]
            }
            while (map.size() > k) {
                if (--map[s[end]] == 0) map.erase(s[end]);
                start++;
            }
            res = max(res, end - start + 1);
            end++;
        }
        return res;
    }

    int lengthOfLongestSubstringKDistinct(string s, int k) {
        int start = 0, end = 0;
        int res = -1;
        unordered_map<char, int> map;
        while (end < s.size()) {
            map[s[end]] = i
            while (map.size() > k) {
                if (--map[s[end]] == 0) map.erase(s[end]);
                start++;
            }
            res = max(res, end - start + 1);
            end++;
        }
        return res;
    }
};

class SolutionT154 {
public:
    int findMin(vector<int>& nums) {
        int l = 0, r = nums.size() - 1;
        int mid = (r - l)/2 + l;
        while (l < r) {
            if (nums[mid] > nums[r]) {
                l = mid + 1;
            }
            else if (nums[mid] < nums[r]) {
                r = mid;
            }
            else if (nums[mid] == nums[r]) {
                r--;
            }
        }
        return nums[l];
    }

    int findMin(vector<int>& nums) {
        return helper(nums, 0, nums.size()-1);
    }

    int helper(vector<int>& nums, int start, int end) {
        if (nums[start] < nums[end]) return nums[start];
        int mid = (end - start)/2 + start;
        return min(helper(nums, start, mid), helper(nums, mid+1, end));
    }
};

class SolutionT540 {
public:
    int singleNonDuplicate(vector<int>& nums) {
        int l = 0, r = nums.size() - 1, n = nums.size() - 1;
        while (l < r){
            int mid = (r - l)/2 + l;
            if (nums[mid] == nums[mid+1]) {
                if(mid%2 == 0) l = mid+1;
                else r = mid;
            }
            else {
                if (mid == 0 || nums[mid] != nums[mid - 1]) return nums[mid];
                if((mid+1)%2 == 0) l = mid+1;
                else r = mid;
            }
        }
        return nums[l];
    }

    int singleNonDuplicate(vector<int>& nums) {
        int l = 0, r = nums.size() - 1, n = nums.size() - 1;
        while (l < r){
            int mid = (r - l)/2 + l;
            if (nums[mid] == nums[mid^1]) l = mid+1;
            else
            {
                r = mid;
            }
        }
        return nums[l];
    }
};

class SolutionT451 {
public:
    string frequencySort(string s) {
        unordered_map<char,int> map;
        int maxVal = 0;
        for (int i = 0; i < s.size(); i++) {
            map[s[i]]++;
            maxVal = max(maxVal, map[s[i]]);
        }
        vector<vector<char> > bucket(maxVal + 1);
        for (auto p:map){
            bucket[p.second].push_back(p.first);
        }
        string res = "";
        for (int i = maxVal; i > 0; i--) {
            while (!bucket[i].empty())
            {
                int count = i;
                char temp = bucket[i].back();
                while(count--) res += temp;
                bucket[i].pop_back();
            }
        }
        return res;
    }

    string frequencySort(string s) {
        int counts[256] = {0};
        for (char ch:s) {
            counts[ch]++:
        }
        sort(s.begin(), s.end(), [](char a, char b){
            return counts[a] > counts[b] || (counts[a] == counts[b] && a < b);
        })
        return s;

    }
};

class SolutionT75 {
public:
    void sortColors(vector<int>& nums) {
        int res = 0, blue = nums.size() - 1;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] == 0) {
                swap(nums[red++], nums[i]);
            }
            else if (nums[i] == 2) {
                swap(nums[blue--], nums[i]);
            }
        }
    }
};

class SolutionT127 {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> set(wordList.begin(), wordList.end());
        unordered_map<string, int> map{{beginWord, 1}};
        if (set.count(endWord) == 0) return 0;
        queue<string> q{{beginWord}};
        int res = 0;
        while(!q.empty()){
            for (int k = q.size(); k > 0; k--) {
                string temp = q.front();
                q.pop();
                for (int i = 0; i < temp.size(); i++) {
                    string newtemp = temp;
                    for (char ch = 'a'; ch <= 'z'; ch++) {
                        newtemp[i] = ch;
                        if (set.count(newtemp) && newtemp != endWord) {
                            q.push(newtemp);
                            set.erase(newtemp);
                        }
                        else if (newtemp == endWord) {
                            return res + 1;
                        }
                    }
                }
            }
            res++;
        }
        return 0;
    }
};

class SolutionT51 {
public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> res;
        if (n == 0) return res;
        vector<string> board(n, string(n, '.'));
        vector<bool> col(n, false);
        vector<bool> ldiag(2*n-1, false);
        vector<bool> rdiag(2*n-1, false);
        backtracking(res, board, col, ldiag, rdiag, 0, n);
        return res;
    }

    void backtracking(vector<vector<string>> &res, vector<string> &board, vector<bool> &col, vector<bool> &ldiag, vector<bool> &rdiag, int row, int n){
        if (row == n) {
            res.push_back(board);
            return ;
        }
        for (int i = 0; i < n; i++) {
            if (col[i] || rdiag[row + i + 1] || ldiag[n - row + i - 1]) continue;
            board[row][i] = 'Q';
            col[i] = true;
            rdiag[row+i+1] = true;
            ldiag[n - row + i - 1] = true;
            backtracking(res, board, col, ldiag, rdiag, row + 1, n);
            board[row][i] = '.';
            col[i] = rdiag[row+i+1] = ldiag[n - row + i - 1] = false;
        }
    }

    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> res;
        vector<int> queenCol(n, -1);
        helper(0, queenCol, res);
        return res;
    }

    void helper(int curRow, vector<int>& queenCol, vector<vector<string>>& res) {
        int n = queenCol.size();
        if (curRow == n) {
            vector<string> out(n, string(n, '.'));
            for (int i = 0; i < n; ++i) {
                out[i][queenCol[i]] = 'Q';
            }
            res.push_back(out);
            return;
        }
        for (int i = 0; i < n; i++) {
            if (isValid(queenCol, curRow, i)) (
                queenCol[curRow] = i;
                helper();
                queenCol[curRow] = -1;
            )
        }
    }

    bool isValid(vector<int>& queenCol, int row, int col) {
        for (int i = 0; i < row; i++) {
            if (queenCol[i] == col || abs(row - i) == abs(col - queenCol[i])) return false;
        }
    }
};

class SolutionT126 {
public:
    vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> dict(wordList.begin(), wordList.end());
        unordered_set<string> set;
        vector<string> cur({beginWord});
        queue<vector<string>> paths({cur});
        vector<vector<string>> res;
        int level = 1, minlevel = INT_MAX;
        while (!paths.empty()){
            auto temp = paths.front(); paths.pop();
            if (temp.size() > level) {
                for (auto s:set) dict.erase(s);
                set.clear();
                level = temp.size();
                if (level > minlevel) break;
            }
            string lastword = temp.back();
            for (int i = 0; i < lastword.size(); i++) {
                    string newWord = lastword;
                    for (char ch = 'a'; ch <= 'z'; ++ch) {
                        newWord[i] = ch;
                        if (!dict.count(newWord)) continue;
                        set.insert(newWord);
                        vector<string> newPath = temp;
                        newPath.push_back(newWord);
                        if (newWord == endWord) {
                            minlevel = level;
                            res.push_back(newPath);
                        } else {
                            paths.push(newPath);
                        }
                    }
            }
        }
        return res;
    }
};

class SolutionT130 {
public:
    int directions[5] = {-1, 0, 1, 0, -1};
    void solve(vector<vector<char>>& board) {
        int m = board.size() - 1;
        int n = board[0].size() - 1;
        for (int i = 0; i <= n; i++) {
            if (board[0][i] == 'O') dfs(board, 0, i);
            if (board[m][i] == 'O') dfs(board, m, i);
        }
        for (int j = 1; j <= m - 1; j++) {
            if (board[j][0] == 'O') dfs(board, j, 0);
            if (board[j][n] == 'O') dfs(board, j, n);
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O') board[i][j] = 'X';
                else if (board[i][j] == '#') board[i][j] = 'O';
            }
        }
    }

    void dfs(vector<vector<char>>& board, int i, int j) {
        if (i < 0 || i >= board.size() || j < 0 || j >= board[0].size() || board[i][j] != 'O') return ;
        board[i][j] = '#';
        for (int k = 0; k < 4; k++) {
            int row = i + directions[k];
            int col = j + directions[k + 1];
            dfs(board, row, col);
        }
    }
};

class SolutionT257 {
public:
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> res;
        dfs(root, res, "");
        return res;
    }

    void dfs(TreeNode* root, vector<string> res, string s) {
        if (!root) return ;
        s = s + to_string(root->val);
        if (!root->left && !root->left) {
            res.push_back(s);
            return;
        }
        s = s + "->";
        if (root->left) dfs(root->left, res, s);
        if (root->right) dfs(root->right, res, s);
        return ;
    }
};

class SolutionT46 {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        backtracking(res, nums, 0);
        return res;
    }

    void backtracking(vector<vector<int>> &res, vector<int>& nums, int k) {
        if (k == nums.size()) {
            res.push_back(nums);
            return ;
        }
        for (int i = k; i < nums.size(); i++) {
            swap(nums[k], nums[i]);
            backtracking(res, nums, k+1);
            swap(nums[k], nums[i]);
        }
        return ;
    }
};

class SolutionT77 {
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> res;
        vector<int> temp;
        backtracking(res, temp, n, k, 1);
        return res;
    }

    void backtracking(vector<vector<int>> &res, vector<int> &temp, int n, int k, int index){
        if (temp.size() == k) {
            res.push_back(temp);
            return ;
        }
        for (int i = index; i <= n; i++) {
            temp.push_back(i);
            backtracking(res, temp, n, k, i);
            temp.pop_back();
        }
        return ;
    }
};

class SolutionT47 {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        vector<int> out, visited(nums.size(), 0);
        backtracking(res, out, nums, 0, visited);
        return res;
    }

    void backtracking(vector<vector<int>> &res, vector<int>& out, vector<int>& nums, int k, vector<int> &visited) {
        if (k == out.size()) {
            res.push_back(out);
            return ;
        }
        for (int i = 0; i < nums.size(); i++) {
            if(visited[i] == 1 || (i > 0 && nums[i] == nums[i-1] && visited[i-1] == 0)) continue;
            visited[i] = 1;
            out.push_back(nums[i]);
            backtracking(res, out, nums, k + 1, visited);
            out.pop_back();
            visited[i] = 0;
        }
        return ;
    }
};

class SolutionT39 {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> res;
        vector<int> temp;
        int sum = 0;
        helper(candidates, res, temp, 0, sum, target);
        return res;
    }

    void helper(vector<int>& candidates, vector<vector<int>> &res, vector<int> &temp, int idx, int &sum, int target){
        if (sum > target) return ;
        if (sum == target) {
            res.push_back(temp);
            return ;
        }
        for (int i = idx; i < candidates.size(); i++) {
            sum+=candidates[i];
            temp.push_back(candidates[i]);
            helper(candidates, res, temp, i, sum, target);
            temp.pop_back();
            sum-=candidates[i];
        }
    }
};

class SolutionT40 {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<int> visited(candidates.size(), 0);
        // vector<vector<int>> res;
        set<vector<int>> res;
        vector<int> temp;
        int sum = 0;
        helper(candidates, res, visited, temp, 0, sum, target);
        return vector<vector<int>> (res.begin(), res.end());
    }

    void helper(vector<int>& candidates, set<vector<int>> &res, vector<int> &visited, vector<int> &temp, int idx, int &sum, int target){
        if (sum == target) {
            res.insert(temp);
            return ;
        }
        for (int i = idx; i < candidates.size(); i++) {
            if (visited[i] == 1) continue;
            visited[i] = 1;
            sum+=candidates[i];
            temp.push_back(candidates[i]);
            helper(candidates, res, visited, temp, i+1, sum, target);
            temp.pop_back();
            sum-=candidates[i];
            visited[i] = 0;
        }
    }

    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> res;
        vector<int> temp;
        int sum = 0;
        helper(candidates, res, temp, 0, sum, target);
        return res;
    }

    void helper(vector<int>& candidates, vector<vector<int>> &res, vector<int> &temp, int idx, int &sum, int target){
        if (sum > target) return ;
        if (sum == target) {
            res.push_back(temp);
            return ;
        }
        for (int i = idx; i < candidates.size(); i++) {
            if (i > idx && candidates[i] == candidates[i-1]) continue;
            sum+=candidates[i];
            temp.push_back(candidates[i]);
            helper(candidates, res, temp, i+1, sum, target);
            temp.pop_back();
            sum-=candidates[i];
        }
    }
};

class SolutionT37 {
public:
    void solveSudoku(vector<vector<char>>& board) {
        helper(board);
    }
    bool helper(vector<vector<char>>& board) {
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (board[i][j] != '.') continue;
                for (char c = '1'; c <= '9'; ++c) {
                    if (!isValid(board, i, j, c)) continue;
                    board[i][j] = c;
                    if (helper(board)) return true;
                    board[i][j] = '.';
                }
                return false;
            }
        }
        return true;
    }
    bool isValid(vector<vector<char>>& board, int i, int j, char val) {
        for (int k = 0; k < 9; ++k) {
            if (board[k][j] != '.' && board[k][j] == val) return false;
            if (board[i][k] != '.' && board[i][k] == val) return false;
            int row = i / 3 * 3 + k / 3, col = j / 3 * 3 + k % 3;
            if (board[row][col] != '.' && board[row][col] == val) return false;
        }
        return true;
    }


    void solveSudoku(vector<vector<char>>& board) {
        int n = board.size();
        vector<vector<int>> row(n, vector<int> (n,0));
        vector<vector<int>> col(n, vector<int> (n,0));
        vector<vector<int>> block(n, vector<int> (n,0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == '.') continue;
                int num = board[i][j] - '0';
                row[i][num] = 1;
                col[num][j] = 1;
                block[num][((j+1)%3)*3 + (i+1)%3] = 1;
            }
        }
    }
};


class SolutionT310 {
public:
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        sort(edges.begin(), edges.end(), [](vector<int> a, vector<int> b){
            return a[0] != b[0] ? a[0] < b[0] : a[1] < b[1];
        });
        vector<int> res;
        if (edges.empty()) {
            res.push_back(0);
            return res;
        }
        unordered_map<int, vector<int> > map;
        for (auto edge:edges) {
            map[edge[0]].push_back(edge[1]);
            map[edge[1]].push_back(edge[0]);
        }
        int minlevel = INT_MAX;
        for (int i = 0; i < n; i++) {
            unordered_set<int> set;
            set.insert(i);
            vector<int> temp = map[i];
            int level = 0;
            queue<int> q;
            for (auto t:temp) q.push(t);
            while(!q.empty()){
                int size = q.size();
                for (int j = 0; j < size; j++) {
                    int head = q.front(); q.pop();
                    if (set.count(head) != 0) continue;
                    set.insert(head);
                    vector<int> v = map[head];
                    for (auto num:v) {
                        if(set.count(num) != 0) continue;
                        q.push(num);
                    }
                }
                level++;
                if (level > minlevel) break;
            }
            if (!q.empty()) continue;
            else if (level == minlevel){
                res.push_back(i);
            }
            else if (level < minlevel) {
                res.clear();
                res.push_back(i);
                minlevel = level;
            }
        }
        return res;
    }

    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        if (n == 1) return {0};
        vector<int> res;
        vector<unordered_set<int>> adj(n);
        queue<int> q;
        for (auto edge : edges) {
            adj[edge[0]].insert(edge[1]);
            adj[edge[1]].insert(edge[0]);
        }
        for (int i = 0; i < n; ++i) {
            if (adj[i].size() == 1) q.push(i);
        }
        while (n > 2){
            int size = q.size();
            n = n - size;
            for (int i = 0; i < size; i++) {
                int temp = q.front(); q.pop();
                for (auto t:adj[temp]) {
                    adj[t].erase(temp);
                    if (adj[t].size() == 1) q.push(t);
                }
            }
        }
        while (!q.empty()) {
            res.push_back(q.front()); q.pop();
        }
        return res;
    }
};

class SolutionT413 {
public:
    int numberOfArithmeticSlices(vector<int>& A) {
        int size = A.size();
        if (n < 3) return 0;
        vector<int> dp(size, 0);
        for (int i = 2; i < size; i++) {
            if (A[i] - A[i - 1] == A[i - 1] - A[i - 2]){
                dp[i] = dp[i-1] + 1;
            }
        }
        return accumulate(A.begin(), A.end());
    }
};

class SolutionT64 {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        if (m == 0 || n == 0) return 0;
        vector<int> dp(n, 0);
        for (int i = 0; i < n; i++) {
            if (i==0) dp[i] = grid[0][i];
            else dp[i]+=dp[i-1];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j==0) dp[j] = dp[j] + grid[i][j];
                else dp[j] = grid[i][j] + min(dp[j-1], dp[j]);
            }
        }
        return dp.back();
    }
};

class SolutionT279 {
public:
    int numSquares(int n) {
        vector<int> dp(n + 1, 0);
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j*j < i; j++) {
                dp[i] = min(dp[i - j*j], dp[i] + 1);
            }
        }
        return dp[n];
    }
};

class SolutionT91 {
public:
    int numDecodings(string s) {
        int n = s.size();
        if (s.empty() || s[0] == '0') return 0;
        vector<int> dp(n + 1, 0);
        dp[0] = 1, dp[1] = 1;
        for (int i = 1; i < n; i++) {
            int dpIndex = i + 1;
            if (s[i] == '0' && (s[i-1] == '0' || s[i-1] >= '3')) return 0;
            if ( s[i-1] == '2' || s[i-1] == '1' ) {
                if (s[i] == '0') dp[dpIndex] = dp[dpIndex - 2];
                else if ((s[i-1] == '2' && s[i] <= '6') || s[i-1] == '1') dp[dpIndex] = dp[dpIndex - 1] + dp[dpIndex - 2];
                else dp[dpIndex] = dp[dpIndex - 1];
            }
            else dp[dpIndex] = dp[dpIndex - 1];
        }
        return dp.back();
    }
    int numDecodings(string s) {
        if (s.empty() || s[0] == '0') return 0;
        vector<int> dp(s.size() + 1, 0);
        dp[0] = 1;
        for (int i = 1; i < dp.size(); i++) {
            dp[i] = (s[i-1] == '0') ? 0:dp[i-1];
            if (i > 1 && (s[i-2] == '1' || (s[i-2] == '2' && s[i-1] <= '6'))) dp[i] += dp[i-2];
        }
        return dp.back();
    }    
};

class SolutionT139 {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        vector<bool> dp(s.size() + 1, false);
        dp[0] = true;
        for (int i = 1; i <= s.size(); i++) {
            for (auto word: wordDict) {
                int len = word.size();
                if (i >= len && s.substr(i - len, len) == word) {
                    dp[i] = dp[i] || dp[i - len];
                }
            }
        }
        return dp.back();
    }
};

class SolutionT221 {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if (matrix.empty() || !matrix[0].empty()) return 0;
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int> > dp(m, vector<int>(n, 0));
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '0') dp[i][j];
                else if (i==0 || j==0) dp[i][j] = 1;
                else {
                    dp[i][j] = min(dp[i-1][j-1], min(dp[i-1][j], dp[i][j-1])) + 1;
                    res = max(res, dp[i][j]);
                }
            }
        }
        return res * res;
    }

    int maximalSquare(vector<vector<char>>& matrix) {
        if (matrix.empty() || !matrix[0].empty()) return 0;
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> sum(m, vector<int>(n, 0));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n;j ++) {
                int t = matrix[i][j] - '0';
                if (i > 0) t += sum[i - 1][j];
                if (j > 0) t += sum[i][j - 1];
                if (i > 0 && j > 0) t -= sum[i - 1][j - 1];
                sum[i][j] = t;
                int cnt = 1;
                for (int k = min(i, j); k >= 0; k--) {
                    int d = sum[i][j]
                    if (i - cnt >= 0) d-=sum[i-cnt][j];
                    if (j - cnt >= 0) d-=sum[i][j-cnt];
                    if (i - cnt >= 0 && j - cnt >= 0) d += sum[i - cnt][j - cnt];
                    if (d == cnt * cnt) res = max(res, d);
                    ++cnt;j
                }
            }
        }
    }
};

class SolutionT650 {
public:
    int minSteps(int n) {
        if (n == 1) return 0;
        int res = n;
        for (int i = n - 1; i >= 1; i--) {
            if (n%i == 0) {
                res = min(res, minSteps(n/i) + i);
            }
        }
        return res;
    }

    //dp
    int minSteps(int n) {
        vector<int> dp(n + 1, 0);
        for (int i = 2; i <= n; ++i) {
            dp[i] = i;
            for (int j = i - 1; j > 1; --j) {
                if (i % j == 0) {
                    dp[i] = min(dp[i], dp[j] + i / j);
                }
            }
        }
        return dp[n];
    }

    int minSteps(int n) {
        vector<int> dp(n+1, 0);
        int h = sqrt(n);
        for (int i = 2; i <= n; i++) {
            dp[i] = i;
            for (int j = 2; j <= h; j++) {
                if (i%j == 0) {
                    dp[i] = dp[j] + dp[i/j];
                    break;
                }
            }
        }
        return dp[n];
    }
};

class SolutionT188 {
public:
    int maxProfit(int k, vector<int>& prices) {
        int days = prices.size();
        if (days < 2) return 0;
        if (k >= days) return maxProfitUnlimited(prices);
        vector<int> sell(k+1, 0), buy(k+1);
        for (int i = 0; i < days; i++) {
            for (int j = 1; j <= k; j++) {
                buy[j] = max(buy[j], sell[j-1] - prices[i]);
                sell[j] = max(sell[j], buy[j] + prices[i]);
            }
        }
        return sell[k];
    }

    int maxProfitUnlimited(vector<int>& prices) {
        int res = 0;
        for (int i = 1; i < prices.size()) {
            if (prices[i] > prices[i-1]) {
                res += prices[i] - prices[i-1];
            }
        }
        return res;
    }
    public int maxProfit(int k, int[] prices) {
        /**
        当k大于等于数组长度一半时, 问题退化为贪心问题此时采用 买卖股票的最佳时机 II
        的贪心方法解决可以大幅提升时间性能, 对于其他的k, 可以采用 买卖股票的最佳时机 III
        的方法来解决, 在III中定义了两次买入和卖出时最大收益的变量, 在这里就是k租这样的
        变量, 即问题IV是对问题III的推广, t[i][0]和t[i][1]分别表示第i比交易买入和卖出时
        各自的最大收益
        **/
        if(k < 1) return 0;
        if(k >= prices.length/2) return greedy(prices);
        int[][] t = new int[k][2];
        for(int i = 0; i < k; ++i)
            t[i][0] = Integer.MIN_VALUE;
        for(int p : prices) {
            t[0][0] = Math.max(t[0][0], -p);
            t[0][1] = Math.max(t[0][1], t[0][0] + p);
            for(int i = 1; i < k; ++i) {
                t[i][0] = Math.max(t[i][0], t[i-1][1] - p);
                t[i][1] = Math.max(t[i][1], t[i][0] + p);
            }
        }
        return t[k-1][1];
    }
    
    private int greedy(int[] prices) {
        int max = 0;
        for(int i = 1; i < prices.length; ++i) {
            if(prices[i] > prices[i-1])
                max += prices[i] - prices[i-1];
        }
        return max;
    }
};

class SolutionT309 {
public:
    int maxProfit(vector<int>& prices) {

    }
};

class SolutionT213 {
public:
    // dp[1] = max(nums[0], nums[1])
    int rob(vector<int>& nums) {
        if (nums.size() <= 0) return 0;
        if (nums.size() == 1) return nums[0];
        vector<int> dp(nums.size(), 0);
        dp[0] = nums[0], dp[1] = max(nums[0], nums[1]);
        int res = max(dp[0], dp[1]);
        for (int i = 2; i < nums.size() - 1; i++) {
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);
            res = max(res, dp[i]);
        }
        vector<int> dp2(nums.size(), 0);
        dp2[0] = 0, dp2[1] = nums[1];
        res = max(res, dp2[1]);
        for (int i = 2; i < nums.size(); i++) {
            dp2[i] = max(dp2[i - 1], dp2[i - 2] + nums[i]);
            res = max(res, dp2[i]);
        }
        return res;
    }
};

class SolutionT424 {
public:
    int characterReplacement(string s, int k) {
        int res = 0, maxCount = 0, start = 0;
        vector<int> bucket(26, 0);
        for (int i = 0; i + res <= s.size(); i++) {
            maxCount = max(maxCount, ++bucket[s[i] - 'A']);
            while (i - start - maxCount + 1> k) {
                --counts[s[start] - 'A'];
                ++start;
            }
            res = max(res, i - start + 1);
        }
        return res;
    }
};

class SolutionT0204 {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode *p = head, *q = head;
        while(q) {
            if (q->val < x) {
                int temp = q->val;
                q->val = p->val
                p->val = temp;
                p = p->next;
            }
            q = q->next;
        }
        return head;
    }
};

class SolutionT0205 {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* temp = l1;
        int len1 = 0;
        while(temp) {
            len1++;
            temp = temp->next;
        }
        temp = l2;
        while(temp) {
            len2++;
            temp = temp->next;
        }
        ListNode* List1 = len1 >= len2 ? l1:l2;
        ListNode* List2 = len1 >= len2 ? l2:l1;
        int dif = abs(len1 - len2);
        ListNode* cur = List1;
        while(dif > 1) {
            cur = cur->next;
            dif--;
        }
        ListNode* Next = addNode(cur->next, l2);
        if (Next->val == 0) cur->val;
    }
};

class SolutionT343 {
public:
    int integerBreak(int n) {
        vector<int> dp(n+1, 0);
        dp[0] = 1, dp[1] = 1, dp[2] = 1, dp[3] = 3, dp[4] = 4;
        if (n <= 4) {
            dp[3] = 2;
            return dp[i]
        }
        for (int i = 5; i <= n; i++) {
            for (int j = 1; j < i; j++) {
                dp[i] = max(dp[j]*(i - j), dp[i]);
            }
        }
        return dp[n];
    }
};

class SolutionT300 {
public:
    int lengthOfLIS(vector<int>& nums) {
        if (nums.size() <= 1) return nums.size();
        vector<int> dp;
        dp.push_back(nums[0]);
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] > dp.back()) dp.push_back(nums[i]);
            else {
                auto itr = lower_bound(dp.begin(), dp.end(), nums[i]);
                *itr = nums[i];
            }
        }
        return dp.size();
    }
};

class SolutionT646 {
public:
    //Greedy
    int findLongestChain(vector<vector<int>>& pairs) {
        sort(pairs.begin(), pairs.end(), [](vector<int> a, vector<int> b){
            return a[0] == b[0] ? a[1] <= b[1]:a[0] < b[0];
        });
        int res = 1, cur_start = pairs[0][0], cur_end = pairs[0][1];
        for (int i = 1; i < pairs.size(); i++) {
            if (pairs[i][0] > cur_end) {
                res++;
                cur_start = pairs[i][0];
                cur_end = pairs[i][1];
            }
            else if (pairs[i][0] >= cur_start && pairs[i][1] < cur_end) {
                cur_start = pairs[i][0];
                cur_end = pairs[i][1];
            }
        }
        return res;
    }

    //DP

};

class SolutionT376 {
public:
    //[0,0,0]出问题
    int wiggleMaxLength(vector<int>& nums) {
        if (nums.size() <= 2) return nums.size();
        vector<int> dp(nums.size(), 1);
        int res = 2;
        dp[0] = 1, dp[1] = nums[1] > nums[0] ? 2:-2;
        for (int i = 2; i < nums.size(); i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    if (j == 0) {
                        dp[i] = 2;
                        continue;
                    }
                    if(dp[j] < 0) dp[i] = max(abs(dp[i]), abs(dp[j]) + 1);
                }
                else if (nums[i] < nums[j] && dp[j] >0) {
                    dp[i] = - max(abs(dp[i]), abs(dp[j]) + 1);
                }
            }
            res = max(res, abs(dp[i]));
        }
        return res;
    }

    int abs(int n) {
        return n < 0 ? -n : n;
    }

    //gready妙啊！
    int wiggleMaxLength(vector<int>& nums) {
        int p = 1, q = 1, n = nums.size();
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] > nums[i-1]) p = q + 1;
            else if (nums[i] < nums[i-1]) q = p + 1;
        }
        return min(n, max(p, q));
    }

    //两个dp数组解决
    int wiggleMaxLength(vector<int>& nums) {
        if (nums.empty()) return 0;
        vector<int> p(nums.size(), 1);
        vector<int> q(nums.size(), 1);
        for (int i = 1; i < nums.size(); ++i) {
            for (int j = 0; j < i; ++j) {
                if (nums[i] > nums[j]) p[i] = max(p[i], q[j] + 1);
                else if (nums[i] < nums[j]) q[i] = max(q[i], p[j] + 1);
            }
        }
        return max(p.back(), q.back());
    }
};

class SolutionT494 {
public:
    int findTargetSumWays(vector<int>& nums, int S) {
        int m = nums.size();
        vector<unordered_map<int, int>> memo(nums.size());
        return helper(nums, S, 0, dp);
    }

    //memo[index]{之前是个map}[sum]从最后到index能够成sum的个数
    int helper(vector<int>& nums, int sum, int index,  vector<unordered_map<int, int>> &memo) {
        if (index >= nums.size()) return sum == 0;
        if (memo[index].count(sum)) return memo[index][sum];
        int cnt1 = helper(nums, sum - nums[index], index+1, dp); //index及之后 和为sum - nums[index]的数量
        int cnt2 = helper(nums, sum + nums[index], index+1， dp);
        return memo[index][sum] = cnt1 + cnt2;
    }


    //到第i步，sum = k，有多少种
    int findTargetSumWays(vector<int>& nums, int S) {
        int n = nums.size();
        vector<unordered_map<int, int>> dp(n + 1);
        dp[0][0] = 1;
        for (int i = 0; i < nums.size(); i++) {
            for (auto &a : dp[i]) {
                int sum = a.first, cnt = a.second;
                dp[i+1][sum + nums[i]] += cnt;
                dp[i+1][sum - nums[i]] += cnt;
            }
        }
        return dp[n][S];
    }

    int findTargetSumWays(vector<int>& nums, int S) {
        vector<vector<int>> dp(nums.size(), (2001, 0));
        dp[0][0 + nums[0] + 1000] = 1;
        dp[0][0 - nums[0] + 1000] +=1;
        for (int i = 1; i < nums.size(); i++) {
            for (int sum = -1000; sum <= 1000; sum++) {
                if (dp[i - 1][sum + 1000] > 1) {
                    dp[i][sum + nums[i] + 1000] += dp[i - 1][sum + 1000];
                    dp[i][sum - nums[i] + 1000] += dp[i - 1][sum + 1000];
                }
            }
        }
        return S > 0 ? 0 : dp[nums.size() - 1][S + 1000];
    }
};