#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<stack>
#include<list>
#include<utility>
#include<cstring>
#include<string>
#include<unordered_map>
#include<unordered_set>
#include<hash_map>
#include<deque>
using namespace std;

class SolutionT1249 {
public:
    string minRemoveToMakeValid(string s) {
        int left = 0, right = 0, target = 0, expect_left = 0, cur_right = 0;
        for (auto a : s) {
            if (a == '(') left++;
            else if (a == ')') right++;
        }
        target = min(left, right);
        stack<char> st;
        string res = "";
        for (int i = 0; i < s.size(); i++) {
            if (isalpha(s[i])) {
                res.push_back(s[i]);
            }
            if (s[i] == '(') {
                target -=1;
                if (target >= 0 && cur_right < right) {
                    expect_left++;
                    if (right - cur_right >= expect_left) {
                        st.push('(');
                        res.push_back(s[i]);
                    }
                }
            } else if (s[i] == ')') {
                cur_right++;
                if (!st.empty()) {
                    expect_left--;
                    st.pop();
                    res.push_back(s[i]);
                }
            }
        }
        return res;
    }
};

class SolutionT953 {
public:
    bool isAlienSorted(vector<string>& words, string order) {
        unordered_map<char, int> dict;
        for (int i = 0 ; i < order.size(); i++) {
            dict[order[i]] = i;
        }
        for (int j = 0 ; j < words.size() - 1; j++) {
            string word1 = words[j], word2 = words[j + 1];
            int size = min(word1.size(), word2.size()), k = 0, equal = 0;
            for (; k < size; k++) {
                if (dict[word1[k]] < dict[word2[k]]) break;
                if (dict[word1[k]] > dict[word2[k]]) return false;
                if (dict[word1[k]] == dict[word2[k]]) equal++;
            }
            if (equal == size && size < word1.size()) return false;
        }
        return true;
    }
};

class SolutionT680 {
public:
    bool validPalindrome(string s) {
        int start = 0, end = s.size() - 1;
        while (start < end) {
            if (s[start] == s[end]) {
                start++;
                end--;
            } else {
                return isPalindrome(s, start+1, end) || isPalindrome(s, start, end-1);
            }
        }
        return true;
    }

    bool isPalindrome(string s, int start, int end) {
        while (start < end) {
            if (s[start++] != s[end--]) return false;
        }
        return true;
    }
};

class SolutionT1762 {
public:
    vector<int> findBuildings(vector<int>& heights) {
        deque<int> inc;
        for (int i = 0; i < heights.size(); i++) {
            if (inc.empty()) inc.push_back(i);
            while(!inc.empty() && inc.back() != i && heights[i] >= heights[inc.back()]) {
                inc.pop_back();
            }
            inc.push_back(i);
        }
        vector<int> res(inc.begin(), inc.end());
        return res;
    }
};

class SparseVector {
 public:
  SparseVector(vector<int>& nums) {
    for (int i = 0; i < nums.size(); ++i)
      if (nums[i])
        v.push_back({i, nums[i]});
  }

  // Return the dotProduct of two sparse vectors
  int dotProduct(SparseVector& vec) {
    int ans = 0;

    for (int i = 0, j = 0; i < v.size() && j < vec.v.size();)
      if (v[i].first == vec.v[j].first)
        ans += v[i++].second * vec.v[j++].second;
      else if (v[i].first < vec.v[j].first)
        ++i;
      else
        ++j;

    return ans;
  }

 private:
  vector<pair<int, int>> v;  // {index, num}
};

class SolutionT42 {
public:
    int trap(vector<int>& height) {
        int left = 0, right = height.size() - 1, res = 0;
        while (left < right) {
            int mn = min(height[left], height[right]);
            if (mn == height[left]) {
                left++;
                while (left < right && height[left] < mn) {
                    res += mn - height[left];
                }
            } else {
                right--;
                while(left < right && height[right] < mn) {
                    res += mn - height[right];
                }
            }
        }
        return res;
    }

    int trap(vector<int>& height) {
        stack<int> st;
        int i = 0, res = 0;
        while (i < height.size()) {
            if (st.empty() || height[i] < height[st.top()]) {
                st.push(i++);
            } else {
                int t = st.top(); st.pop();
                if (st.empty()) continue;
                res += ((min(height[i], height[st.top()]) - height[t]) * (i - st.top() - 1));
            }
        }
        return res;
    }
};

class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        stack<int> st;
        int res = INT_MIN;
        for (int i = 0; i < heights.size(); i++) {
            if (st.empty() || heights[i] < heights[st.top()]) {
                st.push(i);
                res = max(res, heights[i] * 2);
            } else {
                while (heights[i] > heights[st.top()]) {
                    int temp = heights[st.top()] * (i - st.top() + 1);
                    st.pop();
                    res = max(res, temp);
                }
                st.push(i);
            }
        }
        return res;
    }
};

class SolutionT739 {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        vector<int> res;
        vector<int> temp;
        for (int i = temperatures.size() - 1; i >= 0; i--) {
            if (temp.empty() || temperatures[i] < temperatures[temp.back()]) {
                if (!temp.empty()) res.push_back(temp.back() - i);
                else res.push_back(0);
                temp.push_back(i);
            } else {
                while (!temp.empty() && temperatures[i] >= temperatures[temp.back()]) {
                    temp.pop_back();
                }
                if (temp.empty()) res.push_back(0);
                else res.push_back(temp.back() - i);
                temp.push_back(i);
            }
        }
        reverse(res.begin(), res.end());
        return res;
    }

    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        vector<int> res{n, 0};
        stack<int> st;
        for (int i = 0; i < temperatures.size(); i++) {
            while (!st.empty() && temperatures[i] > temperatures[st.top()]) {
                auto t = st.top(); st.pop();
                res[t] = i - t;
            }
            st.push(i);
        }
        return res;
    }
};

class SolutionT907 {
public:
    int sumSubarrayMins(vector<int>& arr) {
        long res = 0;
        int m = 1e9 + 7;
        for (int i = 0; i < arr.size(); i++) {
            // vector<int> st;
            int min_val = INT_MAX;
            for (int j = i; j < arr.size(); j++) {
                min_val = min(min_val, arr[j]);
                res += min_val;
            }
        }
        return res % m;
    }

    int sumSubarrayMins(vector<int>& arr) {
        int n = arr.size(), res = 0, M = 1e9+7;
        stack<int> st{{-1}};
        vector<int> dp(n+1); 
        //dp[i] 表示以数字 arr[i-1] 结尾的所有子数组最小值之和
        // 所以dp[i] 是 i - 1个子数组，他们的最小值，的和
        for (int i = 0; i < arr.size(); i++) {
            while (st.top() != -1 && arr[i] <= st.top()) st.pop();
            //经过这一步操作，栈顶st.top()一定是第一个比当前元素小的元素
            // dp[st.top() + 1] 是以st.top结尾的所有子数组的最小元素和
            // (i - st.top()) * arr[i]
            //是从 st.top() --- i, 这一段，以arr[i]结尾的子数组的最小值的和
            dp[i + 1] = (dp[st.top() + 1] + (i - st.top()) * arr[i]) % M;
            st.push(i);
            res = (res + dp[i + 1]) % M;
        }
        return res;
    }
};

class SolutionT503 {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        int n = nums.size();
        vector<int> temp(2 * n, 0);
        vector<int> res(n, -1);
        for (int i = 0; i < 2 * n; i++) {
            temp[i] = nums[i % n];
        }
        stack<int> st;
        for (int j = 0; j < 2 * n; j++) {
            while (!st.empty() && temp[j] > temp[st.top()]) {
                int cur_pos = st.top(); st.pop();
                if (res[cur_pos % n] != -1) {
                    res[cur_pos % n] = temp[j];
                }
            }
            st.push(j);
        }
        return res;
    }
};

class StockSpanner {
public:
    StockSpanner() {
        
    }
    
    int next(int price) {
        int cur_span = 0;
        while(!st.empty() && price >= st.top().first) {
            int pre_span = st.top().second; st.pop();
            cur_span += pre_span;
        }
        st.push(make_pair(price, cur_span));
        return cur_span;
    }
private:
    stack<pair<int, int>> st;
};

class SolutionT321 {
public:
    vector<int> maxNumber(vector<int>& nums1, vector<int>& nums2, int k) {
        
    }
};

class BinaryMatrix {
  public:
    int get(int row, int col);
    vector<int> dimensions();
};

class SolutionT1428 {
public:
    int leftMostColumnWithOne(BinaryMatrix &binaryMatrix) {
        int n = binaryMatrix.dimensions()[0];
        int m = binaryMatrix.dimensions()[1];

        int check = m - 1;
        for (int i = 0; i < n; i++) {
            while (check >= 0 && binaryMatrix.get(i, check) == 1) {
                check--;
            }
        }
        return check == m - 1 ? -1 : check + 1;
    }
};

class Node {
public:
    int val;
    Node* left;
    Node* right;

    Node() {}

    Node(int _val) {
        val = _val;
        left = NULL;
        right = NULL;
    }

    Node(int _val, Node* _left, Node* _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};

class Solution {
public:
    Node* treeToDoublyList(Node* root) {
        Node* head = new Node(0);
        stack<Node*> st;
        Node *cur = root, *pre = nullptr;
        while (cur || !st.empty()) {
            while(cur) {
                st.push(cur);
                cur = cur->left;
            }
            cur = st.top(); st.pop();
            if (!pre) {
                head = cur;
                pre = cur;
            } else {
                pre->right = cur;
                cur->left = pre;
                pre = pre->right;
            }
            cur = cur->right;
        }
        pre->right = head;
        head->left = pre;
        return head;
    }
};

class SolutionT560 {
public:
    int subarraySum(vector<int>& nums, int k) {
        int sum = 0, left = 0, right = 0, n = nums.size(), res = 0;
        while (right < n) {
            while (right < n && sum < k) {
                sum += nums[right++];
            }
            while (left <= right && res > k) {
                sum -= nums[left];
                left++;
            }
            if (sum == k) res++;
            right++;
        }
        return res;
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

class SolutionT938 {
public:
    int rangeSumBST(TreeNode* root, int low, int high) {
        TreeNode* p = root;
        stack<TreeNode*> st;
        int sum = 0;
        while(p || !st.empty()) {
            while (p) {
                st.push(p);
                p = p->left;
            }
            p = st.top(); st.pop();
            if (low <= p->val && p->val <= high) sum += p->val;
            p = p->right;
        }
        return sum;
    }
};

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

class SolutionT236 {
public:
    TreeNode* lowestCommonAncestor1(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root) return NULL;
        if (root == p || root == q) return root == p ? p : q;
        TreeNode* left = lowestCommonAncestor1(root->left, p, q);
        TreeNode* right = lowestCommonAncestor1(root->right, p, q);
        if (left && right) return root;
        else return left != NULL ? left : right;
    }

    //优化剪枝
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
       if (!root || p == root || q == root) return root;
       TreeNode *left = lowestCommonAncestor(root->left, p, q);
       if (left && left != p && left != q) return left;
       TreeNode *right = lowestCommonAncestor(root->right, p , q);
       if (left && right) return root;
       return left ? left : right;
    }
};

class SolutionT1644 {
public:
    bool pFound = false;
    bool qFound = false;
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root) return NULL;
        TreeNode* LCA = helper(root, p, q);
        return pFound && qFound? LCA : nullptr;
    }

    TreeNode* helper(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root) return NULL;
        TreeNode* left = helper(root->left, p, q);
        TreeNode* right = helper(root->right, p, q);
        if (root == p) {
            pFound = true;
            return root;
        }
        if (root == q) {
            qFound = true;
            return root;
        }
        return left == nullptr ? right : ( (right == nullptr) ? left : root);
    }
};

//T1644 老方法找到后，再进行一次搜索
class SolutionT1644_2 {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root) return NULL;
        TreeNode* LCA = helper(root, p, q);
        if (LCA == q) {
            return helper(LCA, p, p) == NULL ? NULL : LCA;
        } else if (LCA == p) {
            return helper(LCA, q, q) == NULL ? NULL : LCA;
        }
        else return LCA;
    }

    TreeNode* helper(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root || root == p || root == q) return root;
        TreeNode* left = helper(root->left, p, q);
        TreeNode* right = helper(root->right, p, q);
        return left == NULL ? right : right == NULL ? left : root;
    }
};


class SolutionT523 {
public:
    bool checkSubarraySum(vector<int>& nums, int k) {
        int sum = 0, n = nums.size();
        unordered_map<int, int> map{{0, 0}};
        for (int i = 0; i < nums.size(); i++) {
            sum += nums[i];
            int rest = (k==0) ? sum : sum % k;
            if (map.count(rest)) {
                if (i - map[rest] > 1) return true;
            } else map[rest] = i;
        }
        return false;
    }
};

class Solution {
public:
    Solution(vector<int>& w) {
        sum = w;
        for (int i = 1; i < w.size(); i++) {
            sum[i] = sum[i - 1] + w[i];
        }
    }
    
    int pickIndex() {
        int x = rand() % sum.back(), left = 0, right = sum.size() - 1;
        while (left < right) {
            int mid = (right - left) / 2 + left;
            if (sum[mid] <= x) left = mid + 1;
            else right = mid; 
        }
        return right;
    }
private:
    vector<int> sum;
};


class SolutionT301 {
public:
    vector<string> removeInvalidParentheses(string s) {
        int left = 0, right = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') left++;
            else if (s[i] == ')') {
                if (left == 0) right++;
                else left--;
            };
        }
        vector<string> res;
        string temp = "";
        helper(s, 0, left, right, res);
        return res;
    }

    void helper(string s, int index, int left, int right, vector<string> &res) {
        if (left == 0 && right == 0) { 
            if (isValid(s)) {
                res.push_back(s);
                return ;
            }
        }
        for (int i = index; i < s.size(); i++) {
            if (i > index && s[i] == s[i-1]) continue;
            if (left > 0 && s[i] == '(') helper(s.substr(0, i) + s.substr(i + 1), i, left - 1, right, res);
            if (right > 0 && s[i] == ')') helper(s.substr(0, i) + s.substr(i + 1), i, left, right - 1, res);
        }
    }

    bool isValid(string s) {
        int cnt = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') cnt++;
            else if (s[i] == ')' && --cnt < 0) return false;
        }
        return cnt == 0;
    }
};

class SolutionT314 {
public:
    vector<vector<int>> verticalOrder(TreeNode* root) {
        
    }
};

class SolutionT1263 {
public:
    vector<vector<int>> directions{{0, 1}, {-1, 0}, {0, -1}, {1, 0}};
    bool flag = false;
    int minPushBox(vector<vector<char>>& grid) {
        pair<int, int> box, target, person;
        int n = grid.size(), m = grid[0].size(), res = 0;
        for (int i = 0 ; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == 'T') target = {i, j};
                if (grid[i][j] == 'S') person = {i, j};
                if (grid[i][j] == 'B') box = {i, j};
            }
        }
        
        for (int j = 0; j < 4; j++) {
            if (approachable(grid, box, target, j)) {
                auto from = getNext(box, (j + 2)%3);
                if (isValid(from) && grid[from.first][from.second] != '#' && arrivable(target, person)) res = min(res, step);
            }
        }
        return res;
    }

    visited

    bool approachable(vector<vector<char>> grid, pair<int, int> start, pair<int, int> target, int towards) {
        pair<int, int> from = getNext(start, towards + 2);
        if (target == start) {
            return flag = true;
        }
        for (int i = 0; i < 4; i++) {
            if (i == (towards + 2)%3) continue;
            auto next = getNext(start, i);
            auto oppsite = getNext(start, i + 2);
            if (isValid(grid, next) && grid[oppsite.first][oppsite.second] != '#' && approachable(grid, next, target, i)) return true;
        }
        return false;
    }

    pair<int, int> getNext(pair<int, int> start, int towards) {
        return {start.first + directions[towards % 3][0], start.second + directions[towards % 3][1]}; 
    }
};

//
class SolutionT1263 {
public:
    vector<vector<int>> directions{{0, 1}, {-1, 0}, {0, -1}, {1, 0}};
    int minPushBox(vector<vector<char>>& grid) {
        int m=grid.size(),n=grid[0].size();
        queue<pair<int,int>> q; 
        //store the next valid box position: it shall store: player,box,
        //bfs的层循环队列，放的往往是你需要的数据，这里存放box的坐标，player的坐标
        unordered_set<string> v;
        int src=0,dst=0,player=0;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j]=='S') {player=i*n+j;grid[i][j]='.';}
                if(grid[i][j]=='B') {src=i*n+j;grid[i][j]='.';}
                if(grid[i][j]=='T') {dst=i*n+j;grid[i][j]='.';}
            }
        }
        if (src == dst) return 0;
        q.push({src, player});
        int step=0;
        int dir[][4]={{-1,0},{1,0},{0,-1},{0,1}};
        while (!q.empty()) {
            int sz = q.size();
            for (int i = 0;  i < sz; i++) {
                auto item = q.front(); q.pop();
                int box = item.first, player = item.second;
                if (box == dst) return step;
                //拆解出当前的横纵坐标
                int xb = box / n, yb = box % n;
                for (int i = 0; i < 4; i++) {
                    int next_x = xb + dir[i][0], next_y = yb + dir[i][1];
                    //下一个位置box对应的player的位置
                    int next_px = xb - dir[i][0], next_py = yb - dir[i][1];
                    if(next_x<0||next_y<0||next_x>=m||next_y>=n||grid[next_x][next_y]=='#') continue;
                    if(next_px<0||next_py<0||next_px>=m||next_py>=n||grid[next_px][next_py]=='#') continue;
                    string s=to_string(box)+","+to_string(next_px*n+next_py);//box pos+person pos
                    //将这个string，作为visited里面放的内容就很巧
                    //用当前位置的box，而不是下一个box的位置，表示了from源头+ 
                    //下一个player应该在的位置，即代表了去向和方向
                    //整个string就代表了from to 以及方向
                    // x        x
                    // x        x
                    // x next_B xxxxxxxxxx
                    // x   B (当B变成next_B的时候，原来的B肯定是player的位置) 但是如果想把B往上推，player肯定在B的下方，但是是wall
                    // xxxxxxxxxxxxxxxxxxxx
                    //   player
                    if (v.count(s)) continue;
                    if (can_access(grid, player, next_px*n+next_py, box)) {
                        q.push({next_x * n + next_y, box});
                        v.insert(s);
                    }
                }
            }
            step++;
        }
        return -1;
    }

    //box此时是个障碍物，无法穿越
    //因为从上一个player，到下一个player的距离应该会很紧，用bfs会很省时间
    bool can_access(vector<vector<char>>& g,int src,int dst,int box){
        int m=g.size(),n=g[0].size();
        //bfs shall be better than dfs
        queue<int> q;
        vector<bool> v(m*n);
        q.push(src);
        v[src]=1;
        int dir[][2]={{-1,0},{1,0},{0,-1},{0,1}};
        g[box/n][box%n]='#';
        while(q.size()){
            int sz=q.size();
            while(sz--){
                int p=q.front();
                q.pop();
                if(p==dst) {g[box/n][box%n]='.';return 1;}
                int x0=p/n,y0=p%n;
                for(auto d: dir){
                    int x=x0+d[0],y=y0+d[1];
                    if(x<0||y<0||x>=m||y>=n||g[x][y]!='.'||v[x*n+y]) continue;
                    v[x*n+y]=1;
                    q.push(x*n+y);
                }
            }
        }
        g[box/n][box%n]='.';
        return 0;
    }
};

class SolutionT140 {
public:
    vector<string> wordBreak(string s, vector<string>& wordDict) {
        vector<string> res;
        unordered_map<string, vector<string>> map;
        res = dfs(s, wordDict, map);
        return res;
    }

    vector<string> dfs(string s, vector<string> wordDict, unordered_map<string, vector<string>>& map) {
        if (s == "") return {""};
        if (map.count(s)) return map[s];
        vector<string> cur_res;
        for (auto word : wordDict) {
            int sz = word.size();
            if (sz > s.size() || s.substr(0, sz) != word) continue;
            auto temp = dfs(s.substr(sz), wordDict, map);
            for (auto t : temp) {
                cur_res.push_back(word + (t.empty() ? "" : " ") + word);
            }
        }
        return map[s] = cur_res;
    }
};

class SolutionT282 {
public:
    vector<string> addOperators(string num, int target) {
        vector<string> res;
        int sum = 0;
        string temp = "";
        unordered_map<string, unordered_map<int, string>> map;
        dfs(0, num, target, temp, sum, map);
        return res;
    }

    void dfs(int index, string num, int target, string temp, int sum, unordered_map<string, unordered_map<int, string>> &mao) {
        if (index == num.size() && sum == target) {
            res.push_back(temp);
            return ;
        }
        for (int i = index; i < num.size(); i++) {
            for (int j = i; j < num.size(); j++) {
                auto str = stoi(num.substr(i, j - i + 1));
            }
        }
    }
private:
    vector<string> res;
};

class SolutionT282 {
public:
    vector<string> addOperators(string num, int target) {
        vector<string> res;
        helper(num, target, 0, 0, "", res);
        return res;
    }

    //不用指明当前的位置吗？
    //直接对当前string操作了
    void helper(string num, int target, long diff, long curNum, string out, vector<string>& res) {
        if (num.size() == 0 && curNum == target) {
            res.push_back(out);
        }
        for (int i = 1; i <= num.size(); i++) {
            string cur = num.substr(0, i);
            if (cur.size() > 1 && cur[0] == '0') return; //这种情况以0打头，直接return
            string next = num.substr(i);
            if (out.size() > 0) {
                helper(next, target, stoll(cur), curNum + stoll(cur), out + "+" + cur, res);
                helper(next, target, -stoll(cur), curNum - stoll(cur), out + "-" + cur, res);
                helper(next, target, diff * stoll(cur), (curNum - diff) + diff * stoll(cur), out + "*" + cur, res);
            } else {
                helper(next, target, stoll(cur), stoll(cur), cur, res);
            }
        }
    }
};

class SolutionT921 {
public:
    int minAddToMakeValid(string s) {
        stack<char> st;
        int res = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') st.push(s[i]);
            else if (s[i] == ')') {
                if (!st.empty()) {
                    st.pop();
                } else res++;
            }
        }
        res += st.size();
        return res;
    }
};

class SolutionT636 {
public:
    vector<int> exclusiveTime(int n, vector<string>& logs) {
        vector<int> res(n, 0);
        stack<pair<int, int>> st;
        for (auto log : logs) {
            int found1 = log.find(":");
            int found2 = log.find_last_of(":");
            int idx = stoi(log.substr(0, found1));
            string type = log.substr(found1 + 1, found2 - found1 -1);
            int time = stoi(log.substr(found2+1));
            if (type == "start") {
                if (!st.empty()) {
                    auto t = st.top(); //st.pop();
                    int last_idx = t.first, last_time = t.second;
                    res[last_idx] += time - last_time;
                    //st.push({last_idx, time});
                }
                st.push({idx, time});
            } else {
                auto ts = st.top(); st.pop();
                int ts_idx = ts.first, ts_time = ts.second;
                res[ts_idx] += time - ts_time + 1;

                if (!st.empty()) {
                    auto t = st.top(); st.pop();
                    int last_idx = t.first, last_time = t.second;
                    st.push({last_idx, time + 1});
                }
            }
        }
        return res;
    }
};

