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

class SolutionT39 {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        unordered_set<int> set(candidates.begin(), candidates.end());
        vector<int> temp;
        dfs(candidates, target, 0, res, temp);
    }

    void dfs(vector<int>& candidates, int target, int index, vector<vector<int>> &res, vector<int> &temp) {
        if (target < 0) return;
        if (target == 0) {res.push_back(temp); return;}
        for (int i = index; i < candidates.size(); i++) {
            temp.push_back(candidates[i]);
            dfs(candidates, target - candidates[i], i, res, temp);
            temp.pop_back();
        }
    }
};

class Solution {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> res;
        vector<int> temp;
        dfs(candidates, target, 0, res, temp);
        return res;
    }
    void dfs(vector<int>& candidates, int target, int index, vector<vector<int>> &res, vector<int> &temp) {
        if (target < 0) return ;
        if (target == 0) { res.push_back(temp); return ;}
        for (int i = index; i < candidates.size(); i++) {
            if (i >index && candidates[i-1] == candidates[i]) continue;
            temp.push_back(candidates[i]);
            dfs(candidates, target - candidates[i], i + 1, res, temp);
            temp.pop_back();
        }
    }
};

class SolutionT47 {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> out, visited(nums.size(), 0);
        sort(nums.begin(), nums.end());
        permuteUniqueDFS(nums, 0, visited, out, res);
        return res;
    }
    void permuteUniqueDFS(vector<int>& nums, int level, vector<int>& visited, vector<int>& out, vector<vector<int>>& res) {
        if (level >= nums.size()) {res.push_back(out); return;}
        for (int i = 0; i < nums.size(); i++) {
            if (visited[i] == 1) continue;
            if (i != 0 && nums[i] == nums[i-1] && visited[i - 1] == 0) continue;
            visited[i] == 1;
            out.push_back(nums[i]);
            permuteUniqueDFS(nums, level + 1, visited, out, res);
            out.pop_back();
            visited[i] = 0;
        }
    } 

};

class Solution {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        set<vector<int>> res;
        permute(nums, 0, res);
        return vector<vector<int>> (res.begin(), res.end());
    }
    void permute(vector<int>& nums, int start, set<vector<int>>& res) {
        if (start >= nums.size()) res.insert(nums);
        for (int i = start; i < nums.size(); ++i) {
            if (i != start && nums[i] == nums[start]) continue;
            swap(nums[i], nums[start]);
            permute(nums, start + 1, res);
            swap(nums[i], nums[start]);
        }
    }
};

class SolutionT78 {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> temp;
        dfs(nums, 0, temp, res);
        return res;
    }

    void dfs(vector<int>& nums, int index, vector<int> &temp, vector<vector<int>> &res) {
        if (index == nums.size()) {
            res.push_back(temp);
            return ;
        }
        temp.push_back(nums[index]);
        dfs(nums, index + 1, temp, res);
        temp.pop_back();
        dfs(nums, index + 1, temp, res);
    }
};

class SolutionT78_2 {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> out;
        helper(nums, 0, out, res);
        return res;
    }

    void helper(vector<int>& nums, int pos, vector<int>& out, vector<vector<int>>& res) {
        res.push_back(out);
        for (int i = pos; i < nums.size(); i++) {
            out.push_back(nums[i]);
            helper(nums, pos, out, res);
            out.pop_back();
        }
    }
};

class SolutionT90 {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> res;
        vector<int> temp;
        vector<int> visited(nums.size());
        dfs(nums, 0, temp, res, visited);
        return res;
    }

    void dfs(vector<int>& nums, int index, vector<int> &temp, vector<vector<int>> &res, vector<int> &visited) {
        if (index == nums.size()) {
            res.push_back(temp);
            return ;
        }
        if (index > 0 && nums[index-1] == nums[index] && visited[index-1] == 0) return;
        temp.push_back(nums[index]);
        visited[index] = 1;
        dfs(nums, index + 1, temp, res, visited);
        temp.pop_back();
        visited[index] = 0;
        dfs(nums, index + 1, temp, res, visited);
    }
};

class SolutionT90_2 {
private:
    vector<vector<int>> res;
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        res.clear();
        sort(nums.begin(), nums.end());
        vector<int>tmpres;
        dfs(nums, 0, tmpres);
        return res;
    }

    void dfs(vector<int> &nums, int index, vector<int> &tmpres) {
        if (index == nums.size()) {
            res.push_back(tmpres);
            return ;
        }
        int firstSame = index;
        while (firstSame >= 0 && nums[firstSame] == nums[index]) firstSame--;
        //firstSame是第一个和nums[index]相同的数字的位置
        int sameNum = index - firstSame;
        //和 nums[index]相同的数字的个数
        //当前递归遍历到了nums中的，与nums[index]相同的，第sameNum个
        if (sameNum == 0 || (tmpres.size() >= sameNum && tmpres[tmpres.size() - sameNum] == nums[index])) {
            //如果还没有重复数字
            //当前递归遍历到了nums中的，与nums[index]相同的，第sameNum个
            //如果前面sameNum - 1个与nums[index]相同的都在tmpres里，才把nums[index]放进去

            //选择nums[index];
            tmpres.push_back(nums[index]);
            dfs(nums, index+1, tmpres);
            tmpres.pop_back();
        }
        dfs(nums, index+1, tmpres);
    }
};

class Solution {
private:
    vector<vector<int> >res;
public:
    vector<vector<int> > subsetsWithDup(vector<int> &nums) {
        int len = nums.size();
        sort(nums.begin(), nums.end());
        vector<vector<int>> res{1};
        int last = nums[0], opResNum = 1;
        for (int i = 0; i < len; ++i) {
            if (nums[i] != last) {
                last = nums[i];
                opResNum = res.size();
            }
            int resSize = res.size();
            for (int j = resSize - 1; j >= resSize - opResNum; j--) {
                res.push_back(res[j]);
                res.back().push_back(nums[i]);
            }
        }
        return res;
    }
};

class SolutionT51 {
public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> res;
        vector<string> queens(n, string(n, '.'));
        helper(0, queens, res);
        return res;
    }

    void helper(int curRow, vector<string>& queens, vector<vector<string>>& res) {
        int n = queens.size();
        if (curRow == n) {
            res.push_back(queens);
            return ;
        }
        for (int i = 0; i < n; ++i) {
            queens[curRow][i] = 'Q';
            if (isValid(queens, curRow, i)) {
                helper(curRow + 1, queens, res);
            }
            queens[curRow][i] = '。';
        }
    }
    bool isValid(vector<string>& queens, int row, int col) {
        for (int i = 0; i < row; ++i) {
            if (queens[i][col] == 'Q') return false;
        }
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; --i, --j) {
            if (queens[i][j] == 'Q') return false;
        }
        for (int i = row - 1, j = col + 1; i >= 0 && j < queens.size(); --i, ++j) {
            if (queens[i][j] == 'Q') return false;
        }
        return true;
    }
};

class SolutionT51_2 {
public:
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
        for (int i = 0; i < n; ++i) {
            if (isValid(queenCol, curRow, i)) {
                queenCol[curRow] = i;
                helper(curRow + 1, queenCol, res);
                queenCol[curRow] = -1;
            }
        }
    }
    bool isValid(vector<int>& queenCol, int row, int col) {
        for (int i = 0; i < row; i++) {
            if (col == queenCol[i] || abs(i - row) == abs(queenCol[i] - col)) return false;
        }
        return true;
    }
};

class SolutionT52 {
public:
    int totalNQueens(int n) {
        int res = 0;
        vector<bool> cols(n), diag(2 * n), anti_diag(2 * n);
        helper(n, 0, cols, diag, anti_diag, res);
        return res;
    }
    void helper(int n, int row, vector<bool>& cols, vector<bool>& diag, vector<bool>& anti_diag, int& res) {
        if (row == n) ++res;
        for (int col = 0; col < n; ++col) {
            int idx1 = col - row + n, idx2 = col + row;
            if (cols[col] || diag[idx1] || anti_diag[idx2]) continue;
            cols[col] = diag[idx1] = anti_diag[idx2] = true;
            helper(n, row + 1, cols, diag, anti_diag, res);
            cols[col] = diag[idx1] = anti_diag[idx2] = false;
        }
    }
};

class SolutionT254 {
public:
    vector<vector<int>> getFactors(int n) {
        vector<vector<int>> res;
        helper(n, 2, {}, res);
        return res;
    }
    void helper(int n, int start, vector<int> out, vector<vector<int>>& res) {
        if (start == 1) {
            if (out.size() > 1) res.push_back(out);
            return ;
        }
        for (int i = start; i <= sqrt(n) + 1; i++) {
            if (n % i != 0) continue;
            out.push_back(i);
            helper(n / i, i, out, res);
            out.pop_back();
        }
    }
};

class SolutionT301 {
public:
    vector<string> removeInvalidParentheses(string s) {
        int left = 0, right = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') left ++;
            else if (s[i] == ')') right++;
        }
        int diff = left - right;
        vector<string> res;
        // string out = "";
        helper(0, s, res, diff);
        return res;
    }

    void helper(int index, string s, vector<string>& res, int diff) {
        if (diff == 0) {
            if (isValid(s)) res.push_back(s);
            return;
        }
        for (int i = index; i < s.size(); i++) {
            if (i != index && s[i] == s[i - 1]) continue;
            if (diff > 0 && s[i] == '(') {
                helper(i + 1, s.substr(0, i) + s.substr(i+1), res, diff - 1);
            }
            if (diff < 0 && s[i] == ')') {
                helper(i + 1, s.substr(0, i) + s.substr(i+1), res, diff + 1);
            }
        }
    }
    
    bool isValid(string t) {
        int cnt = 0;
        for (int i = 0; i < t.size(); ++i) {
            if (t[i] == '(') ++cnt;
            else if (t[i] == ')' && --cnt < 0) return false;
        }
        return cnt == 0;
    }   
};

class SolutionT491 {
public:
    //这个方法 对 [1,2,3,4,1,2,3,4]
    //[1,2,3,4,5,6,7,8,9,10,1,1,1,1,1]报错
    vector<vector<int>> findSubsequences(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> temp;
        vector<int> visited(nums.size(), -1);
        helper(nums, 0, temp, res, visited);
        return res;
    }

    void helper(vector<int>& nums, int index, vector<int> &temp, vector<vector<int>>& res, vector<int>& visited) {
        if (index >= nums.size()) {
            return ;
        }
        for (int i = index; i < nums.size(); i++) {
            if (i > index && nums[i] == nums[i-1] && visited[i-1] == -1) continue;
            if (temp.empty() || nums[i] >= temp.back()) {
                temp.push_back(nums[i]);
                visited[i] = 1;
                if (temp.size() > 1) res.push_back(temp);
                helper(nums, i + 1, temp, res, visited);
                temp.pop_back();
                visited[i] = -1;
            }
        }
    }
};

class SolutionT93 {
public:
    vector<string> restoreIpAddresses(string s) {
        vector<string> res;
        string temp = "";
        restore(s, 4, temp, res);
        return res;
    }

    void restore(string s, int k, string out, vector<string>& res) {
        if (k == 0) {
            if (s.empty()) res.push_back(out);
            return ;
        }
        for (int i = 1 ; i <= 3; i++) {
            if (i <= s.size() && isValid(s.substr(0, i))) {
                if (k == 1) restore(s.substr(i), k - 1, out + s.substr(0, i), res);
                else restore(s.substr(i), k - 1, out + s.substr(0, i) + ".", res);
            }
        }
    }

    bool isValid(string s) {
        if (s.empty() || s.size() > 3 || (s.size() > 1 && s[0] == '0')) return false;
        int res = atoi(s.c_str());
        return res <= 255 && res >= 0;
    }
};

class Solution {
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> res;
        if (s.empty()) return {{}};
        for (int i = 0 ; i < s.size(); i++) {
            if (!isPalindrome(s, i + 1)) continue;
            for (auto list : partition(s.substr(i + 1))) {
                list.insert(list.begin(), s.substr(0, i + 1));
                res.push_back(list);
            }
        }
        return res;
    }

    bool isPalindrome(string s, int n) {
        for (int i = 0; i < n / 2; i++) {
            if (s[i] != s[n - i - 1]) return false;
        }
        return true;
    }
};

class SolutionT132 {
public:
    int minCut(string s) {
        int n = s.size();
        vector<vector<bool>> dp(n, vector<bool>(n));
        for (int i = 0; i < s.size(); i++) {
            for (int j = 0; j <= i; j++) {
                if (s[i] == s[j] && (i - j <= 2 || dp[i-1][j+1])) dp[i][j] = true;
            }
        }
    }
};

class Solution {
public:
    /**
     * @param str: A string
     * @return: all permutations
     */
    vector<string> stringPermutation2(string &str) {
        // write your code here
        sort(str.begin(), str.end());
        vector<string> res;
        vector<int> visited(str.size(), -1);
        helper(str, "", visited, res);
        return res;
    }

    void helper(string str, string temp, vector<int> &visited, vector<string>& res) {
        if (temp.size() == str.size()) {
            res.push_back(temp);
            return ;
        }
        for (int i = 0; i < str.size(); i++) {
            if (visited[i] == 1) continue;
            if (i > 0 && str[i] == str[i-1] && visited[i-1] == -1) continue;
            temp.push_back(str[i]);
            visited[i] = 1;
            helper(str, temp, visited, res);
            temp.pop_back();
            visited[i] = -1;
        }
    }
};

class Solution {
public:
    /**
     * @param n: An integer
     * @param str: a string with number from 1-n in random order and miss one number
     * @return: An integer
     */
    int findMissing2(int n, string &str) {
        // write your code here
        vector<bool>visit(n+1,false);
        return dfs(str,n,visit,0);
    }
    int dfs(string &str,int n,vector<bool> &visit,int index)
    {
        if(index==str.size())
        {
            vector<int>res;
            for (int i = 1; i <= n; i++) {
                if(!visit[i]) res.push_back(i);
            }
            if(res.size()==1) return res[0];
            else return -1;
        }
        else if(str[index]=='0') return -1;
        else
        {
            for(int i = 1; i < 3;i++)
            {
                if(index+i-1<str.size())
                {
                    int num=stoi(str.substr(index,i));
                    if(num>=1&&!visit[num]&&num<=n)
                    {
                        visit[num]=true;
                        int ret=dfs(str,n,visit,index+i);
                        if(ret!=-1) return ret;
                        visit[num]=false;
                    }
                }
            }
            return -1;
        }
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

class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        vector<vector<int>> res;
        vector<int> temp;
        TreeNode *cur = root, *pre = nullptr;
        vector<TreeNode*> st;
        int target = 0;
        while (cur || !st.empty()) {
            while (cur) {
                st.push_back(cur);
                target += cur->val;
                cur = cur->left;
            }
            cur = st.back();
            if (!cur->left && !cur->right && target == sum) {
                vector<int> v;
                for (auto &a : st) v.push_back(a->val);
                res.push_back(v);
            }
            if (cur->right && cur->right != pre) {
                cur = cur->right;
            } else {
                pre = cur;
                target -= cur->val;
                st.pop_back();
                cur = nullptr;
            }
        }
    }
};

class SolutionL246 {
public:
    /*
     * @param root: the root of binary tree
     * @param target: An integer
     * @return: all valid paths
     */
    vector<vector<int>> binaryTreePathSum2(TreeNode * root, int target) {
        // write your code here
        vector<vector<int>> res;
        vector<int> temp;
        if (!root) return res;
        helper(root, temp, res, target);
        return res;
    }

    void helper(TreeNode* root, vector<int> &temp, vector<vector<int>> &res, int target) {
        if (!root) return ;
        temp.push_back(root->val);
        int temp_sum = target;
        for (int i = temp.size() - 1; i >= 0; i--) {
            temp_sum -= temp[i];
            if (temp_sum == 0) {
                vector<int> temp_res;
                for (int j = i; j < temp.size(); j++) {
                    temp_res.push_back(temp[j]);
                }
                res.push_back(temp_res);
            }
        }
        helper(root->left, temp, res, target);
        helper(root->right, temp, res, target);
        temp.pop_back();
    }
};

class ParentTreeNode {
public:
    int val;
    ParentTreeNode *parent, *left, *right;
};

class SolutionT472 {
public:
    /*
     * @param root: the root of binary tree
     * @param target: An integer
     * @return: all valid paths
     */
    vector<vector<int>> binaryTreePathSum3(ParentTreeNode * root, int target) {
        // write your code here
        vector<vector<int>> results;
        dfs(root, target, results);
        return results;
    }

    //这个dfs是用来遍历以各个点作为起始点，来找和为target的
    void dfs(ParentTreeNode* root, int target, vector<vector<int>> &results) {
        if (!root) return ;

        vector<int> buffer;
        //这个findSum的作用，和246一样，中序遍历，用vec来回过头找target==0的情况
        findSum(root, nullptr, target, buffer, results); 
        
        dfs(root->left, target, results);
        dfs(root->right, target, results);
    }

    void findSum(ParentTreeNode *root, ParentTreeNode *father, int target,
        vector<int> &buffer, vector<vector<int>> &results) {
        if (!root) return ;
        buffer.push_back(root->val);
        target -= root->val;
        if (target == 0) {
            results.push_back(buffer);
        }
        //father这个节点保证在这个三个方向递归的时候，不会来回重复穿梭
        if (root->parent != NULL && root->parent != father)
            findSum(root->parent, root, target, buffer, results);

        if (root->left != NULL && root->left  != father)
            findSum(root->left, root, target, buffer, results);

        if (root->right != NULL && root->right != father)
            findSum(root->right, root, target, buffer, results);

        buffer.pop_back();
                    
    }
};

class SolutionT140 {
public:
    vector<string> wordBreak(string s, vector<string>& wordDict) {
        unordered_map<string, vector<string>> m;
        vector<string> res;
        res = helper(s, wordDict, m);
        return res;   
    }

    vector<string> helper(string s, vector<string>& wordDict, unordered_map<string, vector<string>>& m) {
        if (m.count(s)) return m[s];
        if (s.empty()) return {""};
        vector<string> res;

        for (auto word : wordDict) {
            int temp_size = word.size();
            if (temp_size > s.size() || s.substr(0, temp_size) != word) continue;
            if (s.substr(0, temp_size) == word) {
                auto temp_res = helper(s.substr(temp_size), wordDict, m);
                for (auto r : temp_res) {
                    res.push_back( word + (r.empty() ? "" : " ") + r);
                }
            }
        }

        m[s] = res;
        return res;
    }
};

class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> set(wordList.begin(), wordList.end());
        if (!set.count(endWord)) return 0;
        queue<pair<string, int>> q;
        q.push({beginWord, 1});
        int res = INT_MAX;
        while (!q.empty()) {
            for (int j = q.size(); j >0; j--) {
                auto t = q.front(); q.pop();
                string temp_word = t.first;
                int temp_step = t.second;
                for (int i = 0; i < temp_word.size(); i++) {
                    for (int k = 'a'; k <= 'z'; k++) {
                        string new_word = temp_word;
                        new_word[i] = k;
                        int cur_step = temp_step + 1;
                        if (!set.count(new_word)) continue;
                        if (new_word == endWord) {
                            return cur_step;
                        }
                        if (new_word != temp_word) {
                            q.push({new_word, cur_step});
                            set.erase(new_word);
                        }
                    }
                }
            }
        }
        return 0;
    }
};

class SolutionT1192
{
public:
    vector<vector<int>> res;
    vector<vector<int>> criticalConnections(int n, vector<vector<int>>& connections) 
    {
        unordered_map<int, unordered_set<int>> adjvex;
        for (auto v : connections)
        {
            int x = v[0];
            int y = v[1];
            adjvex[x].insert(y);
            adjvex[y].insert(x);
        }


        //------------------------ tarjan算法找桥 ------------------------//
        vector<int> dfn(n, 0);          //dfs到的时间
        vector<int> low(n, 0);          //可以回溯到的最早的时间点
        int T = 1;                  //全局时间


        tarjan(0, -1, dfn, low, T, adjvex);
        return res;
    }

    void tarjan(int x, int parent, vector<int> &dfn, vector<int> &low, int T,  unordered_map<int, unordered_set<int>> adjvex) {
        dfn[x] = T;
        low[x] = T;
        T++;

        for (auto y : adjvex[x]) {
            if (y == parent) continue;
            if (dfn[y] == 0) {
                tarjan(y, x, dfn, low, T, adjvex);
                low[x] = min(low[x], low[y]);

                if (low[y] > dfn[x]) res.push_back({x, y});
            }
            else if (dfn[y] != 0)
                low[x] = min(low[x], dfn[y]);
        }
    }
};