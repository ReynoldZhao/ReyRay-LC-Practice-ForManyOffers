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

class Solution {
public:
    vector<int> sortArray(vector<int>& nums) {
        quicksort(nums, 0, nums.size() - 1);
        return nums;
    }

    void quicksort(vector<int> &nums, int start, int end) {
        if (start >= end) return ;
        int pivot = partition(nums, start, end);
        if (start > pivot) quicksort(nums, start, pivot - 1);
        if (end > pivot + 1)quicksort(nums, pivot + 1, end);
    }

    int random_partion(vector<int> &nums, int left, int right) {
        int pos = rand() % (right - left + 1) + left;
        swap(nums[pos], nums[right]);
    }

    int partition(vector<int> &nums, int left, int right) {
        int pivot = nums[right];
        int pos = left - 1;
        for (int i = left; i < right; i++) {
            if (nums[i] <= pivot) {
                swap(nums[++pos], nums[i]);
            }
        }
        swap(nums[right], nums[++pos]);
        return pos;
    }
};

class Solution {
public:
    int partition(vector<int>& nums, int l, int r) {
        int pivot = nums[r];
        int i = l - 1;
        //起始点是 l - 1
        //方便后面遍历，与swap交换，最稳妥的快排
        for (int j = l; j <= r - 1; ++j) {
            if (nums[j] <= pivot) {
                i = i + 1;
                swap(nums[i], nums[j]);
            }
        }
        swap(nums[i + 1], nums[r]);
        return i + 1;
    }
    // 基于随机的划分
    
    int randomized_partition(vector<int>& nums, int l, int r) {
        int i = rand() % (r - l + 1) + l;
        swap(nums[r], nums[i]);
        return partition(nums, l, r);
    }
    //注意 k一直指的是第k个
    void randomized_selected(vector<int>& arr, int l, int r, int k) {
        if (l >= r) return ;
        int pos = randomized_partition(arr, l, r);
        if (pos == k - 1) {
            return ;
        }
        if (pos < k - 1) {
            randomized_selected(arr, l, pos - 1, k);
        } else {
            randomized_selected(arr, pos + 1, r, k - pos - 1);
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
    int trap(vector<int>& height) {
        int res = 0;
        stack<int> st;
        for (int i = 0; i < height.size(); i++) {
            if (st.empty() || height[i] < height[st.top()]) {
                st.push(i);
                continue;
            }
            while (height[i] > height[st.top()]) {
                if (st.size() >= 2) {
                    int cur = height[st.top()]; st.pop();
                    int roof = min(height[i], height[st.top()]);
                    res += (roof - cur) * (i - st.top() - 1);
                }
                else st.pop();
            }
            st.push(i);
        }
        return res;
    }
};

class SolutionT76 {
public:
    string minWindow(string s, string t) {
        unordered_map<int, int> map;
        for (auto a : t) map[a]++;
        int left = 0, minLen = INT_MAX, cnt = 0;
        string res = "";
        for (int i = 0; i < s.size(); i++) {
            if (--map[s[i]] >= 0) cnt++;
            while (cnt == t.size()) {
                if (minLen > i - left + 1) {
                    minLen = i - left + 1;
                    res = s.substr(left, minLen);
                }
                if (++map[left++] > 0) --cnt;
            }
        }
        return res;
    }
};

class SolutionT440 {
public:
    /**
     * @param A: an integer array
     * @param V: an integer array
     * @param m: An integer
     * @return: an array
     */
    int backPackIII(vector<int> &A, vector<int> &V, int m) {
        // write your code here
        vector<int> dp(m + 1, 0);
        int n = A.size();
        for (int i = 0; i < n; i++) {
            for (int j = A[i]; j <= m; j++) {
                dp[j] = max(dp[j], dp[j - A[i]] + V[i]);
            }
        }
        return dp[m];
    }
};

class SolutionT562 {
public:
    /**
     * @param nums: an integer array and all positive numbers, no duplicates
     * @param target: An integer
     * @return: An integer
     */
    int backPackIV(vector<int> &nums, int target) {
        // write your code here
        vector<int> dp(target + 1, INT_MIN);
        dp[0] = 0;
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            for (int j = nums[i]; j <= target; j++) {
                dp[j] = max(dp[j], dp[j - nums[i]] + 1);
            }
        }
        return dp[target];
    }

    int backPackIV(vector<int> &nums, int target) {
        int m = target;
        vector<int> A = nums;
        vector<vector<int>> F(nums.size() + 1, vector<int> (target + 1, 0));

        F[0][0] = 1;
        for (int i = 1; i <= A.size(); i++) {
            for (int j = 0; j <= m; j++) {
                int k = 0;
                while (k * A[i - 1] <= j) {
                    F[i][j] += F[i-1][j - A[i-1]*k];
                    k+=1;
                }
            }
        }
        return F[A.size()][m];
    }
};

class Solution {
public:
    /**
     * @param nums: an integer array and all positive numbers
     * @param target: An integer
     * @return: An integer
     */
    int backPackV(vector<int> &nums, int target) {
        // write your code here
        int res = 0;
        sort(nums.begin(), nums.end());
        helper(nums, target, 0, res);
        return res;
    }
    void helper(vector<int>& num, int target, int start, int& res) {
        if (target < 0) return;
        if (target == 0) { res++; return; }
        for (int i = start; i < num.size(); ++i) {
            helper(num, target - num[i], i + 1, res);
        }
    }

    int backPackV(vector<int> &nums, int target) {
        vector<int> dp(target + 1);
        for (auto a : nums) {
            for (int j = target; j >= a; j--) {
                dp[j] += dp[j - a];
            }
        }
        return dp[target];
    }
};

class SolutionT564 {
public:
    /**
     * @param nums: an integer array and all positive numbers, no duplicates
     * @param target: An integer
     * @return: An integer
     */
    int backPackVI(vector<int> &nums, int target) {
        // write your code here
        vector<int> dp(target + 1, 0);
        dp[0] = 1;
        int n = nums.size();
        for (int i = 1; i <= target; i++) {
            for (int j = 0; j < n; j++) {
                if (j - nums[i] >= 0) dp[i] += dp[i - nums[j]];
            }
        }
        return dp[target];
    }
};

class SolutionT971 {
public:
    /**
     * @param k1: The coefficient of A
     * @param k2: The  coefficient of B
     * @param c: The volume of backpack
     * @param n: The amount of A
     * @param m: The amount of B
     * @param a: The volume of A
     * @param b: The volume of B
     * @return: Return the max value you can get
     */
    //dp[i][j] A类的前i个物品，B类的前j个物品，选择若干个可以得到的最大价值
    long long getMaxValue(int k1, int k2, int c, int n, int m, vector<int> &a, vector<int> &b) {
        // Write your code here
        sort(a.begin(), a.end());
        sort(b.begin(), b.end());
        vector<vector<long long>> dp(n, vector<long long> (m, 0));
        long long ans = 0;
        long long aa[2011], bb[2011];
        aa[0] = a[0], bb[0] = b[0];
        for (int i = 1; i < n; i++) aa[i] = aa[i-1] + a[i];
        for (int j = 1; j < m; j++) bb[j] = bb[j-1] + b[j];
        for (int i =1; i < n; i++) {
            dp[i][0] = dp[i-1][0] + k1 * (c - aa[i - 1]);
            ans = max(ans, dp[i][0]);
        }
        for (int j =1; j < n; j++) {
            dp[0][j] = dp[0][j-1] + k2 * (c - bb[j - 1]);
            ans = max(ans, dp[0][j]);
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (aa[i-1] + bb[j-1] > c) break;
                long long x = dp[i-1][j] + k1 * (c - aa[i-1] - bb[j-1]);
                long long y = dp[i][j-1] + k2 * (c - aa[i-1] - bb[j-1]);
                dp[i][j] = max(x, y);
                ans = max(ans, dp[i][j]);
            }
        }
        return ans;
    }
};

class SolutionT474 {
public:
    int findMaxForm(vector<string>& strs, int m, int n) {
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
        for (string str : strs) {
            int zeros = 0, ones = 0;
            for (char c : str) (c == '0') ? ++zeros : ++ones;
            for (int i = m; i >= zeros; --i) {
                for (int j = n; j >= ones; --j) {
                    dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1);
                }
            }
        }
        return dp[m][n];
    }
};

class SolutionT395 {
public:
    int longestSubstring(string s, int k) {
        int res = 0, n = s.size();
        for (int cnt = 1; cnt <= 26; cnt++) {
            int i = 0, start = 0, uniqueCnt = 0;
            vector<int> charCnt(26);
            while ( i < n ) {
                bool isValid = false;
                if (charCnt[s[i++] - 'a']++ == 0) uniqueCnt++;
                while (uniqueCnt > cnt) {
                    if (--charCnt[s[start++] - 'a'] == 0) uniqueCnt--;
                }
                for (int j = 0; j < 26; j++) {
                    if (charCnt[j] > 0 && charCnt[j] < k) isValid = false;
                }
                if (isValid) res = max(res, i - start);
            }
        }
    }
};

class Solution {
public:
    vector<int> findDiagonalOrder(vector<vector<int>>& nums) {
        map<int, vector<int>> map;
        int m = nums.size(), n = nums[0].size();
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                int key = i + j;
                map[key].push_back(nums[i][j]);
            }
        }

        // for (int i = 0; i < m; i++) {
        //     for (int j = 0; j < nums[i].size(); j++) {
        //         int key = i + j;
        //         map[key].push_back(nums[i][j]);
        //     }
        // }

        vector<int> res;
        for (auto m : map) {
            auto temp = m.second;
            reverse(temp.begin(), temp.end());
            for (auto t : temp) res.push_back(t);
        }
        return res;
    }
};

class SolutionT463 {
public:
    int islandPerimeter(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    res += 4;

                    if (i - 1 >= 0 && grid[i-1][j] == 1) res--;
                    if (i + 1 < m && grid[i+1][j] == 1) res--;
                    if (j - 1 >= 0 && grid[i][j-1] == 1) res--;
                    if (j + 1 < n && grid[i][j+1] == 1) res--;
                }
            }
        }
        return res;
    }
};

class SolutionT1911 {
public:
    long long maxAlternatingSum(vector<int>& nums) {
        int n = nums.size();
        vector<vector<long long>> dp(n, vector<long long> (2, 0));
        // 0 +, 1 -
        dp[0][0] = nums[0];
        for (int i = 1; i < n; i++) {
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + nums[i]);
            dp[i][1] = max(dp[i-1][1], dp[i - 1][0] - nums[i]);
        }
        return dp[n-1][0];
    }
};

class SolutionT209 { 
public: 
    int minSubArrayLen(int s, vector<int>& nums) { 
        int len = nums.size(), res = len + 1; 
        vector<int> sums(len+1, 0);
        for (int i = 1; i < len + 1; ++i) sums[i] = sums[i - 1] + nums[i - 1]; 
        for (int i = 0; i < len + 1; ++i) { 
            int right = searchRight(i + 1, len, sums[i] + s, sums); 
            if (right == len + 1) break; 
            if (res > right - i) res = right - i; 
        } 
        return res == len + 1 ? 0 : res; 
    }

    int searchRight(int left, int right, int key, vector<int> sums) {
        while (left < right) {
            int mid = (right - left) / 2 + left;
            if (sums[mid] >= key) right = mid;
            else left = mid + 1;
        }
        return right;
    } 
};

class SolutionT480 {
public:
    vector<double> medianSlidingWindow(vector<int>& nums, int k) {
        vector<double> res;
        multiset<int> small, large;
        for (int i = 0; i < nums.size(); ++i) {
            if (i >= k) {
                if (small.count(nums[i - k])) small.erase(small.find(nums[i - k]));
                else if (large.count(nums[i - k])) large.erase(large.find(nums[i - k]));
            }
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
};

class SolutionT567 {
public:
    bool checkInclusion(string s1, string s2) {
        int n1 = s1.size(), n2 = s2.size(), cnt = n1, left = -1;
        unordered_map<int, int> map;
        for (auto s : s1) map[s]++;
        for (int i = 0; i < n2; i++) {
            if (map.count(s2[i]) == 0) {
                while (left < i) {
                    ++left;
                    if (map.count(s2[left])) {
                        map[s2[left]]++;
                        if (map[s2[left]] > 0) cnt++;
                    }
                }
                //此时 left = i, 下一轮 i = i + 1
            }
            else {
                if (--map[s2[i]] >= 0) cnt--;
                while (cnt == 0) {
                    if (i - left == n1) return true;
                    if (++map[s2[++left]] > 0) ++cnt;
                }
            }
        }
    }
};

class SolutionT727 {
public:
    string minWindow(string s1, string s2) {
        int n1 = s1.size(), n2 = s2.size(), cnt = n1, left = 0, minLen = INT_MAX;
        string res = "";
        unordered_map<int, int> map;
        for (auto s : s1) map[s]++;
        for (int i = 0; i < n2; i++) {
            --map[s2[i]];
            if (map[s2[i]] >= 0 ) cnt--;
            while (cnt == 0 && left < i) {
                if (i - left + 1 < minLen) {
                    minLen = i - left + 1;
                    res = s2.substr(left, minLen);
                }
                if (++map[s2[left++]] > 0) cnt++;
            }
        }
        return res;
    }
};

class SolutionT727 {
public:
    string minWindow(string S, string T) {
        int m = S.size(), n = T.size(), start = -1, minLen = INT_MAX;
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, -1));
        for (int i = 0; i <= m; ++i) dp[i][0] = i;
        for (int i = 0; i < )
    }
};

class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int res = INT_MIN, left = 0, temp_sum = 0;
        for (int i = 0; i < nums.size(); i++) {
            temp_sum += nums[i];
            res = max(res, temp_sum);
            while (temp_sum < 0 && left <= i)
            {
                temp_sum-=nums[left++];
            }
        }
        return res;
    }
};

class SolutionT53 {
public:
    int maxSubArray(vector<int>& nums) {
        if (nums.size() == 1) return nums[0];
        long long temp_min = nums[0];
        long long sum = nums[0], res = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            sum += nums[i];
            res = max(max(res,sum), sum - temp_min);
            temp_min = min(temp_min, sum);
        }
        return res;
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
        istringstream in;
        return deserialize(in);
    }
private:
    void serialize(TreeNode* root, ostringstream& out) {
        if (root) {
            out << root->val << ' ';
            serialize(root->left, out);
            serialize(root->right, out);
        } else {
            out << "# ";
        }
    }

    TreeNode* deserialize(istringstream& in) {
        string val;
        in >> val;
        if (val == "#") return nullptr;
        TreeNode* root = new TreeNode(stoi(val));
        root->left = deserialize(in);
        root->right = deserialize(in);
        return root;
    }
};

class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        if (!root) return "#";
        return to_string(root->val)+","+serialize(root->left)+","+serialize(root->right);
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string &data) {
        if (data[0] == "#") {
            if (data.size() > 1) data = data.substr(2);
            return nullptr;
        } else {
            TreeNode* root = new TreeNode(helper(data));
            root->left = deserialize(data);
            root->right = deserialize(data);
            return root;
        }
    }
private:
    int helper(string &data) {
        int pos = data.find(",");
        data = data.substr(pos+1);
        return stoi(data.substr(0, pos));
    }
};

class Codec {
public:
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        if (root == nullptr) return "#";
        return to_string(root->val)+","+serialize(root->left)+","+serialize(root->right);
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        return mydeserialize(data);
    }
    TreeNode* mydeserialize(string& data) {
        if (data[0]=='#') {
            if(data.size() > 1) data = data.substr(2);
            return nullptr;
        } else {
            TreeNode* node = new TreeNode(helper(data));
            node->left = mydeserialize(data);
            node->right = mydeserialize(data);
            return node;
        }
    }
private:
    int helper(string& data) {
        int pos = data.find(',');
        int val = stoi(data.substr(0,pos));
        data = data.substr(pos+1);
        return val;
    }
};

struct DirectedGraphNode {
    int label;
    vector<DirectedGraphNode *> neighbors;
    DirectedGraphNode(int x) : label(x) {};
};

class SolutionT127 {
public:
    /**
     * @param graph: A list of Directed graph node
     * @return: Any topological order for the given graph.
     */
    vector<DirectedGraphNode*> topSort(vector<DirectedGraphNode*> graph) {
        // write your code here
        vector<DirectedGraphNode*> res;
        if (graph.empty()) return res;
        unordered_map<DirectedGraphNode*, int> map;
        for (auto node : graph) {
            for (auto t : node->neighbors) {
                map[t]++;
            }
        }
        queue<DirectedGraphNode*> q;
        for (auto node : graph) {
            if (map.count(node) == 0) { 
                q.push(node); 
                res.push_back(node);
            }
        }
        while(!q.empty()) {
            auto temp_node = q.front(); q.pop();
            for (auto t : temp_node->neighbors) {
                map[t]--;
                if (map[t] == 0) {
                    q.push(t);
                    res.push_back(t);
                }
            }
        }
        return res;
    }
};

class SolutionT630 {
public:
    int scheduleCourse(vector<vector<int>>& courses) {
        priority_queue<int> pq;
        sort(courses.begin(), courses.end(), [](vector<int>& a, vector<int> &b){ return a[1] < b[1];});
        int cur_time = 0, res = 0;
        for (auto course : courses) {
            cur_time += course[0];
            pq.push(course[0]);
            if (cur_time > course[1]) {
                cur_time -= pq.top(); pq.pop();
            }
        }
        return pq.size();
    }
};

class SolutionT490 {
public:
    vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}};
    bool hasPath(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination) {
        return helper(maze, start[0], start[1], destination[0], destination[1]); 
    }

    bool helper(vector<vector<int>>& maze, int i, int j, int di, int dj) {
        if (i == di && j == dj) return true;
        bool res = false; 
        int m = maze.size(), n = maze[0].size(); 
        maze[i][j] = -1; 
        for (auto dir : dirs) {
            int x = i, y = j;
            while ( x >= 0 && x < i && y >= 0 && y < j ) {
                x += dir[0], y += dir[1];
            }
            x -= dir[0], y -= dir[1];
            if (maze[x][y] != -1) 
                res |= helper(maze, x, y, di, dj);
        }
        return res;
    }
};

class SolutionT490 { 
public: 
    bool hasPath(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination) { 
        if (maze.empty() || maze[0].empty()) return true; 
        int m = maze.size(), n = maze[0].size(); 
        vector<vector<bool>> visited(m, vector<bool>(n, false)); 
        vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}}; 
        queue<pair<int, int>> q; 
        q.push({start[0], start[1]}); 
        visited[start[0]][start[1]] = true; 
        while (!q.empty()) { 
            auto t = q.front(); q.pop();
            int x = t.first, y = t.second;
            if (x == destination[0] && y == destination[1]) return true;
            for (auto dir : dirs) {
                int tx = x, ty = y;
                while (tx >= 0 && tx < m && ty >= 0 && ty < n && maze[tx][ty] == 0) { 
                    tx += dir[0]; ty += dir[1]; 
                } 
                tx -= dir[0]; ty -= dir[1]; 
                if (!visited[tx][ty]) {
                    visited[tx][ty] = true;
                    q.push({tx, ty});
                }
            }

        }
        return false;
    }
};

class SolutionT505 {
public:
    int shortestDistance(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination) {
        if (maze.empty() || maze[0].empty()) return -1; 
        int m = maze.size(), n = maze[0].size(), res = INT_MAX; 
        vector<vector<bool>> visited(m, vector<bool>(n, false)); 
        vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}}; 
        queue<vector<int>> q; 
        q.push({start[0], start[1], 0}); 
        visited[start[0]][start[1]] = true; 
        while (!q.empty()) { 
            auto t = q.front(); q.pop();
            int x = t[0], y = t[1], step = t[2];
            if (x == destination[0] && y == destination[1]) res = min(res, step);
            for (auto dir : dirs) {
                int tx = x, ty = y, ts = step;
                while (tx >= 0 && tx < m && ty >= 0 && ty < n && maze[tx][ty] == 0) { 
                    tx += dir[0]; ty += dir[1], ts++; 
                } 
                tx -= dir[0]; ty -= dir[1], ts--; 
                if (!visited[tx][ty]) {
                    visited[tx][ty] = true;
                    q.push({tx, ty, ts});
                }
            }
        }
        return res == INT_MAX ? -1 : res;
    }

    int shortestDistance(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination) {
        int m = maze.size(), n = maze[0].size();
        vector<vector<int>> dists(m, vector<int>(n, INT_MAX));
        vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}};
        queue<pair<int, int>> q;
        q.push({start[0], start[1]});
        dists[start[0]][start[1]] = 0;
        while (!q.empty()) {
            auto t = q.front(); q.pop();
            for (auto d : dirs) {
                int x = t.first, y = t.second, dist = dists[t.first][t.second];
                while (x >= 0 && x < m && y >= 0 && y < n && maze[x][y] == 0) {
                    x += d[0];
                    y += d[1];
                    ++dist;
                }
                x -= d[0];
                y -= d[1];
                --dist;
                if (dists[x][y] > dist) {
                    dists[x][y] = dist;
                    if (x != destination[0] || y != destination[1]) q.push({x, y});
                }
            }
        }
        return dists[destination[0]][destination[1]];
    }
};

class SolutionT505 { 
public: 
    int shortestDistance(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination) { 
        int m = maze.size(), n = maze[0].size(); 
        vector<vector<int>> dists(m, vector<int>(n, INT_MAX)); 
        vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}}; 
        auto cmp = [](vector<int> &a, vector<int> &b) {
            return a[2] > b[2];
        };
        priority_queue<vector<int>, vector<vector<int>>, decltype(cmp) > pq(cmp); 
        pq.push({start[0], start[1], 0});
        dists[start[0]][start[1]] = 0; 
        while (!pq.empty()) {
            auto t = pq.top(); pq.pop();
            for (auto dir : dirs) { 
                int x = t[0], y = t[1], dist = dists[t[0]][t[1]]; 
                while (x >= 0 && x < m && y >= 0 && y < n && maze[x][y] == 0) { 
                    x += dir[0]; 
                    y += dir[1]; 
                    ++dist; 
                } 
                x -= dir[0]; 
                y -= dir[1]; 
                --dist; 
                if (dists[x][y] > dist) { 
                    dists[x][y] = dist; 
                    if (x != destination[0] || y != destination[1]) pq.push({x, y, dist}); 
                } 
            } 
        }
        int res = dists[destination[0]][destination[1]]; 
        return (res == INT_MAX) ? -1 : res;
    }
};

class SolutionT973 {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        auto cmp = [](vector<int> &p, vector<int> & q) {
            return p[0] * p[0] + p[1] * p[1] < q[0] * q[0] + q[1] * q[1];
        };
        priority_queue<vector<int>, vector<vector<int>>, decltype(cmp)> pq;
        for (vector<int>& point : points) {
            pq.push(point);
            if (pq.size() > K) {
                pq.pop();
            }
        }
        vector<vector<int>> ans;
        while (!pq.empty()) {
            ans.push_back(pq.top());
            pq.pop();
        }
        return ans;
    }
private:
    struct compare {
        bool operator()(vector<int>& p, vector<int>& q) {
            return p[0] * p[0] + p[1] * p[1] < q[0] * q[0] + q[1] * q[1];
        }
    };
};

class Solution {
public:
    vector<int> numIslands2(int m, int n, vector<vector<int>>& positions) {
        vector<int> res; 
        int cnt = 0; 
        vector<int> roots(m * n, -1); 
        vector<vector<int>> dirs{{0, -1}, {-1, 0}, {0, 1}, {1, 0}};
        for (auto &pos : positions) {
            int id = pos[0] * m + pos[1];
            if (roots[id] != -1) {
                res.push_back(cnt);
                continue;
            }
            roots[id] = id;
            ++cnt;
            for (auto dir : dirs) {
                int x = pos[0] + dir[0], y = pos[1] + dir[1], cur_id = n * x + y; 
                if (x < 0 || x >= m || y < 0 || y >= n || roots[cur_id] == -1) continue; 
                int p = findRoot(roots, cur_id), q = findRoot(roots, id);
                if (p != q) {
                    roots[p] = q;
                    --cnt; 
                }
            }
            res.push_back(cnt);
        }
        return res;    
    }

    int findRoot(vector<int>& roots, int id) {
        return roots[id] == id ? id : findRoot(roots, id);
    }
};

class SolutionT323 {
public:
    int countComponents(int n, vector<vector<int>>& edges) {
        vector<int> roots(n, -1);
        for (int i = 0; i < n; i++) roots[i] = i;
        int res = n;
        for (auto edge : edges) {
            int a = edge[0], b = edge[1];
            int p = findRoot(roots, a), q = findRoot(roots, b);
            if (p != q) {
                roots[p] = q;
                res--;
            }
        }
        return res;
    }

    int findRoot(vector<int>& roots, int id) {
        return roots[id] == id ? id : findRoot(roots, roots[id]);
    }
};

class SolutionT694 {
public:
    vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}};
    int numDistinctIslands(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        set<vector<pair<int, int>>> res;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] != 1) continue;
                vector<pair<int, int>> v;
                helper(grid, i, j, i, j, v);
                res.insert(v);
            }
        }
        return res.size();
    }

    void helper(vector<vector<int>>& grid, int x0, int y0, int i, int j, vector<pair<int, int>>& v) {
        int m = grid.size(), n = grid[0].size();
        v.push_back({i - x0, j - y0});
        grid[i][j] *= -1;
        for (auto dir : dirs) {
            int tx = i + dir[0], ty = j + dir[1];
            if (tx < 0 || tx >= m || ty < 0 || ty >= n || grid[tx][ty] <= 0) continue;
            helper(grid, x0, y0, tx, ty, v);
        }
    }
};

class Solution { 
public: 
    /** 
     * @param grid: a chessboard included 0 (false) and 1 (true) 
     * @param source: a point 
     * @param destination: a point 
     * @return: the shortest path  
     */ 
    struct Point {
      int x;
      int y;
      Point() : x(0), y(0) {}
      Point(int a, int b) : x(a), y(b) {}
    };
  
    int shortestPath(vector<vector<bool>> &grid, Point &source, Point &destination) { 
        // write your code here 
        if (grid.empty() || grid[0].empty()) return -1; 
        int n = grid.size(), m = grid[0].size(); 
        if(source.x == destination.x && source.y == destination.y) 
            return 0; 
        vector<vector<int>> dirs = {{1,2},{1,-2},{-1,2},{-1,-2},{2,1},{2,-1},{-2,1},{-2,-1}}; 
        queue<Point> q({{source.x, source.y}}); 
        unordered_map<int, int> map{{source.x * m + source.y, 0}}; 
        while (!q.empty()) { 
            auto t = q.front(); q.pop(); 
            for (auto dir : dirs ) { 
                Point temp_p(t.x + dir[0], t.y + dir[1]); 
                if(!isValidPath(grid, temp_p)) 
                    continue; 
                if(map.count(temp_p.x * m + temp_p.y)) 
                    continue; 
                map[temp_p.x * m + temp_p.y] = map[t.x * m + t.y] + 1; 
                if(temp_p.x == destination.x && temp_p.y == destination.y) 
                    return map[temp_p.x * m + temp_p.y]; 
                q.push(temp_p); 
            } 
        }
        return -1; 
    } 
    bool isValidPath(vector<vector<bool>> grid, Point p){ 
        if(p.x < 0 || p.y < 0 || p.x >= grid.size() || p.y >= grid[0].size())
            return false; 
        if(grid[p.x][p.y] == true) 
            return false; 
        return true; 
    } 
};

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

class SolutionT133 {
public:
    Node* cloneGraph(Node* node) {
        unordered_map<Node*, Node*> map;
        return helper(map, node);
    }

    Node* helper(unordered_map<Node*, Node*>& m, Node* node) {
        if (!node) return nullptr;
        if (m.count(node)) return m[node];
        Node *clone = new Node(node->val);
        m[node] = clone;
        for (auto t : node->neighbors) {
            clone->neighbors.push_back(helper(m, t));
        }
        return clone;
    }
};

class SolutionT1306 {
public:
    unordered_set<int> set;
    bool canReach(vector<int>& arr, int start) {
        if (start >= 0 && start < arr.size() && set.insert(start).second) {
            return arr[start] == 0 || canReach(arr, start - arr[start]) || canReach(arr, start + arr[start]);
        }
    }
};

class Solution {
public:
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        vector<bool> visited(rooms.size(), -1);
        queue<int> q({0});
        while(!q.empty()) {
            int index = q.front(); q.pop();
            auto keys = rooms[index];
            for (auto key : keys) {
                if (visited[key] == true) continue;
                q.push(key);
            }
            visited[index] == true;
        }
        int flag = false;
        for (auto v : visited) if(!flag) return false;
        return true;
    }
};

class SolutionT261 { 
public: 
    bool validTree(int n, vector<pair<int, int>>& edges) { 
        vector<vector<int>> g(n, vector<int>()); 
        vector<bool> v(n, false); 
        for (auto a : edges) { 
            g[a.first].push_back(a.second); 
            g[a.second].push_back(a.first); 
        } 
        if (!dfs(g, v, 0, -1)) return false;
        for (auto a : v) if (!a) return false;
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
        vector<unordered_set<int>> g(n, unordered_set<int>()); 
        unordered_set<int> s{{0}}; 
        queue<int> q{{0}}; 
        for (auto a : edges) { 
            g[a.first].insert(a.second); 
            g[a.second].insert(a.first); 
        }
        while(!q.empty()) {
            int t = q.front(); q.pop();
            if (s.count(t)) return false;
            for (auto a : g[t]) {
                if (s.count(a)) return false;
                s.insert(a);
                q.push(a);
                g[a].erase(t);
            }
        }
        return true; 
    }
};

class SolutionT624 {
public:
    /**
     * @param s: a string
     * @param dict: a set of n substrings
     * @return: the minimum length
     */
    int minLength(string &s, unordered_set<string> &dict) {
        // write your code here
        int N = s.size(); 
        if (N == 0) return 0; 
        queue<string> q({s}); 
        unordered_set<string> visited; 
        int minLen = N;
        while (!q.empty()) {
            int t_size = q.size();
            for (int i = 0; i < t_size; i++) {
                string temp = q.front(); q.pop();
                for (int j = 0; j < temp.size(); j++) {
                    for (auto s : dict) {
                        int pos = temp.find(s);
                        while (pos != -1) {
                            string new_s = temp.substr(0, pos) + temp.substr(pos + s.size());
                            if (visited.find(new_s) == visited.end()) {
                                visited.insert(new_s);
                                q.push(new_s);
                                minLen = min(minLen, int(new_s.size()));
                            }
                            pos = temp.find(s, pos + 1);
                        }
                    }
                }
            }
        }
        return minLen;
    }

    vector<int> choosingShops(int cntProducts, vector<vector<int>> quantities, vector<vector<int>> costs, vector<vector<int>> meals) {
        priority_queue<int, vector<int>, greater<int>> pq;

    }

    int segmentsCovering(vector<vector<int>> seg) {
        sort(seg.begin(), seg.end());
        int n = seg.size(), res = 0, i = 0;
        while (i < n) {
            if (i == n - 1) {
                res++;
                i++;
                continue;
            }
            if ( seg[i][1] < seg[i + 1][0] ) {
                i++;
                res++;
            } else {
                int og_end = seg[i][1];
                i++;
                while (i < n && og_end >= seg[i][0]) {
                    i++;
                }
                res++;
            }
        }
        return res;
    }
};

class SolutionT1272 {
public:
    vector<vector<int>> removeInterval(vector<vector<int>>& intervals, vector<int>& toBeRemoved) {
        vector<vector<int>> res;
        for (auto interval : intervals) {
            if (interval[1] <= toBeRemoved[0] || interval[0] >= toBeRemoved[1]) {
                res.push_back(interval);
            } else if (interval[0] >= toBeRemoved[0] && interval[1] <= toBeRemoved[1]) {
                continue;
            } else {
                auto temp = interval;
                if (temp[0] < toBeRemoved[0]) {
                    if (temp[1] > toBeRemoved[1]) {
                        res.push_back({temp[0], toBeRemoved[0]});
                        res.push_back({toBeRemoved[1], temp[1]});
                        continue;
                    }
                    else {
                        temp[1] = toBeRemoved[0];
                        res.push_back(temp);
                        continue;
                    }
                } else {
                    temp[0] = toBeRemoved[1];
                    res.push_back(temp);
                    continue;
                }
            }
        }
        return res;
    }

    vector<vector<int>> removeInterval(vector<vector<int>>& intervals, vector<int>& toBeRemoved) {
        vector<vector<int>> res;
        auto start = toBeRemoved[0], end = toBeRemoved[1];
        for (auto &v : intervals) {
            if (v[1] <= start || v[0] >= end) res.push_back(v);
            else {
                if (v[0] < start) res.push_back({v[0], start});
                if (v[1] > end) res.push_back({end, v[1]});
            }
        }
        return res;
    }

};

class Solution {
public:
    int countArrangement(int n) {
        vector<bool> visited(n + 1, false);
        int res = 0;
        dfs(visited, 1, res);
        return res;
    }

    void dfs(vector<bool> &visited, int pos, int &res) {
        int N = visited.size() - 1;
        if (pos > N) {
            res++;
            return ;
        }
        for (int i = 1; i <= N; i++) {
            if (visited[i]) continue;
            if (visited[i] == false && (i % pos == 0 || pos % i == 0)) {
                visited[i] = true;
                dfs(visited, pos + 1, res);
                visited[i] = false;
            }
        }
    }
};

class SolutionT526 {
public:
    int countArrangement(int n) {
        vector<int> arr(n + 1);
        for (int i = 0; i <= n; i++) arr[i] = i;
        return dfs(arr, n);
    }

    int dfs(vector<int> &arr, int pos) {
        if (pos <= 1) return 1;
        int res = 0;
        for (int i = 1; i <= pos; i++) {
            if (pos % arr[i] == 0 || arr[i] % pos == 0) {
                swap(arr[i], arr[pos]);
                res += dfs(arr, pos - 1);
                swap(arr[i], arr[pos]);
            }
        }
        return res;
    }
};

class Solution {
public:
    int closestValue(TreeNode* root, double target) {
        TreeNode *p = root;
        int res = root->val;
        double diff = numeric_limits<double>::max();
        while (p) {
            if ( abs(double(p->val) - target) <= diff ) {
                res = p->val;
            }
            if (p->val < target) {
                p = p->right;
            } else if (p->val >= target) {
                p = p->left;
            }
        }
        return res;
    }
};

class Solution {
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
        while (root) {
            if (root->val >= p->val) root = root->left;
            else {
                if (!pre || root->val > pre->val) {
                    pre = root;
                }
                root = root->right;
            }
        }
        return pre;
    }
};

int binary(vector<int> a, int left, int right, int target) {
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (a[mid] < target) left = mid + 1;
        else right = mid;
    }
    return right;
}

int maxArithmeticLength(vector<int> a, vector<int> b) {
    int res = -1, na = a.size(), nb = b.size();
    int max_gap = a[1] - a[0];
    for (int i = 1; i <= max_gap; i++) {
        int cur_aidx = 0, cur_bidx = 0, count = 0, cur_val = a[0];
        bool flag = true;
        while (cur_aidx < na) {
            if (cur_aidx == na - 1) break;
            if (a[cur_aidx] + i == a[cur_aidx + 1] || cur_val + i == a[cur_aidx + 1]) cur_aidx++;
            else {
                cur_val = a[cur_aidx];
                int t_pos = binary(b, cur_bidx, nb, cur_val + i);
                if (t_pos >= nb || b[t_pos] != cur_val + i) {
                    flag = false;
                    break;
                }
                cur_bidx = t_pos;
                cur_val = b[t_pos];
                count++;
            }
        }
        if (flag && a.size() + count > res) res = a.size() + count;
    }
    return res;
}

class Solution {
public:
    /**
     * @param root: the root of binary tree
     * @return: the root of the minimum subtree
     */
    TreeNode * findSubtree(TreeNode * root) {
        // write your code here
        help(root);
        return res;
    }

    int help(TreeNode* root) {
        if (!root) return 0;

        int sum = help(root->left) + help(root->right) + root->val;
        if (sum <= minSum) {
            minSum = sum;
            res = root;
        }
        return sum;
    }

    bool isPal(string s, int l, int r) {
        while (l <= r) {
            if (s[l++] != s[r--]) return false;
        }
        return true;
    }

    bool constructorNames(string className, string methodName) {
        unordered_map<char, int> map1;
        unordered_set<char> set1;
        multiset<int> set11;
        unordered_map<char, int> map2;
        unordered_set<char> set2;
        multiset<int> set22;
        if (className.size() != methodName.size()) return false;
        for (int i = 0; i < className.size(); i++) {
            map1[className[i]]++;
            if (set1.count(className[i]) == 0) set1.insert(className[i]);
        }
        for (int i = 0; i < methodName.size(); i++) {
            map2[methodName[i]]++;
            if (set2.count(methodName[i]) == 0) set2.insert(methodName[i]);
        }
        if (set1 != set2) return false;
        for (auto t : map1) {
            set11.insert(t.second);
        }
        for (auto t : map2) {
            set22.insert(t.second);
        }
        if (set11 != set22) return false;
        return true;
    }

private:
    TreeNode* res = nullptr;
    int minSum = INT_MAX;
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
            if(visited[i] == 1 || (i > 0 && nums[i] == nums[i-1] && visited[i-1] == 0)) continue;
            visited[i] = 1;
            out.push_back(nums[i]);
            backtracking(res, out, nums, k + 1, visited);
            out.pop_back();
            visited[i] = 0;
        }
        return ;
    }

    vector<bool> boundedRatio(vector<int> a, int l, int r) {
        vector<bool> res(a.size(), false);
        for (int i = 0 ; i < a.size() ; i++) {
            double div = a[i] / (i + 1);
            if (l <= div && div <= r && (a[i] % (i + 1)) == 0) res[i] = true;
        }
        return res;
    }

    bool isInc(vector<int> num) {
        for (int i = 0; i < num.size() - 1; i++) {
            if (num[i] >= num[i + 1]) return false;
        }
        return true;
    }

    bool makeIncreasing(vector<int> nums) {
        int idx = 0, count = 0;
        for (int i = 0; i < nums.size() - 1; i++) {
            if (nums[i] >= nums[i + 1]) {
                idx = i;
                count++;
            }
        }
        if (count == 0) return true;
        if (count > 1) return false;
        int org_num = nums[idx];
        string t = to_string(nums[idx]);
        for (int i = 0; i < t.size(); i++) {
            string cur = t;
            swap(cur[i], cur[(i + 1) % t.size()]);
            while (cur[0] == '0') cur = cur.substr(1);
            cout << cur + " first " << endl;
            int newNum = stoi(cur);
            nums[idx] = newNum;
            if (isInc(nums)) return true;
            nums[idx] = org_num;
        }
        idx = idx + 1;
        if (idx < nums.size()) {
            org_num = nums[idx ];
            string t = to_string(nums[idx]);
            for (int i = 0; i < t.size(); i++) {
                string cur = t;
                swap(cur[i], cur[(i + 1) % t.size()]);
                while (cur[0] == '0') cur = cur.substr(1);
                cout << cur + " second " << endl;
                int newNum = stoi(cur);
                nums[idx] = newNum;
                if (isInc(nums)) return true;
                nums[idx] = org_num;
            }
        }
        return false;
    }

    string mergeStrings(string s1, string s2) {
        unordered_map<char, int> set1, set2;
        for (auto a : s1) set1[a]++;
        for (auto a : s2) set2[a]++;
        int m = s1.size(), n = s2.size(), p = 0, q = 0;
        string res = "";
        while (p < m && q < n) {
            if (set1[s1[p]] == set2[s2[q]]) {
                if (s1[p] < s2[q]) {
                    res += s1[p++];
                }
                else {
                    res += s2[q++]
                }
            } else if (set1[s1[p]] < set2[s2[q]]){
                res += s1[p++];
            } else {
                res += s2[q++];
            }
        }
        while (p < m) res += s1[p++];
        while (q < n) res += s2[q++];
        return res;
    }

    long long countDecreasingSubarrays(vector<int> arr) {
        for (int i = 0; i < arr.size(); i++) {
            for (int j = i + 1; j < arr.size(); j++) {

            }
        }
        vector<long long> dp(arr.size(), 1);
        dp[0] = 1;
        for (int i = 1; i < arr.size(); i++) {
            if (arr[i] < arr[i - 1]) {
                dp[i] = dp[i] + dp[i-1];
            }
        }
        long long res = 0;
        for (int i = 0; i < arr.size(); i++) {
            res += dp[i];
        }
        return res;
    }
};

int dirX[8] = {1, 0, -1, 0, 1, 1, -1, -1};
int dirY[8] = {0, -1, 0, 1, 1, -1, 1, -1};

bool check(int curX, int curY, int X, int Y) {
    if (curX <0 || curX >= X || curY < 0 || curY >= Y) return false;
    return true;
}

void infect(vector<vector<bool>> &field, vector<vector<int>> &res, int x, int y) {
    if (res[x][y] >= 0) return ;
    int m = field.size(), n = field[0].size();
    int near = 0;
    for (int i = 0; i < 8; i++) {
        int curX = x + dirX[i], curY = y + dirY[i];
        if (check(curX, curY, m, n) && field[curX][curY] == true) {
            near++;
        }
    }
    if (near > 0) {
        res[x][y] = near;
        return ;
    } else {
        res[x][y] = 0;
        for (int i = 0; i < 8; i++) {
            int curX = x + dirX[i], curY = y + dirY[i];
            if (check(curX, curY, m, n)){
                infect(field, res, curX, curY);
            }
        }
    }
}

vector<vector<int>> minesweeperClick(vector<vector<bool>> field, int x, int y) {
    int m = field.size(), n = field[0].size();
    vector<vector<int>> res(m, vector<int> (n, -1));
    infect(field, res, x, y);
    return res;
}

long long subarraysCountBySum(vector<int> arr, int k, long long s) {
    int n = arr.size();
    vector<long long> sum(n, 0);
    unordered_map<long long, vector<int>> map;
    map[0] = {-1};
    for (int i = 0; i < n; i++) {
        sum[i] = arr[i] + (i > 0 ? sum[i-1] : 0);
        map[sum[i]].push_back(i);
    }
    long long res = 0;
    for (int i = 0; i < n; i++) {
        if (i == 0 && sum[i] == s) {
            res++;
            continue;
        }
        long long cur = sum[i];
        if (map.count(cur - s) == 0) continue;
        auto pos = map[cur - s];
        for (auto p : pos) {
            if (p > i) break;
            cout << p<< endl;
            cout << i << endl;
            if (i - p + 1 <= k) res++;
        }
    }
    return res;
}

class SolutionCS {
    long long subarraysCountBySum(vector<int> arr, int k, long long s) {
        unordered_map<long long, deque<int>> map;
        long long sum = 0, res = 0;
        map[0].push_back(-1);
        for (int i = 0; i < arr.size(); i++) {
            sum += arr[i];
            if (map.count(sum - s)) {
                while (i - map[sum - s].front() > k) map[sum - s].pop_front();
                res += map[sum - s].size();
            }
            // 判空 while (i - map[sum].front() > k) map[sum].pop_front();
            map[sum].push_back(i);
        }
        return res;
    }
};

struct cmp {
    bool operator() (vector<int> &a, vector<int> &b) {
        if (a[0] == b[0]) {
            if (a[1] == b[1]) {
                return a[2] > b[2];
            }
            return a[1] > b[1];
        }
        return a[0] > b[0];
    }
};

vector<vector<int>> meanAndChessboard(vector<vector<int>> matrix, vector<vector<int>> queries) {
    int m = matrix.size(), n = matrix[0].size();
    // auto cmp = [](vector<int> &a, vector<int> &b) {
    //     return a[0] > b[0];
    // };
    vector<vector<int>> res = matrix;
    
    priority_queue<vector<int>, vector<vector<int>>, cmp> pqBlack;
    priority_queue<vector<int>, vector<vector<int>>, cmp> pqWhite;
    for (int i = 0; i < m ; i++) {
        for (int j = 0; j < n; j++) {
            if (i + j % 2== 0) pqWhite.push({matrix[i][j], i, j});
            else pqBlack.push({matrix[i][j], i, j});
        }
    }
    for (auto query : queries) {
        int kBlack = query[0], kWhite = query[1];
    }
}

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class SolutionT1019 {
public:
    vector<int> nextLargerNodes(ListNode* head) {
        vector<int> nums, res;
        stack<int> st;
        int pos = 0;
        while (head) {
            int curVal = head->val;
            nums.push_back(curVal);
            while (!st.empty() && curVal > nums[st.top()]) {
                res[st.top()] = curVal;
                st.pop();
            }
            st.push(pos);
            pos++;
            res.resize(pos);
            head = head->next;
        }
        return res;
    }
};