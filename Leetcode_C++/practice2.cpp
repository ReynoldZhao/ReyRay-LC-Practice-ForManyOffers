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

class SolutionT1756 {
public:
    string modifyString(string s) {
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '?') {
                for (s[i] = 'a'; s[i] <= 'c'; ++s[i]) {
                    if ( (i==0 || s[i-1] != s[i]) && (i == s.size() - 1 || s[i] != s[i+1]))
                        break;
                }
            }
        }
        return s;
    }
};

class SolutionT1577 {
public:
    int helper(vector<int>& A, vector<int>& B) {
        unordered_map<int, int> map;
        int ans = 0;
        for (auto b : B) map[b]++;
        for (auto a : A) {
            long target = (long) a * a;
            for (auto &[b, cnt] : map) {
                if (target % b != 0 && map[target / b] == 0) continue;
                if (a == b) ans+= cnt * (cnt - 1)
                else ans += cnt * map[target / b]
                //不必在乎先后，因为存在就行了，又不让你得出位置
                //每次结果成立的那对pair，不管是b，还是target/b，总让在前面的那个做j
                //只是因为b会来一遍，遍历到target / b又会来一遍，所以/2
            }
        }
        return ans / 2;
    }

    int numTriplets(vector<int>& nums1, vector<int>& nums2) {
        return helper(nums1, nums2) + helper(nums2, nums1);
    }
};

class SolutionT1578 {
public:
    int minCost(string s, vector<int>& cost) {
        int sum = 0;
        for (int i = 0; i < s.size(); i++) {
            char temp = s[i];
            int curCost = cost[i];
            while(i <= s.size() - 2 && s[i+1] == temp) {
                sum += min(curCost, cost[i+1]);
                temp = s[i+1];
                i++;
                curCost = max(curCost, cost[i]);
            }
        }
        return sum;
    }
};

class SolutionT1583 {
public:
//大模拟
    int unhappyFriends(int n, vector<vector<int>>& pref, vector<vector<int>>& pairs) {
        vector<unordered_map<int, int>> m(n); //就是要知道每个人对每个人的距离
        vector<int> dist(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < pref[i].size(); ++j)
                m[i][pref[i][j]] = j; //每个人所在的pref距离可以直接用vec（map）得到
        }
        for (auto &p : pairs) {
            dist[p[0]] = m[p[0]][p[1]]; //第 i 个人，他配对的人，所在的pref距离
            dist[p[1]] = m[p[1]][p[0]];
        }
        int res = 0;
        for (int i = 0; i < n; ++i) { //n个人
            for (int d = 0; d < dist[i]; ++d) { //假设对于x，只用遍历他配对的人所在距离，前面的人
                int j = pref[i][d]; //按上面那个距离遍历，在pref遍历找到更喜欢的人
                for (int d1 = 0; d1 < dist[j]; ++d1) { //x更喜欢的人，u1，他的配对的人的距离，遍历
                    if (i == pref[j][d1]) { //如果出现了x，那么x更不开心
                        ++res;
                        d = d1 = n;
                    }
                }
            }
        }
        return res;
    }
};

class SolutionT1584 {
public:
//prims
    int minCostConnectPoints(vector<vector<int>>& ps) {
        int n = ps.size(), res = 0, i = 0, connected = 0;
        vector<bool> visited(n, false);
        visited[0] = true;
        priority_queue<pair<int, int> > pq;
        while(connected < n-1) {
            visited[i] = true;
            for (int j = 0; j < n; j++) {
                if (!visited[j]) {
                    pq.push({-(abs(ps[i][0] - ps[j][0]) + abs(ps[i][1] - ps[j][1])), j});
                }
            }
            // 这一步非常妙
            //prim算法，第一个点，其他所有点到他的距离，（dis，j）放入pq大顶堆中
            //选取一个最小的距离，并且该点不在已划分中，加入该点到已划分中
            //下一步，是更新其他所有点，到已划分中所有点的最短距离，选择并加入
            //这一步while循环，不浪费之前已经算了的，未加入的点到已加入点的距离
            //一直剔除j已经在已划分的点在pq中占有的值，将其pop，（因为是前一步选择最短距离时没有用上的距离）
            //删除了两个点都在已划分的距离，找到一个最小的，j不在已划分中的距离，将其加入
            while(visited[pq.top().second]) pq.pop();
            auto it = pq.top(); pq.pop();
            res -= it.first;
            i = it.second;
            visited[it.first] = true;
        }
        return res;
    }
};

class SolutionT309 {
public:
    int maxProfit(vector<int>& prices) {
        int dp_i_0 = 0, dp_i_1 = INT_MIN, dp_pre_0 = 0;
        for (int i = 0 ; i < prices.size(); i++) {
            int temp = dp_i_0;
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i]);
            dp_i_1 = max(dp_i_1, dp_pre_0 - prices[i]);
            dp_pre_0 = temp;
        }
        return dp_i_0;
    }
};

class SolutionT1535 {
public:
    int getWinner(vector<int>& arr, int k) {
        int cur = arr[0], win = 0;
        for (int i = 0; i < arr.size(); i++) {
            if (arr[i] > cur) {
                cur = arr[i];
                win = 0;
            }
            win++;
            if(win == k) return cur;
        }
        return cur;
    }
};

class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> map;
        for (const auto &num:nums) map[num]++;
        auto cmp = [](const pair<int, int> &a, const pair<int, int> &b){
            return a.first < b.first;
        }; //less maxheap
        priority_queue<pair<int, int>, vector<pair<int,int> >, decltype(cmp)> pq(cmp);
        for (auto it:map) {
            pq.push(make_pair(it.second, it.first));
        }
        vector<int> res;
        for (int i = 0; i < k; i++) {
            auto temp = pq.top(); pq.pop();
            res.push_back(temp.second);
        }
        return res;
    }

    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> map;
        for (const auto &num:nums) map[num]++;
        auto cmp = [](const pair<int, int> &a, const pair<int, int> &b) {
            return a.first < b.first;
        };
        priority_queue<pair<int, int>, vector<pair<int, int> >, decltype(cmp)> pq(cmp);
        for (auto it : map) {
            pq.push(make_pair(it.second, it.first));
        }
        vector<int> res;
    }
};

class SolutionT845 {
public:
    int longestMountain(vector<int>& arr) {
        int n = arr.size() - 1, left = 0, right = 0, length = 1;
        int pos = 0, prepos = n;
        while(prepos == pos) {
            prepos = pos;
            pos = peakIndexInMountainArray(arr, left, right);
            if (pos != n || (pos == n && arr[pos] < arr[pos - 1])) {
                int tempLength = findMaxLength(arr, pos);
                length = max(length, tempLength);
            }
        }
        return length;
    }

    int findMaxLength(vector<int>& arr, int pos) {
        int leftEnd = pos, rightEnd = pos, n = arr.size() - 1;
        while(leftEnd - 1 >= 0 && arr[leftEnd - 1] < arr[leftEnd]) leftEnd--;
        while(rightEnd + 1 <= n && arr[rightEnd + 1] < arr[rightEnd]) rightEnd++；
        return rightEnd - leftEnd + 1;
    }

    int peakIndexInMountainArray(vector<int>& A, int left, int right) {
		while (left < right) {
			int mid = left + (right - left) / 2;
			if (A[mid] < A[mid + 1]) left = mid + 1;
			else right = mid;
		}
        return right;
    }


    //双数组up down，记录以该点为终点的最长递增、递减的长度；
    int longestMountain(vector<int>& arr) {
        int res = 0, n = arr.size();
        vector<int> up(n), down(n);
        for (int i = n - 2; i >= 0; --i) {
            if (arr[i] > arr[i + 1]) {
                down[i] = down[i] + 1;
            }
        }
        for (int i = 1; i < n; ++i) {
            if (A[i] > A[i - 1]) up[i] = up[i - 1] + 1;
            if (up[i] > 0 && down[i] > 0) res = max(res, up[i] + down[i] + 1);
        }
        return res;
    }
};

class SolutionT986 {
public:
    vector<vector<int>> intervalIntersection(vector<vector<int>>& firstList, vector<vector<int>>& secondList) {
        int m = firstList.size(), n = secondList.size(), i = 0, j = 0;
        vector<vector<int>> res;
        while ( i < m && j < n) {
            if (firstList[i][1] < secondList[j][0]) {
                i++;
            } else if (secondList[j][1] < firstList[i][0]) {
                j++;
            } else {
                if (secondList[j][0] <= firstList[i][1] && secondList[j][1] >=firstList[i][1]) {
                    res.push_back({max(secondList[j][0], firstList[i][0]), min(firstList[i][1], secondList[j][1])});
                    i++;
                } else if (firstList[i][0] <= secondList[j][1] && firstList[i][1] >= secondList[j][1]) {
                    res.push_back({max(secondList[j][0], firstList[i][0]), min(firstList[i][1], secondList[j][1])});
                    j++;
                }
            }
        }
        return res;
    }
};

class SolutionT3{
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<int, int> map;
        int left = 0, res = 0;
        for (int i = 0; i < s.size(); i++) {
            if (map.count(s[i]) && map[s[i]] > left) {
                left = map[s[i]];
            }
            map[s[i]] = i;
            res = max(res, i - left);
        }
        return res;
    }
};

class SolutionT159 {
public:
    int lengthOfLongestSubstringTwoDistinct(string s) {
        unordered_map<int, int> map;
        int left = 0, res = 0;
        for (int i = 0; i < s.size(); i++) {
            map[s[i]]++;
            while(map.size() > 2) {
                if (--map[s[left]] == 0) map.erase(s[left]);
                left++;
            }
            res = max(res, i - left + 1);
        }
    }
};

class SolutionT713 {
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        int temp_product = 1, left = 0, res = 0;
        for (int i = 0; i < nums.size(); i++) {
            temp_product = temp_product * nums[i];
            while(temp_product >= k && left <= i) {
                temp_product /= nums[left];
                left++;
            }
            res += i - left + 1;
        }
        return res;
    }
};

class SolutionT395 {
public:
    int longestSubstring(string s, int k) {
        int res = 0, i = 0, n = s.size();
        while (i + k >= n) {
            int m[26] = {0}, mask = 0, max_idx = i;
            for (int j = i; j < n; j++) {
                int val = s[j] - 'a';
                m[val]++;
                if (m[val] < k) mask |= (1 << val);
                else mask &= (~(1 << val));
                if (mask == 0) {
                    res = max(res, j - i + 1);
                    max_idx = j;
                }
            }
            i = max_idx = 1;
        }
        return res;
    }
};

class SolutionT395 {
public:
    int longestSubstring(string s, int k) {
        int res = 0, n = s.size();
        for (int cnt = 1; cnt <= 26; cnt++) {
            int start = 0, i = 0, uniqueCnt = 0; 
            vector<int> charCnt(26);
            while (i < n) {
                bool isValid = true;
                if (charCnt[s[i] - 'a']++ == 0) uniqueCnt++;
                while(uniqueCnt > cnt) {
                    if (--charCnt[s[start++] - 'a'] == 0) --uniqueCnt;
                }
                for (int j = 0; j < 26; ++j) {
                    if (charCnt[j] > 0 && charCnt[j] < k) isValid = false;
                }
                if (isValid) res = max(res, i - start);
            }
        }
        return res;
    }
};

class SolutionT480 {
public:
    vector<double> medianSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        vector<double> res;
        multiset<double> ms(nums.begin(), nums.begin() + k);
        auto mid = next(ms.begin(), k / 2);
        for (int i = k; i < n; i++) {
            res.push_back((*mid + *prev(mid, 1 - k%2 )) / 2);
            if (i == n) return res;
            ms.insert(nums[i]);
            if (nums[i] < *mid) --mid;
            if (nums[i - k] <= *mid) ++mid;
            ms.erase(ms.lower_bound(nums[i - k]));
        }
    }
};

class SolutionT480 {
public:
    vector<double> medianSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        vector<double> res;
        multiset<double> ms(nums.begin(), nums.begin() + k);
        auto mid = next(ms.begin(), k / 2);
        for (int i = k; i < n; i++) {
            res.push_back((*mid + *prev(mid, 1 - k%2 )) / 2);
            if (i == n) return res;
            ms.insert(nums[i]);
            if (nums[i] < *mid) --mid;
            if (nums[i - k] <= *mid) ++mid;
            ms.erase(ms.lower_bound(nums[i - k]));
        }
    }
};

class Solution {
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
        int n1 = s1.size(), n2 = s2.size();
        vector<int> m1(128), m2(128);
        for (int i = 0; i < n1; i++) {
            ++m1[s1[i]];
            ++m2[s2[i]];
        }
        if (m1 == m2) return true;
        for (int i = n1; i < n2; i++) {
            ++m2[s2[i]];
            --m2[s2[i - n1]];
            if(m1 == m2) return true;
        }
        return false;
    }
};

class SolutionT567 {
public:
    bool checkInclusion(string s1, string s2) {
        int n1 = s1.size(), n2 = s2.size(), cnt = n1, left = 0;
        unordered_map<int, int> map;
        for (auto s : s1) map[s]++;
        for (int i = 0 ; i < n2; i++) {
            if (map[s2[i]]-- > 0) --cnt;
            while (cnt == 0) {
                if (i - left + 1 == n1) return true;
                if (++map[s2[left++]] > 0) ++cnt;
            }
        }
        return false;
    }
};

class SolutionT727 {
public:
    string checkInclusion(string s1, string s2) {
        int n1 = s1.size(), n2 = s2.size(), cnt = n1, left = 0;
        unordered_map<int, int> map;
        int temp_len = INT_MAX;
        string res = "";
        for (auto s : s1) map[s]++;
        for (int i = 0 ; i < n2; i++) {
            if (map[s2[i]]-- > 0) --cnt;
            while (cnt == 0) {
                if (i - left + 1 < temp_len) {
                    temp_len = i - left + 1;
                    res = s2.substr(left, i - left + 1);
                }
                if (++map[s2[left++]] > 0) ++cnt;
            }
        }
        return res;
    }
};

class Solution {
public:
    string minWindow(string S, string T) {
        int m = S.size(), n = T.size(), start = -1, minLen = INT_MAX;
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, -1));
        for (int i = 0; i <= m; ++i) dp[i][0] = i;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= min(i, n); j++) {
                dp[i][j] = (S[i] == T[j]) ? dp[i-1][j-1] : dp[i-1][j];
            }
            if (dp[i][n] != -1) {
                int len = i - dp[i][n];
                if (minLen > len) {
                    minLen = len;
                    start = dp[i][n];
                }
            }
        }
        return (start != -1) ? S.substr(start, minLen) : "";
    }
};

class SolutionT727 {
public:
    string minWindow(string S, string T) {
        int m = S.size(), n = T.size(), start = -1, minLen = INT_MAX, i = 0, j = 0;
        while (i < m) {
            if (S[i] == T[j]) {
                if (++j == n) {
                    int end = i + 1;
                    while (--j >= 0) {
                        while (S[i--] != T[j]);
                    }
                    ++i; ++j;
                    if (end - i < minLen) {
                        minLen = end - i;
                        start = i;
                    }
                }
            }
            ++i;
        }
        return (start != -1) ? S.substr(start, minLen) : "";
    }
};

class MedianFinder {
public:
    /** initialize your data structure here. */
    MedianFinder() {
        
    }
    
    void addNum(int num) {
        small.push(num);
        if (small.size() - large.size() > 1) {
            large.push(small.top());
            small.pop();
        }
    }
    
    double findMedian() {
        return small.size() > large.size() ? small.top() : 0.5 *(small.top() - large.top());
    }

private:
    priority_queue<int, vector<int>, less<int>> large;
    priority_queue<int, vector<int>, greater<int>> small;
};

class SummaryRanges {
public:
    /** Initialize your data structure here. */
    SummaryRanges() {
        
    }
    
    void addNum(int val) {
        vector<vector<int, int>> res;
        vector<int> newInterval{val, val};
        int cur = 0;
        for (auto interval : intervals) {
            if (newInterval[1] + 1 < interval[0]) {
                res.push_back(interval);
            } else if (newInterval[0] > interval[1] + 1) {
                res.push_back(interval);
                ++cur;
            } else {
                newInterval[0] = min(newInterval[0], interval[0]);
                newInterval[1] = max(newInterval[1], interval[1]);
            }
        }
        res.insert(res.begin() + cur, newInterval);
        intervals = res;
    }
    
    vector<vector<int>> getIntervals() {
        return intervals;
    }

private:
    vector<vector<int>> intervals;
};


class MovingAverage {
public:
    MovingAverage(int size) {
        this->size = size;
        sum = 0;
    }
    
    double next(int val) {
        if (q.size() >= size) {
            sum -= q.front(); q.pop();
        }
        q.push(val);
        sum += val;
        return sum / q.size();
    }
    
private:
    queue<int> q;
    int size;
    double sum;
};

class SummaryRanges {
public:
    SummaryRanges() {}
    
    void addNum(int val) {
        vector<int> newInterval{val, val};
        int i = 0, overlap = 0, n = intervals.size();
        for (; i < n; ++i) {
            if (newInterval[1] + 1 < intervals[i][0]) break; 
            if (newInterval[0] <= intervals[i][1] + 1) {
                newInterval[0] = min(newInterval[0], intervals[i][0]);
                newInterval[1] = max(newInterval[1], intervals[i][1]);
                ++overlap;
            }
        }
        if (overlap > 0) {
            intervals.erase(intervals.begin() + i - overlap, intervals.begin() + i);
        }
        intervals.insert(intervals.begin() + i - overlap, newInterval);
    }
    vector<vector<int>> getIntervals() {
        return intervals;
    }
private:
    vector<vector<int>> intervals;
};

class SolutionT53 {
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

    int maxSubArray(vector<int>& nums) {
        int res = INT_MIN, temp_sum = 0;
        for (int i = 0; i < nums.size(); i++) {
            temp_sum += nums[i];
            res = max(res, temp_sum);
            if (temp_sum < 0) temp_sum = 0;
        }
        return res;
    }
};

class Solution {
public:
    int maxSubArrayLen(vector<int>& nums, int k) {
        if (nums.empty()) return 0;
        unordered_map<int, vector<int>> map;
        int temp_sum = 0, res = INT_MIN;
        for (int i = 0; i < nums.size(); i++) {
            temp_sum += nums[i];
            map[temp_sum].push_back(i);
        }
        for (auto it : map) {
            if (it.first == k) res = max(res, it.second.back() + 1);
            else if (m.find(it.first - k) != m.end()) {
                res = max(res, it.second.back() - m[it.first - k][0]);
            }
        }
        return res;
    }
};

class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> map;
        int temp_sum = 0, res = 0;
        for (int i = 0; i < nums.size(); i++) {
            temp_sum+=nums[i];
            map[temp_sum]++;
            if (map.count(abs(k - temp_sum))) {
                res += map[abs(k - temp_sum)];
            }
        }
        return res;
    }
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

// class Codec {
// public:

//     // Encodes a tree to a single string.
//     string serialize(TreeNode* root) {
//         ostringstream out;
//         serialize(root, out);
//         return out.str();
//     }

//     // Decodes your encoded data to tree.
//     TreeNode* deserialize(string data) {
//         istringstream in(data);
//         return deserialize(in);
//     }
// private:
//     void serialize(TreeNode* root, ostringstream &out) {
//         if (root) {
//             out << root->val << " ";
//             serialize(root->left, out);
//             serialize(root->right, out);
//          } else {
//              out << "# ";
//          }
//     }
//     TreeNode* deserialize(istringstream &in) {
//         string temp;
//         in >> temp;
//         if (temp == "#") return nullptr;
//         TreeNode* root = new TreeNode(stoi(temp));
//         root->left = deserialize(in);
//         root->right = deserialize(in);
//         return root;
//     }
// };

class Codec {
public:
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        ostringstream out;
        queue<TreeNode*> q;
        if (root) q.push(root);
        while (!q.empty())
        {
            TreeNode *t = q.front(); q.pop();
            if (t) {
                out << t->val << ' ';
                q.push(t->left);
                q.push(t->right);
            } else {
                out << "# ";
            }

        }
        return out.str();
    }
    TreeNode* deserialize(string data) {
        if (data.empty()) return nullptr;
        istringstream in(data);
        queue<TreeNode*> q;
        string val;
        in >> val;
        TreeNode* root = new TreeNode(stoi(val)), *cur = res;
        q.push(res);
        while(!q.empty()) {
            TreeNode *t = q.front(); q.pop();
            if (!(in >> val)) break;
            if (val != "#") {
                cur = new TreeNode(stoi(val));
                q.push(cur);
                t->left = cur;
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
};


struct DirectedGraphNode {
    int label;
    vector<DirectedGraphNode *> neighbors;
    DirectedGraphNode(int x) : label(x) {};
};


class Solution {
public:
    /**
     * @param graph: A list of Directed graph node
     * @return: Any topological order for the given graph.
     */
    vector<DirectedGraphNode*> topSort(vector<DirectedGraphNode*> graph) {
        // write your code here
        vector<DirectedGraphNode*> ret;
        if (graph.empty()) return ret;

        unordered_map<DirectedGraphNode*, int> mymap;
        for (auto& node : graph) {
            for (auto& neighbor : node->neighbors) {
                mymap[neighbor]++;
            }
        }

        queue<DirectedGraphNode*> q;
        for (auto& node : graph) {
            if (mymap.count(node) == 0) {
                q.push(node);
                ret.push_back(node);
            }
        }

        while (!q.empty()) {
            auto& cur = q.front(); q.pop();
            for (auto& next : cur->neighbors) {
                mymap[next]--;
                if (mymap[next] == 0) {
                    q.push(next);
                    ret.push_back(next);
                }
            }
        }

        return ret;
    }
};

class SolutionT210 {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> graph(numCourses, vector<int>());
        vector<int> visit(numCourses);
        for (auto a : prerequisites) {
            graph[a[1]].push_back(a[0]);
        }
        for (int i = 0; i< numCourses; i++) {
            if (!canFinishDFS(graph, visit, i)) return false;
        }
        return true;
    }

    bool canFinishDFS(vector<vector<int>>& graph, vector<int>& visit, int i) {
        if (visit[i] == -1) return false;
        if (visit[i] == 1) return true;

    }
};

class Solution {
public:
    string alienOrder(vector<string>& words) {
        set<pair<int, int>> st;
        unordered_set<char> ch;
        vector<int> in(256);
        queue<char> q;
        string res;
        for (auto a : words) ch.insert(a.begin(), a.end());
        for (int i = 0; i < words.size(); i++) {
            int mn = min(words[i].size(), words[i+1].size()), j = 0;
            for (; j < mn; j++) {
                if (words[i][j] != words[i + 1][j]) {
                    st.insert(make_pair(words[i][j], words[i + 1][j]));
                    break;
                }
            }
            if (j == mn && words[i].size() > words[i + 1].size()) return "";
        }
        for (auto a : st) in[a.second];
        for (auto a : ch) {
            if (in[a] == 0) {
                q.push(a);
                res += a;
            } 
        }
        while (!q.empty()) {
            char c = q.front(); q.pop();
            for (auto a : st) {
                if (a.first == c) {
                    --in[a.second];
                    if (in[a.second] == 0) {
                        q.push(a.second);
                        res += a.second;
                    }
                }
            }
        }
        return res.size() == ch.size() ? res : "";
    }
};

class SolutionT444 {
public:
    bool sequenceReconstruction(vector<int>& org, vector<vector<int>>& seqs) {
        int n = org.size(), cnt = n - 1;
        unordered_map<int, int> map;
        for (int i = 0; i < n; i++) {
            map[org[i]] = i;
        }
        vector<int> flag(n, 0);
        bool res = false;
        for (auto seq : seqs) {
            for (int i = 0; i < seq.size(); i++) {
                if (seq[i] <= 0 || seq[i] > n) return false;
                if (i == 0 ) continue;
                if (map[seq[i - 1]] >= map[seq[i]]) return false;
                res = true;
                if (flag[seq[i]] == 0 && map[seq[i-1]] + 1 = map[seq[i]]) {
                    flag[seq[i]] = 1;
                    --cnt;
                }
            }
        }
        return cnt == 0 && res;
    }
};


//BFS
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        if (grid.empty() || grid[0].empty()) return 0;
        int m = grid.size(), n = grid[0].size(), res = 0;
        vector<vector<bool>> visited(m, vector<bool>(n));
        vector<int> dirX{-1, 0, 0, 1}, dirY{0, 1, -1, 0};
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!visited[i][j] && grid[i][j] == 1) {
                    ++res;
                    queue<int> q{{i * n + j}};
                    while(!q.empty()) {
                        int t = q.front(); q.pop();
                        for (int k = 0; k < 4; ++k) {
                            int x = t / n + dirX[k], y = t % n + dirY[k];
                            if (x < 0 || x >= m || y < 0 || y >= n || grid[x][y] == '0' || visited[x][y]) continue;
                            visited[x][y] = true;
                            q.push(x * n + y);
                        }
                    }
                }
            }
        }
    }
};

//DFS
class SolutionT490 {
public:
    vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}};
    bool hasPath(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination) {int m = maze.size(), n = maze[0].size();
        return helper(maze, start[0], start[1], destination[0], destination[1]);
    }
    bool helper(vector<vector<int>>& maze, int i, int j, int di, int dj) {
        if (i == di && j == dj) return true;
        int m = maze.size(), n = maze[0].size();
        bool res = false;
        maze[i][j] = -1;
        for (auto dir : dirs) {
            int x = i, y = j;
            while (x >= 0 && x < m && y >= 0 && y < m && maze[x][y] != 1) {
                x += dir[0];
                y += dir[1];
            }
            //撞墙了，越界了
            x -= dir[0]; y -= dir[1];
        }
        if (maze[x][y] != -1) {
            res |= helper(maze, x, y, di, dj);
        }
        return res;
    }

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
            if (t.first == destination[0] && t.second == destination[1]) return true;
            for (auto dir : dirs) {
                int x = t.first, y = t.second;
                while (x >= 0 && x < m && y >= 0 && y < n && maze[x][y] == 0) {
                    x += dir[0]; y += dir[1];
                }
                x -= dir[0]; y -= dir[1];
                if (!visited[x][y]) {
                    visited[x][y] = true;
                    q.push({x, y});
                }
            }
        }
        return false;
    }
};

//BFS
class SolutionT505 {
public:
    int shortestDistance(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination) {
        int m = maze.size(), n = maze[0].size();
        vector<vector<int>> dists(m, vector<int>(n, INT_MAX));
        vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}};
        queue<pair<int, int>> q;
        q.push({start[0], start[1]});
        dists[start[0]][start[1]] = 0;
        while(!q.empty()) {
            auto t = q.front(); q.pop();
            for (auto d : dirs) {
                int x = t.first, y = t.second, dist = dists[t.first][t.second];
                while (x >= 0 && x < m && y >= 0 && y < n && maze[x][y] == 0) {
                    x += d[0];
                    y += d[1];
                    ++dist;
                    //整形最大值 + 1 归0
                }
                x -= d[0];
                y -= d[1];
                --dist;
                if (dist[x][y] > dist) {
                    dist[x][y] = dist;
                    if (x != destination[0] || y != destination[1]) q.push({x, y});
                }
        }
        int res = dists[destination[0]][destination[1]];
        return (res == INT_MAX) ? -1 : res;
        }
    }

    int shortestDistance(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination) {
        int m = maze.size(), n = maze[0].size();
        vector<vector<int>> dists(m, vector<int>(n, INT_MAX));
        vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}};
        auto cmp = [](vector<int> &a, vector<int> &b) {
            return a[2] > b[2];
        }
        priority_queue<vector<int>, vector<vector<int>>, decltype(cmp) > pq(cmp);
        pq.push({start[0], start[1], 0});
        dists[start[0]][start[1]] = 0;
        while (!pq.empty()) {
            auto t = pq.top(); pq.pop();
            for (auto dir : dirs) {
                int x = t[0], y = t[1], dist = dist[x][y];
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

class SolutionT542 {
public:
    vector<vector<int>> updateMatrix(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}};
        queue<pair<int, int>> q;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (matrix[i][j] == 0) q.push({i, j});
                else matrix[i][j] = INT_MAX;
            }
        }
        while (!q.empty()) {
            auto temp = q.front(); q.pop();
            int x = temp.first, y = temp.second;
            for (auto dir : dirs) {
                int temp_x = x + dir[0], temp_y = y + dir[1];
                if (x < 0 || x >= m || y < 0 || y >= n || matrix[x][y] <= matrix[temp_x][temp_y]) {
                    matrix[temp_x][temp_y] = matrix[x][y] + 1;
                    q.push({temp_x, temp_y});
                }
            }
        }
        return matrix;
    }
};

class SolutionT994 {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        int cnt = 0, m = grid.size(), n = grid[0].size(), res = 0;
        vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}};
        queue<pair<int, int>> q;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] > 0) {
                    cnt++;
                    if (grid[i][j] == 2) {
                        q.push({i, j});
                        cnt++;
                    }
                }
            }
        }
        
        while (!q.empty()) {
            int temp_size = q.size();
            res++;
            for (int i = 0; i < temp_size; i++) {
                auto temp = q.front(); q.pop();
                int x = temp.first, y = temp.second;
                for (auto dir : dirs) {
                    int tx = x + dir[0], ty = y + dir[1];
                    if (tx < 0 || tx >= m || ty < 0 || ty >= n || grid[tx][ty] %2 == 0) continue;
                    cnt++;
                    grid[tx][ty] = 2;
                }
            }
        }
        return res;
    }
};

class SolutionT305 {
public:
    vector<int> numIslands2(int m, int n, vector<vector<int>>& positions) {
        vector<int> res;
        int cnt = 0;
        vector<int> root(m * n, -1);
        vector<vector<int>> dirs{{0, -1}, {-1, 0}, {0, 1}, {1, 0}};
        for (auto &pos : positions) {
            int id = n * pos[0] + pos[1];
            if (root[id] != -1) {
                res.push_back(cnt);
                continue;
            }
            root[id] = id;
            ++cnt;
            for (auto dir : dirs) {
                int x = pos[0] + dir[0], y = pos[1] + dir[1], cur_id = n * x + y;
                if (x < 0 || x >= m || y < 0 || y >= n || root[cur_id] == -1) continue;
                int p = findRoot(root, cur_id), q = findRoot(root, id);
                if (p != q) {
                    root[p] = q;
                    --cnt;
                }
            }
            res.push_back(cnt);
        }
    }

    int findRoot(vector<int> root, int id) {
        return root[id] == id ? id : findRoot(root, root[id]);
    }
};

class SolutionT773 {
public:
    int slidingPuzzle(vector<vector<int>>& board) {
        int res = 0, m = board.size(), n = board[0].size();
        string target = "123450", start = "";
        vector<vector<int>> dirs{{1,3}, {0,2,4}, {1,5}, {0,4}, {1,3,5}, {2,4}};
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                start += to_string(board[i][j]);
            }
        }
        unordered_set<string> visited{start};
        queue<string> q{{start}};
        while(!q.empty()) {
            for (int i = q.size() - 1; i >= 0; i++) {
                string cur = q.front(); q.pop();
                if (cur == target) return res;
                int zero_idx = cur.find("0");
                for (auto dir : dirs[zero_idx]) {
                    string cand = cur;
                    swap(cand[zero_idx], cand[dir]);
                    if (visited.count(cand)) continue;
                    visited.insert(cand);
                    q.push(cand);
                }
            }
            ++res;
        }
        return res;
    }

    int slidingPuzzle(vector<vector<int>>& board) {
        int m = board.size(), n = board[0].size(), res = 0;
        set<vector<vector<int>>> visited;
        queue<pair<vector<vector<int>> , vector<int>>> q;
        vector<vector<int>> correct{{1, 2, 3}, {4, 5, 0}};
        vector<vector<int>> dirs{{0, -1}, {-1, 0}, {0, 1}, {1, 0}};
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (board[i][j] == 0) q.push({board, {i, j}});
            }
        }
        while(!q.empty()) {
            for (int i = q.size() - 1; i >= 0; --i) {
                auto t = q.front(); q.pop();
                auto cur_board = t.first;
                auto zero_idx = t.second;
                if (cur_board == correct) return res;
                for (auto dir : dirs) {
                    int x = zero_idx[0] + dir[0], y = zero_idx[1] + dir[1];
                    if (x < 0 || x >= 2 || y < 0 || y >= 3) continue;
                    vector<vector<int>> temp_board = cur_board;
                    swap(temp_board[zero_idx[0]][zero_idx[1]], temp_board[x][y]);
                    if (visited.count(temp_board)) continue;
                    q.push({temp_board, {x, y}});
                }
            }
            ++res;
        }
        return -1;
    }

};


class Solution {
public:
    /**
     * @param grid: a 2D grid
     * @return: An integer
     */
    int shortestDistance(vector<vector<int>> &grid) {
        // write your code here
        vector<int> cntRow, cntCol;
        vector<int> distRow, distCol;
        int m = grid.size(), n = grid[0].size(), count = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    count++;
                    cntRow[i]++;
                    cntCol[j]++;
                }
            }
        }
        if (count == m * n) return -1;
        for (int row = 0; row < m; row++) {
            for (int i = 0; i < m; i++) {
                distRow[row] += abs(row - i) * cntRow[i];
            }
        }
        for (int col = 0; col < n; col++) {
            for (int j = 0; j < n; j++) {
                distCol[col] += abs(col - j) * cntCol[j];
            }
        }

        int minDist = INT_MAX;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    continue;
                }
                int dist = distRow[i] + distCol[j];
                minDist = min(minDist, dist);
            }
        }
        
        return minDist;
    }
};

struct Point {
    int x;
    int y;
    Point() : x(0), y(0) {}
    Point(int a, int b) : x(a), y(b) {}
};


class Solution {
public:
    /**
     * @param grid: a chessboard included 0 (false) and 1 (true)
     * @param source: a point
     * @param destination: a point
     * @return: the shortest path 
     */
    int shortestPath(vector<vector<bool>> &grid, Point &source, Point &destination) {
        // write your code here
        if (grid.empty() || grid[0].empty()) return -1;
        int m = grid.size(), n = grid[0].size();
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
                    return map[newP.x * m + newP.y];
                q.push(temp_p);
            }
        }
    }

    boolean isValidPath(vector<vector<int> grid, Point p){
        if(p.x < 0 || p.y < 0 || p.x >= grid.length || p.y >= grid[0].length)
            return false;
        if(grid[p.x][p.y] == true)
            return false;
        return true;
    }
};