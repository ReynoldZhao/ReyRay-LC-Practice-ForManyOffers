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