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

class SolutionT1041 {
public:
    bool isRobotBounded(string instructions) {
        int dir = 0, x = 0, y = 0; // 0 north, 1 east, 2 south, 3 west
        //只要保持一个顺时针或者逆时针的顺序即可，这样向左向右都是+1了
        vector<vector<int>> dirs{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        for (int i = 0; i < instructions.size(); i++) {
            if (instructions[i] == 'G') {
                x += dirs[dir][0];
                y += dirs[dir][1];
            } else if (instructions[i] == 'L') {
                dir = (dir + 4 - 1) % 4;
            }
            else {
                dir = (dir + 1) % 4;
            }
        }
        return (x == 0 && y == 0) || dir > 0;
    }
};

class SolutionT552 {
public:
    int checkRecord(int n) {
        int M = 1e9 + 7;
        int dp[n + 1][2][3];
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 3; ++k) {
                dp[0][j][k] = 1;
            } 
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 3; k++) {
                    //对于 前i, 最多出现j次A, 最多连续出现k次连续L 这个位置当前这个位置有多少种组合
                    int val = dp[i-1][j][2]; //第n位是 P，所以可以加上这个极限，且不考虑连续三个L
                    if (j > 0) val = (val + dp[i-1][j-1][2]) % M; //加上没出现过A的, 即最后一位是A
                    if (k > 0) val = (val + dp[i-1][j][k-1]) % M;//最后一位是L的
                    dp[i][j][k] = val;
                }
            }
        }
        return dp[n][1][2];
    }
};

class Solution {
public:
    int maxArea(int h, int w, vector<int>& horizontalCuts, vector<int>& verticalCuts) {
        int m = 1e9 + 7;
        sort(horizontalCuts.begin(), horizontalCuts.end());
        sort(verticalCuts.begin(), verticalCuts.end());
        int max_h_interval = horizontalCuts[0];
        int max_v_interval = verticalCuts[0];
        for (int i = 1; i < horizontalCuts.size(); i++) {
            max_h_interval = max(max_h_interval, horizontalCuts[i] - horizontalCuts[i - 1]);
        }
        max_h_interval = max(max_h_interval, h - horizontalCuts.back());
        for (int j = 1; j < verticalCuts.size(); j++) {
            max_v_interval = max(max_v_interval, verticalCuts[j] - verticalCuts[j - 1]);
        }
        max_v_interval = max(max_v_interval, w - verticalCuts.back());
        long long area = max_h_interval * max_v_interval;
        int res = area % m;
        return res;
    }
};

class SolutionT937 {
public:
    vector<string> reorderLogFiles(vector<string>& logs) {
        vector<string> digitLogs, ans;
        vector<pair<string,string>> letterLogs;
        for (auto log : logs) {
            int i = 0;
            while (log[i] != ' ') i++;
            if (isalpha(log[i+1])) letterLogs.emplace_back(log.substr(0, i), log.substr(i+1));
            else digitLogs.push_back(log);
        }
        sort(letterLogs.begin(), letterLogs.end(), [&](auto &a, auto &b) {
            return a.second == b.second ? a.first < b.first : a.second < b.second;
        });
        for (auto &p : letterLogs) ans.push_back(p.first + " " + p.second);
        for (string &s : digitLogs) ans.push_back(s);
        return ans;        
    }
};

class LRUCache {
public:
    LRUCache(int capacity) {
        max_cap = capacity;
    }
    
    int get(int key) {
        if (m.count(key)) {
            auto it = m.find(key);
            auto res = m[key]->second;
            l.erase(m[key]);
            l.insert(l.begin(), {key, res});
            m[key] = l.begin();
            return res;
        } else return -1;
    }
    
    void put(int key, int value) {
        if (m.count(key)) {
            l.erase(m[key]);
        }
        l.push_front(make_pair(key, value));
        m[key] = l.begin();
        if (m.size() > max_cap) {
            int del_key = l.rbegin()->first;
            l.pop_back();
            m.erase(del_key);
        }
    }
private:
    list<pair<int, int>> l;
    int max_cap;
    unordered_map<int, list<pair<int, int>>::iterator> m;
};

class LRUCache{
public:
    LRUCache(int capacity) {
        cap = capacity;
    }
    
    int get(int key) {
        auto it = m.find(key);
        if (it == m.end()) return -1;
        l.splice(l.begin(), l, it->second);
        return it->second->second;
    }
    
    void put(int key, int value) {
        auto it = m.find(key);
        if (it != m.end()) l.erase(it->second);
        l.push_front(make_pair(key, value));
        m[key] = l.begin();
        if (m.size() > cap) {
            int k = l.rbegin()->first;
            l.pop_back();
            m.erase(k);
        }
    }
    
private:
    int cap;
    list<pair<int, int>> l;
    unordered_map<int, list<pair<int, int>>::iterator> m;
};

class SolutionT1167 {
public:
    int connectSticks(vector<int>& sticks) {
        priority_queue<int, vector<int>, greater<int>> pq(sticks.begin(), sticks.end());
        int res = 0;
        while (pq.size() > 1) {
            int t1 = pq.top(); pq.pop();
            int t2 = pq.top(); pq.pop();
            res += t1 + t2;
            pq.push(t1 + t2);
        }
        return res;
    }
};

class SolutionT472 {
public:
    //每次都遍历会超时的
    vector<string> findAllConcatenatedWordsInADict(vector<string>& words) {
        vector<string> res;
        unordered_set<string> set(words.begin(), words.end());
        for (auto word : words) {
            if (helper(word, words, set, 0, 0)) res.push_back(word);
        }
        return res;
    }

    bool helper(string word, vector<string>& words, unordered_set<string> set, int pos, int count) {
        if (pos >= word.size() && count >= 2) return true;
        // for (auto word : words) {
        //     int len = word.size();
        //     if (len <= cur_word.size() && cur_word.substr(0, len) == word && helper(cur_word.substr(len), words, count+1)) {
        //         return true;
        //     }
        // }
        for (int i = pos; i < word.size(); i++) {
            string t = word.substr(pos, i - pos + 1);
            if (set.count(t) && helper(word, words, set, i + 1, count+1))
                return true;
        }
        return false;
    }
};

//WordBreak
class SolutionT139 {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
        vector<int> memo(s.size(), -1);
        return check(s, wordSet, 0, memo);
    }

    bool check(string s, unordered_set<string>& wordSet, int start, vector<int>& memo) {
        if (memo[start] != -1) return memo[start];
        for (int i = start; i < s.size(); i++) {
            string temp = s.substr(start, i - start + 1);
            if (wordSet.count(temp) && check(s, wordSet, i + 1, memo))
                return memo[start] = 1;
        }
        return memo[start] = 0;
    }

    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
        vector<bool> dp(s.size() + 1);
        dp[0] = true;
        for (int i = 0; i < dp.size(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordSet.count(s.substr(j, i - j))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp.back();
    }
};

class SolutionT1710 {
public:
    int maximumUnits(vector<vector<int>>& boxTypes, int truckSize) {
        sort(boxTypes.begin(), boxTypes.end(), [](vector<int> a, vector<int> b) {
            return a[1] > b[1];
        });
        int res = 0;
        for (auto t : boxTypes) {
            if (t[0] <= truckSize) {
                res += t[0] * t[1];
                truckSize -= t[0];
            }
            else {
                res += truckSize * t[1];
                return res;
            }
        }
        return res;
    }

    //桶排序
    int maximumUnits(vector<vector<int>>& boxTypes, int truckSize) {
        vector<int> buckets(1001, -1);
        int space_remaining_boxes = truckSize;
        int units_loaded = 0;
        for (int i = 0; i < boxTypes.size(); ++i) {
            if (buckets[ boxTypes[i][1] ] == -1) {
                buckets[ boxTypes[i][1] ] = boxTypes[i][0];
            } else { // already has a value
                buckets[ boxTypes[i][1] ] += boxTypes[i][0];
            }
            
            // optimization idea: when populating, track the highest and lowest boxesperunit for use as indices below
        }
        
        for (int i = 1000; i >= 0; --i) {
            if (buckets[i] == -1) continue;
            
            if (buckets[i] > space_remaining_boxes) { // case:not enough space on truck. eg., we have 2 box but truck space 1.
                units_loaded += space_remaining_boxes*i;
                return units_loaded;
            } else {
                units_loaded += buckets[i]*i; // i is 10units/box. buckets[i] is 2 boxes. total units is 20.
                space_remaining_boxes -= buckets[i]; // space_remaining is in units of boxes.
            }
            
        }
        return units_loaded;
    }
};



