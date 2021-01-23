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
#include<priority_queue>
using namespace std;

class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        auto cmp = [](ListNode*& a, ListNode*& b) {
            return a->val > b->val;
        };
        priority_queue<ListNode*, vector<ListNode*>, decltype(cmp) > q(cmp);
        for (auto node : lists) {
            if (node) q.push(node);
        }
        ListNode *dummy = new ListNode(-1), *cur = dummy;
        while (!q.empty()) {
            auto t = q.top(); q.pop();
            cur->next = t;
            cur = cur->next;
            if (cur->next) q.push(cur->next);
        }
        return dummy->next;
    }
};

class SolutionT373 {
public:
    vector<pair<int, int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        vector<pair<int, int>> res;
        priority_queue<pair<int, int>, vector<pair<int, int>>, cmp> q;
    }
    struct cmp {
        bool operator() (pair<int, int> &a, pair<int, int> &b) {
            return a.first + a.second < b.first + b.second;
        }
    } 

    vector<pair<int, int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        vector<pair<int,int>> result;
        if (nums1.empty() || nums2.empty() || k <= 0)
            return result;
        auto comp = [&nums1, &nums2](pair<int, int> a, pair<int, int> b) {
            return nums1[a.first] + nums2[a.second] > nums1[b.first] + nums2[b.second];};
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(comp)> min_heap(comp);
    }
};
    
class Edge:
    struct Edge{
        int start,end;
        double weight;
        Edge(int start,int end,int weight):start(start),end(end),weight(weight){};
    };
    typedef struct Edge Edge;

    bool operator > (Edge a,Edge b)
    {
        return a.weight > b.weight;
    }

    int main()
    {
        priority_queue<Edge,vector<Edge>,greater<Edge>> pqueue_Edge;
    }

class student{
    private:
        string name;
        int age;
    public:
        bool operator<(const student& obj)const{
            return this->age > obj.age;
        }
        student(string name, int age){
            this->name = name;
            this->age = age;
        }
        string GetName(){
            return this->name;
        }
    };

    int main(){
        priority_queue<student>pq;
    }

class SolutionT973 {
public:
    struct cmp {
        bool operator() (pair<int, int> &a, pair<int, int> &b) {
            return a.first  > b.first;
        }
    };
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        // auto comp = [](pair<int, int> a, pair<int, int> b) {
        //     return a.first < b.first;};
        priority_queue<pair<int, int>, vector<pair<int, int>>, cmp> pq;
        for (int i = 0; i < points.size(); i++) {
            int prod = points[i][0] * points[i][0] + points[i][1] * points[i][1];
            pq.push(make_pair(prod, i));
        }
        vector<vector<int>> res;
        for (int i = 0; i < K; i++) {
            auto temp = pq.top(); pq.pop();
            res.push_back(points[temp.second]);
        }
        return res;
    }
};

class SolutionT778 {
public:
    int swimInWater(vector<vector<int>>& grid) {
        int res = 0, n = grid.size();
        unordered_set<int> visited{0};
        vector<vector<int>> dirs{{0, -1}, {-1, 0}, {0, 1}, {1, 0}};
        auto cmp = [](pair<int, int>& a, pair<int, int>& b) {return a.first > b.first;};
        // 这里写成greater，即为小顶堆
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp) > q(cmp);
        q.push({grid[0][0], 0});
        while (!q.empty()) {
            int i = q.top().second / n, j = q.top().second % n; q.pop();
            res = max(res, grid[i][j]);
            if (i == n - 1 && j == n - 1) return res;
            for (auto dir : dirs) {
                int x = i + dir[0], y = j + dir[1];
                if (x < 0 || x >= n || y < 0 || y >= n || visited.count(x * n + y)) continue;
                visited.insert(x * n + y);
                q.push({grid[x][y], x * n + y});
            }
        }
        return res;
    }
};