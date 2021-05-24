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
#include<hash_map>
#include<deque>
using namespace std;

class SolutionT323 {
public:
    int countComponents(int n, vector<pair<int, int> >& edges) {
        int res = n;
        vector<int> root(n, 0);
        for (int i = 0; i < n; i++) root[i] = i;
        for (auto edge : edges) {
            int x = findRoot(root, edge.first);
            int y = findRoot(root, edge.second);
            if (x != y) {
                --res;
                root[x] = y;
            }
        }
    }

    int findRoot(vector<int> &root, int i) {
        return root[i] == i ? i : findRoot(root, root[i]);
    }
};

class SolutionT547 {
public:
    int findCircleNum(vector<vector<int>>& M) {
        int n = M.size(), res = n;
        vector<int> root(n);
        for (int i = 0; i < n; ++i) root[i] = i;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (M[i][j] == 1) {
                    int p1 = getRoot(root, i);
                    int p2 = getRoot(root, j);
                    if (p1 != p2) {
                        --res;
                        root[p2] = p1;
                    }
                }
            }   
        }
        return res;
    }
    int getRoot(vector<int>& root, int i) {
        while (i != root[i]) {
            root[i] = root[root[i]];
            i = root[i];
        }
        return i;
    }
};

class SolutionT261 {
public:
    bool validTree(int n, vector<pair<int, int>>& edges) {
        vector<int> root(n, -1);
        for (auto edge : edges) {
            int x = findRoot(root, edge.first), y = findRoot(root, edge.second);
            //总有一个点是新的，不然就是成环
            if (x == y) return false;
            root[x] = y;
        }
        return edges.size() == n-1;
    }

    int findRoot(vector<int> &root, int i) {
        return root[i] == -1 ? i : findRoot(root, root[i]);
    }
};