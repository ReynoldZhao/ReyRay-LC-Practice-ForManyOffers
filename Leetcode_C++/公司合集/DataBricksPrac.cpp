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
#include<ext/hash_map>
#include<deque>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <algorithm>
#include <list>
#include <map>
#include <vector>
#include <queue>
#include <stack>
#include <cmath>
using namespace std;

class RevenueShare{
private:
    int globalId = -1;
    unordered_map<int, int> idToRev;
    map<int, set<int> > revToId;
    unordered_map<int, unordered_set<int>> referMap;
    unordered_map<int, int> idToSoloRev;

public:
    RevenueShare() {
        globalId = -1;
        idToRev.clear();
        revToId.clear();
    }

    int insert(int rev) {
        int id = globalId++;
        idToRev[id] = rev;
        revToId[rev].insert(id);
        idToSoloRev[id] = rev;

        return id;
    }

    int insert(int rev, int referralId) {
        int id = insert(rev);

        int oldRev = idToRev[referralId];
        revToId[oldRev].erase(referralId);
        if (revToId[oldRev].empty()) {
            revToId.erase(oldRev);
        }

        int updateRev = oldRev + rev;
        idToRev[referralId] = updateRev;
        revToId[updateRev].insert(referralId);

        referMap[referralId].insert(id);

        return id;
    }

    vector<int> getKLowestRevenue(int k, int target) {
        vector<int> KLowest;
        map<int, set<int>>::iterator it;
        int targetVal = target;

        while (it != revToId.end() && KLowest.size() < k) {
            it = revToId.upper_bound(target);
            if (it == revToId.end()) break;
            set<int> t_set = it->second;
            for (auto t : t_set) {
                if (KLowest.size() < k) KLowest.push_back(t);
                else break;
            }
            targetVal = it->first;
        }

        return KLowest;
    }

    int get_nested_revenue(int id, int max_nesting) {
        int cur_rev = 0;
        queue<int> q({id});

        while (!q.empty() && max_nesting > 0) {
            int t_size = q.size();
            max_nesting--;
            for (int i = 0; i < t_size; i++) {
                int t = q.front(); q.pop();
                cur_rev += idToSoloRev[t];
                for (auto next : referMap[t]) {
                    q.push(next);
                }
            }
        }
        return cur_rev;
    }
};