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
    int idCounter;
    unordered_map<int, int> idToRev;
    unordered_map<int, int> idToSoleRev;
    map<int, unordered_set<int> > revToIds;
    map<int, map<int, int>> referRevMap;
    map<int, set<int>> referMap;

public:
    RevenueShare() {
        // this->idCounter = 0;
        // this->idToRev.clear();
        // this->revToIds.clear();
        idCounter = 0;
        idToRev.clear();
        revToIds.clear();
    }

    int insert(int rev) {
        int id = idCounter++;
        idToRev[id] = rev;
        idToSoleRev[id] = rev;
        revToIds[rev].insert(id);

        // referRevMap[id][0] = rev;
        
        return id;
    }

    int insert(int rev, int referralId) {
        int id = insert(rev);
        idToSoleRev[id] = rev;

        // referRevMap[id][0] = rev;
        // if (referRevMap[referralId].count(1) == 0)  {
        //    referRevMap[referralId][1] = referRevMap[referralId][0] + rev; 
        // } else {
        //    referRevMap[referralId][1] += rev; 
        // }

        int oldRev = idToRev[referralId];
        idToRev[referralId] += rev;

        revToIds[oldRev].erase(referralId);

        if (revToIds[oldRev].empty()){
            revToIds.erase(oldRev);
        }

        revToIds[oldRev + rev].insert(referralId);

        referMap[referralId].insert(id);
        
        return id;
    }

    vector<int> getKLowestRevenue(int k, int target) {
        vector<int> KLowest;
        map<int, unordered_set<int> >::iterator it;
        int targetVal = target;

        while(KLowest.size() < k && it != revToIds.end()) {
            it = revToIds.upper_bound(targetVal);
            if (it == revToIds.end()) break;
            unordered_set<int> temp_set = it->second;
            unordered_set<int>::iterator setIt = temp_set.begin();
            for (auto i : temp_set){
                if (KLowest.size() < k) {
                    KLowest.push_back(i);
                }
            }            
            targetVal = it->first;
        }
        return KLowest;
    }

    int get_nested_revenue(int id, int max_nesting) {
        int cur_rev = 0;
        queue<int> referQ({id});
        while (max_nesting >= 0)
        {
            max_nesting--;
            int t_size = referQ.size();
            for (int i = 0; i < t_size ; i++) {
                int t = referQ.front(); referQ.pop();
                cur_rev += idToSoleRev[t];
                for (auto next : referMap[t]) {
                    referQ.push(next);
                }
            }
        }
        return cur_rev;
    }

    void print() {
        for (auto it : idToRev) {
            cout<< it.first << " : " << idToRev[it.first] << endl;
        }
    }
};

int main() {
    RevenueShare rs;
    int id1 = rs.insert(10);
    // int id2 = rs.insert(20, id1);
    // int id3 = rs.insert(40, id2);
    int id2 = rs.insert(50, id1);
    int id3 = rs.insert(20, id2);
    int id4 = rs.insert(30);
    int id5 = rs.insert(2);
    int id6 = rs.insert(100);
    int id7 = rs.insert(13, id3);
    int id8 = rs.insert(23, id5);
    int id9 = rs.insert(100, id7);
    rs.print();

    int n1 = rs.get_nested_revenue(0, 0);
    int n2 = rs.get_nested_revenue(0, 1);
    int n3 = rs.get_nested_revenue(0, 2);
    int n4 = rs.get_nested_revenue(1, 1);

    cout << "get_nested_revenue(0, 0): " << n1 << endl;
    cout << "get_nested_revenue(0, 1): " << n2 << endl;
    cout << "get_nested_revenue(0, 2): " << n3 << endl;
    cout << "get_nested_revenue(1, 1): " << n4 << endl;

    vector<int> res = rs.getKLowestRevenue(4, 70);
    for (auto r:res) {
        cout << r << endl; 
    }
}