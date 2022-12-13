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
using namespace std;

class LRUCache1 {
public:
    list<pair<int, int>> lru;
    unordered_map<int, list<pair<int, int> >::iterator> map;
    int cap = 0;
    LRUCache(int capacity) {
        cap = capacity;
    }
    
    int get(int key) {
        if (map.count(key) == 0) return -1;
        auto it = map[key];
        int res = (*it).second;
        lru.push_front(make_pair(key, res));
        lru.erase(map[key]);
        // lru.insert(lru.begin(), make_pair(key, res));
        map[key] = lru.begin();
        return res;
    }
    
    void put(int key, int value) {
        if (map.count(key) != 0) {
            lru.erase(map[key]);
        }
        lru.push_front(make_pair(key, value));
        map[key] = lru.begin();
        if (map.size() > cap) {
            auto t = lru.rbegin();
            int k = (*t).first;
            lru.pop_back();
            map.erase(k);
        }
    }
};

class LRUCache2 {
public:
    LRUCache(int capacity) {
        cap = capacity;
    }
    
    int get(int key) {
        if (m.count(key) == 0) return -1;
        auto it = m[key];
        int res = (*it).second;
        l.push_front(make_pair(key, res));
        l.erase(m[key]);
        m[key] = l.begin();
        return res;
    }
    
    void put(int key, int value) {
        if (m.count(key) != 0) {
            l.erase(m[key]);
        }
        l.push_front(make_pair(key, value));
        m[key] = l.begin();
        if (m.size() > cap) {
            auto it = l.rbegin();
            int k = (*it).first;
            l.pop_back();
            m.erase(k);
        }
    }
    
private:
    int cap;
    list<pair<int, int>> l;
    unordered_map<int, list<pair<int, int>>::iterator> m;
};
