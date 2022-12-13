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

class BrowserHistory {
public:
    list<string> history;
    list<string>::iterator it;
    BrowserHistory(string homepage) {
        history.push_back(homepage);
        it = history.begin();
    }
    
    void visit(string url) {
        auto next = it;
        next++;
        history.erase(next, history.end());
        history.push_back(url);
        it++;
    }
    
    string back(int steps) {
        while (it != history.begin() && steps--)
        {
            it--;
        }
        return *it;
    }
    
    string forward(int steps) {
        while (it != --history.end() && steps--)
        {
            it++;
        }
        return *it;
    }
};

/**
 * Your BrowserHistory object will be instantiated and called as such:
 * BrowserHistory* obj = new BrowserHistory(homepage);
 * obj->visit(url);
 * string param_2 = obj->back(steps);
 * string param_3 = obj->forward(steps);
 */