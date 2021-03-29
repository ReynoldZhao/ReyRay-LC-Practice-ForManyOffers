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

class SolutionT528 {
public:
    vector<int> s;
    Solution(vector<int>& w) {
        partial_sum(cbegin(w), cend(w), back_inserter(s));
    }
    
    int pickIndex() {
        int x = s.back();

        //upper_bound
        int index = rand() % x;
        auto it = upper_bound(s.begin(), s.end(), index);

        //lower_bound
        int index = rand() % x + 1;
        auto it = lower_bound(s.begin(), s.end(), index);
        return it - s.begin();
    }
};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution* obj = new Solution(w);
 * int param_1 = obj->pickIndex();