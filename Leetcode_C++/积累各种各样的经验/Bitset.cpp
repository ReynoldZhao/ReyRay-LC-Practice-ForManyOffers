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
#include "bitset"
using namespace std;

bool canPermutePalindrome(const string &s) {
    bitset<128> flags;
    for(auto ch : s){
        flags.flip(ch);
    }
    return flags.count() < 2; //出现奇数次的字符少于2个
}