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
using namespace std;

//class Solution {
//public:
//    string fractionToDecimal(int numerator, int denominator) {
//		if(numerator==0) return "0";
//		if(denominator==0) return "";
//		long long int n = numerator,d = denominator;
//		string ans = "";
//		if(n<0 ^ d<0) ans+="-";
//		long long int r = n%d;
//		if(r==0) {
//			ans+=to_string(n/d);
//			return ans;
//		}
//		ans+=to_string(n/d);
//		ans+=".";
//		unordered_map<int,int> m;
//		while(r){
//			if(m.find(r)!=m.end()){
//				ans.insert(m[r],1,"(");
//				ans+=")";
//				break;
//			}
//			m[r] = ans.size();
//			r*=10;
//			ans+=to_string(r/d);
//			r = r%d;
//		}
//    }
//};
class Solution {
public:
    vector<string> findRepeatedDnaSequences(string s) {
        vector<string> res;
        if(s.size()<10) return res;
        int mask = 0x7ffffff;
		int cursor = 0,i=0;
		while(i<9){
			cursor = (cursor << 3) | (s[i++]&7);
		} 
		unordered_map<int,int> map;
		while(i<s.size()){
			cursor = (cursor << 3) | (s[i++]&7);
			if(map.find(cursor)!=map.end()){
				if(map[cursor]==1) res.push_back(s.substr(i-10,10));
				map[cursor]++;
			}
			else map[cursor] = 1;
		}
		return res;
    }
};

