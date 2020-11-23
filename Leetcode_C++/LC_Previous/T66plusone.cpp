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

class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        int len = digits.size();
        int carry = 0;
        int temp = digits[digits.size()-1]+1;
        carry = temp/10;
        digits[digits.size()-1] = temp%10;
        for(int i=0;digits.size()-2-i>=0;i++){
	        int temp = digits[digits.size()-2-i]+carry;
	        carry = temp/10;
	        digits[digits.size()-2-i] = temp%10;
		}
		vector<int> res;
		if(carry==0) return digits;
		else{
			res.push_back(1);
			for(int i=0;i<digits.size();i++){
				res.push_back(digits[i]);
			}	
		}
        return res;		 
    }
};

// error
class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
		int n = digits.size();
		for(int i=n-1;i>=0;i--){
			if(digits[i]==9){
				digit[i]=0
			}
			else{
				digits[i]++;
				return;
			}
		}
		digits[0] = 1;
		digits.push_back(0);
		return digits;		 
    }
};

