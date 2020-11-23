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
    string addBinary(string a, string b) {
		string res = "";
		int al = a.size()-1,bl = b.size()-1;
		int m,n;
		int length = max(al,bl);
		int temp = 0, carry = 0;
		for(int i=0;i<length;i++){
			m = al-i>=0?a[al-i]-'0':0;
			n = bl-i>=0?b[bl-i]-'0':0;
			temp = (m+n+carry)%2;
			carry = (m+n+carry/2;
			res = res + to_string(temp);
		}
		return res; 
    }
};

class Solution {
public:
    string addBinary(string a, string b) {
        int l1 = a.size(),l2 = b.size();
        int length = max(l1,l2);
        string res;
        if(l1<l2) 
		{
			while(l1<length) {a.insert(a.begin(),'0');l1++;}
		}
		else
		{
			while(l2<length) {b.insert(b.begin(),'0');l2++;}			
		}
		int carry = 0;
		int temp;
		for(int i=length-1;i>=0;i--)
		{
			temp = a[i]-'0'+b[i]-'0'+carry;
			if(temp==0)
			{
				res.insert(res.begin(),'0');
				carry = 0;
			}
			if(temp==1)
			{
				res.insert(res.begin(),'1');
				carry = 0;
			}
			if(temp==2)
			{
				res.insert(res.begin(),'0');
				carry = 1;
			}
			if(temp==3)
			{
				res.insert(res.begin(),'1');
				carry = 1;
			}
			
		}
		if(carry==1){
			res.insert(res.begin(),'1');
		}
		return res;	
    }
};
