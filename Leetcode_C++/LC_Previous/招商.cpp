#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<utility>

using namespace std;

class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
		int temp = data[0];
		for(int i=1;i<data.size();i++){
			temp = temp^data[i];
		}
		int firstindex = FirstBit(temp);
		num1 = num2 = 0;
		for(int i=0;i<data.size();i++){
			if(iorFunc(data[i])) num1 = num1^data[i];
			else num2 = num2 = num2^data[i];
		}
    }
    bool iorFunc(int n,int indexbit){
		n = n>>indexbit;
		return n&1==1;
	}
    int FirstBit(int temp){
    	int pos = 0;
    	while((temp&1)==0&&pos<8*4){
    		temp = temp>>1;
    		pos++;
		}
		return pos;
	}
};
