#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<utility>
#include<cstring>
using namespace std; 
typedef pair<int,int> pii;
class Solution {
public:
    int jump(vector<int>& nums) {
		if (nums.size()<= 1)
			return 0;
		if (nums[0] == 0)
			return -1;
  		int n = nums.size();
  		int start,reach;
  		start = 0;
  		reach = nums[0];
  		int step = 0;
  		int nextmax;
  		for(;start<reach&&start<n-1;){
  			step++;
  			if(reach>=n-1) return jump;
  			nextmax = 0;
  			for(int j=start;j<=reach;j++){
  				if(j+num[j]>nextmax){
  					start = j;
  					nextmax = j+num[j];
				  }
			  }
			reach = nextmax;	
		  }
		      
    }
	int jump(vector<int>& nums) {
		int cur=0;//Ŀǰ�ܴﵽ����Զ���� 
		int step = 0
		int last = 0;
		// last��ָ��֮ǰ�ĵ���reach������Զλ��
		for(int i=0;i<nums.size();i++){
			if(i>last){
				step++;
				last =cur;
			}
			cur = max(num[i]+i,cur);
		}
	}
};
