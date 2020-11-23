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
        int terminal = nums.size()-1;
        if(terminal==0) return true;
		bool acc[nums.size()];
		memset(acc,0,sizeof(acc));
		int temp;
        acc[nums.size()-1] = true;
		for(int i=nums.size()-1;i>=0;i--){
			if(nums[i]==0) acc[i]=false; 
			temp = nums[i];
			for(int j=1;j<=temp;j++){
				if(i+j<=terminal)acc[i] = acc[i]|acc[i+j];
				if(i+j>=terminal) acc[i] = true;
				if(acc[i]==true) break;
			}
		}
		int length = nums[0];
		int jump = 0;
		int next;
		int maxstride=0;
		for(int i=0;i<nums.size();i++){
			if(acc[i]==false) continue;
			length = nums[i];
			for(int j=i+1;j<=i+length;j++){
				if(acc[j]==true){
					if(i+nums[j]>=nums.size()-1) return ++jump;
					if(nums[j]>maxstride){
						next = j;
						maxstride = nums[j];
					}	
				}
			}
			jump++;
			i=next;
		}
		      
    }  
};
