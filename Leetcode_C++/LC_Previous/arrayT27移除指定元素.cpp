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
    int removeElement(vector<int>& nums, int val) {
        if(nums.empty()) return 0;
        int cur=0;
        int i=0;
        while(i<nums.size()){
            if(nums[i]!=val){
                nums[cur++] = nums[i++];
            }
            else i++;
		}
		return cur;
    }
};
