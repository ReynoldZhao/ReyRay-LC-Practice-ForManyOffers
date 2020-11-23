#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<utility>

typedef pair<int,int> pii;
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        unordered_map<int,vector<pii> > store;
		set<vector<int> > res;
		vector<vector<int> > res2;
		sort(nums.begin(),nums.end())
		for(int i=0;i<nums.size()-1;i++){
			for(int j=i+1;j<nums.size(),j++){
				store[nums[i]+nums[j]].push_back(make_pair(i,j));
			}
		}
		
		for(int i=0;i<nums.size()-3;i++){
			if(nums[i]+nums[i+1]+nums[i+2]+nums[i+3]>target) break;
			for(int j=i+1;j<nums.size()-2;j++){
				if(store.find(target-(nums[i]+nums[j])!=store.end()){
					for(auto &&x:store[target-nums[i]-nums[j]])
					{
						if(x.first>j) {
							vector<int> temp{nums[i],nums[j],nums[x.first],nums[x.second]}
							res.insert(temp);
						}
					}
				} 
			}
		}
		for(auto &&x:res){
			res2.push_back(x);
		}
		return res2; 
    }
};

typedef pair<int,int> pii ;
vector<vector<int>> fourSum(vector<int>& nums, int target) {
    unordered_map<int,vector<pii>> mark;
    set<vector<int>> res;
    vector<vector<int>> res2;
    if(nums.size()<4)
        return res2;
 //这个地方也可以不用排序的，排序是因为减少一些计算量，方便下面的循环判定提前跳出条件
    sort(nums.begin(),nums.end());
    for(int i=0;i<nums.size();i++)
        for(int j=i+1;j<nums.size();j++)
            mark[nums[i]+nums[j]].push_back(make_pair(i,j));

//注意注意这个地方有一个巨大的坑，中间的判断条件： i<nums.size()-3，会陷入到死循环中
//因为nums.size()是一个unsigned的类型，其与int相运算，得到的还是unsigned！！！！！
//所以如果nums.size()<3的话就会出现死循环，切记切记
    for(int i=0;i<nums.size()-3;i++){
    //先判定，提前跳出的情况
        if(nums[i]+nums[i+1]+nums[i+2]+nums[i+3]>target)
            break;
        for(int j=i+1;j<nums.size()-2;j++){
            if(mark.find(target-(nums[i]+nums[j])) != mark.end()){
                for(auto &&x:mark[target-(nums[i]+nums[j])]){
                    if(x.first>j){
                        vector<int> tmp{nums[i],nums[j],nums[x.first],nums[x.second]};
                        res.insert(tmp);
                    }
                }
            }
        }
    }

    for(auto &&x:res){
        res2.push_back(x);
    }
    return res2;
}
 
