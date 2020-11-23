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
 //����ط�Ҳ���Բ�������ģ���������Ϊ����һЩ�����������������ѭ���ж���ǰ��������
    sort(nums.begin(),nums.end());
    for(int i=0;i<nums.size();i++)
        for(int j=i+1;j<nums.size();j++)
            mark[nums[i]+nums[j]].push_back(make_pair(i,j));

//ע��ע������ط���һ���޴�Ŀӣ��м���ж������� i<nums.size()-3�������뵽��ѭ����
//��Ϊnums.size()��һ��unsigned�����ͣ�����int�����㣬�õ��Ļ���unsigned����������
//�������nums.size()<3�Ļ��ͻ������ѭ�����м��м�
    for(int i=0;i<nums.size()-3;i++){
    //���ж�����ǰ���������
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
 
