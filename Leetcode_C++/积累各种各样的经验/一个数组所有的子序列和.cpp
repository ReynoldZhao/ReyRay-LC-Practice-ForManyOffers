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
using namespace std;

    vector<int> make(vector<int> nums){
        vector<int> ans(1 << nums.size());
        for(int i = 0; i < nums.size(); ++i){
            for(int j = 0; j < (1 << i); ++j){
                ans[j + (1 << i)] = ans[j] + nums[i];
            }
        }
        return ans;
    }

    a[0] a[1] 第一个 要或者不要
    在nums[0]的基础上 第二个 要或者不要 a[0] a[1] a[2] a[3]
    所以对于nums[i], 自身有要或者不要两种，但是之前的所有可能的结果是2^i种，放在0-2^i上，然后这些都是不要的，
    要的都加在了后2^i --- end上了

作者：Monologue-S
链接：https://leetcode-cn.com/problems/closest-subsequence-sum/solution/mei-ju-shuang-zhi-zhen-by-monologue-s-zhxq/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。