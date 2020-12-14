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
#include<hash_map>
#include<deque>
using namespace std;
//单调栈 递增栈 递减栈 当元素大于/小于栈顶元素的时候开始处理

class SolutionT84 {
public:
// 直方图矩形面积要最大的话，需要尽可能的使得连续的矩形多，并且最低一块的高度要高。
    int largestRectangleArea(vector<int> &height) {
        int res = 0;
        stack<int> st;
        height.push_back(0);
        for (int i = 0; i < height.size(); i++) {
            while(!st.empty && height[st.top] >= height[i]) {
                int cur = st.top(); st.pop();
                res = max(res, height[cur] * (st.empty()? i : i - st.top() - 1));
            }//递增站 递减栈 一直处理到小于当前/大于当前的为止，保证正确性
            st.push(i);
        }
        return res;
    }
};