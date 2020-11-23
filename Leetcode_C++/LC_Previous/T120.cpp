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
using namespace std;

class Solution {
public:
    int minimumTotal(vector<vector<int> >& triangle) {
        int size = triangle.size();
        int result[triangle[size-1].size()];
        for(int i=0;i<triangle[size-1].size();i++){
        	result[i] = triangle[size-1][i];
		}
		for(int i=0;i<size-1;i++){
			for(int j=0;j<triangle[size-2-i].size();j++){
				result[j] = min(result[j],result[j+1]) + triangle[size-2-i][j];
			}
		}
		return result[0];
    }
};
