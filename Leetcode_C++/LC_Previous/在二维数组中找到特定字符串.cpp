#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<utility>
#include<string.h> 
using namespace std;

class Solution {
public:
    bool hasPath(string matrix, int rows, int cols, string str)
    {
    	int flag[matrix.size()];
    	memset(flag,sizeof(flag)*matrix.size(),0);
    	for(int i=0;i<row;i++){
    		for(int j=0;j<cols;j++){
    			return dfs(matrix,i,j,str,k,flag);
			}
		}
		return false;
    }
    bool dfs(string matrix,int i,int j,string str,int k,int &flag)
		int index = i*rows+j;
		if(i<0||i>=rows||j<0||j>=cols||flag[index]==1||str[k]!=matrix[index]) return false;
		if(k==str.size()-1) return true;
		flag[index] = 1;
		if (dfs(matrix, rows, cols, i - 1, j, str, k + 1, flag)
                || dfs(matrix, rows, cols, i + 1, j, str, k + 1, flag)
                || dfs(matrix, rows, cols, i, j - 1, str, k + 1, flag)
                || dfs(matrix, rows, cols, i, j + 1, str, k + 1, flag)) {
            return true;
        }
        flag[index] = 0;
        return false;
};
