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
    int LastRemaining_Solution(int n, int m)
    {
        if(n<1||m<1) return -1;
        int a[n];
        memset(a,n*sizeof(a),0);
        int step=0;
        int count = n;
        int index = -1;
        while(count>0){
			index = (index+1)%n;
			if(a[index]==-1) continue;
			step++;
			if(step==m){
				cout<<index<<endl;
				a[index]=-1;
				step=0;
				count--;
			}
		}
		return index;
    }
};

int main(){
	Solution S;
	S.LastRemaining_Solution(100,5);
}
