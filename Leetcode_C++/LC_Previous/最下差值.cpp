#include<iostream>
#include<queue>

using namespace std;

int main(){
	int n,k;
	queue<int> Q;
	scanf("%d %d",&n,&k);
	int num=1;
	for(int i=1;i<=n;i++){
		Q.push(i);
	}
	int top;
	while(Q.size()>1){
		top = Q.front();
		Q.pop();
		if(num%k!=0&&(num-k)%10!=0) Q.push(top);
		num++; 
	}
	cout<<Q.front();
	return 0;	
} 

