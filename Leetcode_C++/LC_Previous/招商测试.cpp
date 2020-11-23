#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<utility>

using namespace std;

int main(){
    //freopen("1.in","r",stdin);
    int n;
	cin>>n;
	int min1,max1;
	cin>>min1>>max1;
	int min2,max2;
	cin>>min2>>max2;
	int min3,max3;
	cin>>min3>>max3; 
	int totalmin = min1+min2+min3;
	int res1=min1,res2=min2,res3=min3;
	if(totalmin==n){
		printf("%d %d %d\n",res1,res2,res3);
		return 0;
	}
	if(n==max1+max2+max3){
		printf("%d %d %d\n",max1,max2,max3);
		return 0;
	}
	int rest;
		rest = n-totalmin;
		if(rest>=max1-min1){
			if(rest==max1-min1){
				res1 = max1;
				printf("%d %d %d\n",res1,res2,res3);
				return 0;
			}
			else{
				res1 = max1;
				rest = rest-(max1-min1);
				if(rest>=max2-min2){
					if(rest==max2-min2){
						res2 = max2;
						printf("%d %d %d\n",res1,res2,res3);
						return 0;
					}
					else{
						res2 = max2;
						rest = rest-()max2;
						res3 = res3+rest;
						printf("%d %d %d\n",res1,res2,res3);	
						return 0;
					}	
				}
				else{
					res2 = rest + min2;
					printf("%d %d %d\n",res1,res2,res3);
					return 0;					
				}
				
			}			
		}
		else{
			res1 = rest+min1;
			printf("%d %d %d\n",res1,res2,res3);
			return 0;
		}
	int totalmax = max1+max2+max3;
	return 0;		
}
