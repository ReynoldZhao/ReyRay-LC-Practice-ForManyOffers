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
#define pair<int,int> p; 
using namespace std;

int main() {
    string s;
	cin>>s;
	int len = s.size();
	int index = 0;
	int result = 0;
	int flag = 1;
	while(index<len){
		if(s[index]=='-'){
			flag = -1;
			index++;
			continue;
		}
		if(s[index]=='+'){
			index++;
			continue;
		}
		if(s[index]>='0'&&s[index]<='9'){
			int temp = 0;
			while(s[index]>='0'&&s[index]<='9'){
				int temp1 = s[index] - '0';
				index++;
				temp = temp*10 + temp1;
			}
			result = result + temp*flag;
			flag = 1;
		}
	}
	cout<<result<<endl;
}

