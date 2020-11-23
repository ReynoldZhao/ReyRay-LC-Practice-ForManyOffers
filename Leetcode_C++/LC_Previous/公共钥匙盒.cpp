#include<iostream>
#include<vector>
#include<algorithm> 
using namespace std;
struct key{
	int time;
	int kid;
	int flag;//0为取 1位还 
};
bool compare(key a,key b){
	if(a.time!=b.time) return a.time<b.time;
	else if(a.flag!=b.flag) return a.flag>b.flag;
	else{
		return a.kid<b.kid;
	}
}
int main(){
	int N,K;
	vector<key> v;
	scanf("%d %d",&N,&K);
	int index[N+1];
	for(int i=1;i<=N;i++) index[i] = i;
	key k;
	for(int i=0;i<K;i++){
		int w,s,c;
		scanf("%d %d %d",&w,&s,&c);
		k.flag = 0;
		k.kid = w;
		k.time = s;
		v.push_back(k);
		k.flag = 1;
		k.time = s+c;
		v.push_back(k);
	}
	sort(v.begin(),v.end(),compare);
	for(int i=0;i<v.size();i++){
		int id = v[i].kid, flag = v[i].flag;
		if(flag==0){
			for(int i=1;i<=N;i++){
				if(index[i]==id) {
					index[i] = 0;
					break;
				}
			}
		}
		if(flag==1){
			for(int i=1;i<=N;i++){
				if(index[i]==0) {
					index[i] = id;
					break;
				}
			}			
		}
	} 
    cout<<index[1];
    for(int i=2;i<=N;i++){
        cout<<" "<<index[i];
    }
    return 0;
}
