#include<iostream>
#include<string>
using namespace std;
bool f(string s1,string s2,bool flag)
{
	int k1 = 0,k2 = 0,len1 = s1.length(),len2 = s2.length();
	while(k1<len1&&k2<len2){
		if(s1[k1] == s2[k2])
		{
			k1++,k2++;
			continue;
		}
		if(s1[k1++]!='<') return false;
		if(flag) cout<<" ";
		if(s1[k1]=='i')
		{
			bool flag2 = false;
			while(s2[k2]&&s2[k2]>='0'&&s2[k2]<='9')
			{
				if(s2[k2]>'0') flag2 = true;
				if(flag2 && flag) cout<<s2[k2];
				k2++;
				
			}
			if(!flag2) return false;
			k1+=4;
			continue;
		}
		if(s1[k1]=='s')
		{
			bool flag2 = false;
			while(s2[k2]&&s2[k2]!='/'){
				flag2 = true;
				if(flag) cout<<s2[k2];
				k2++;
			}
			if(!flag2) return false;
			k1+=4;
			continue;
		}
		if(s1[k1]=='p'){
			if(flag) while(s2[k2]) cout<<s2[k2++];
			return true;
		}
		return (k1==len1&&k2==len2);
	}
}

int main(){
	int n,m;
	cin>>n>>m;
	string sn[100],name[100],s;
	for(int i=0;i<n;i++) cin>>sn[i]>>name[i];
	for(int i=0;i<m;i++){
		cin>>s;
		bool flag = true;
		for(int i=0;i<n&&flag;i++){
			if(f(sn[i],s,false))
			{
				flag =false;
				cout<< name[i];
				f(sn[i],s,true);
			}
			if(flag) cout<<"404"<<endl;
		}
	}
	return 0;
	
}
