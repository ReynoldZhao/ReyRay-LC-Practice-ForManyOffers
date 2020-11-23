//#include<iostream>
//#include<algorithm>
//#include<vector> 
//using namespace std;
//int sp[101];
//vector<int> v;
//
//int main(){
//	int n,L,t;
//	scanf("%d %d %d",&n,&L,&t);
//	for(int i=1;i<=n;i++){
//		scanf("%d",&sp[i]);
//		if(sp[i]+t<L) v.push_back(sp[i]+t);
//		else{
//			if((sp[i]+t/L)%2==0) v.push_back((sp[i]+t)%L);
//			else v.push_back(L-(sp[i]+t)%L);
//		}
//	}
//	sort(v.begin(),v.end());
//	for(int i=0;i<v.size();i++){
//		printf("%d ",v[i]);
//	}
//
//}
//#include<iostream>
//#include<map>
//
//using namespace std;
//int ball[2][100+10];
//int out[100+10]; 
//
//int main(){
//    int n,L,t,in;
//    map<int,int> posmap;
//
//    cin>>n>>L>>t;
//    //建立输入输出映射关系 
//    for(int i=0; i<n; i++){
//        cin>>in;
//        posmap[in] = i;
//        printf("%d ",&posmap[i]);
//    }
//}
	
//
//    int i=0;
//    for(map<int,int>::iterator it=posmap.begin(); it!=posmap.end(); it++, i++){
//        ball[0][i] = it->first;
//        ball[1][i] = 1;
//    }
//
//    while(t--){
//        for(int j=0; j<n; j++){
//            ball[0][j] += ball[1][j];
//        }
//        for(int j=0; j<n; j++){
//            //左边球碰撞情况 
//            if(ball[0][j]==0){
//                //ball[0][j] = 0;
//                ball[1][j] = 1;
//            //右边秋碰撞情况 
//            }else if(ball[0][j]==L){
//                //ball[0][j] = n-1;
//                ball[1][j] = -1;
//            //中间球与下一个球是否碰撞 
//            }else{
//                if(j<n-1 && ball[0][j]==ball[0][j+1]){
//                    ball[1][j] = -ball[1][j]; 
//                    ball[1][j+1] = -ball[1][j+1];
//                    j++; 
//                }
//            }
//        }
//    }
//
//    i=0; 
//    for(map<int,int>::iterator it=posmap.begin(); it!=posmap.end(); it++,i++){
//        out[it->second] = ball[0][i];
//    }
//
//    for(int i=0; i<n; i++){
//        cout<<out[i]<<" ";
//    }   
//
//
//    return 0;
//}
//#include<bits/stdc++.h>
// 
//using namespace std;
// 
//const int maxn=150;
// 
//struct node{
//    int coor,no,b;
//};
//node a[maxn];
// 
//bool cmp1(node a,node b)
//{
//    return a.coor<b.coor;
//}
// 
//bool cmp2(node a,node b)
//{
//    return a.no<b.no;
//}
// 
//int main()
//{
//    int n,t,l;
//    scanf("%d%d%d",&n,&l,&t);
//    for(int i=0;i<n;i++){
//        scanf("%d",&a[i].coor);
//        a[i].no=i;
//        a[i].b=1;
//    }
//    sort(a,a+n,cmp1);
//    for(int i=0;i<t;i++){
//        a[0].coor=a[0].coor+a[0].b;
//        if(a[0].coor==0){
//            a[0].b*=-1;
//        }
//        for(int j=1;j<n;j++){
//            a[j].coor=a[j].coor+a[j].b;
//            if(a[j].coor==a[j-1].coor){
//                a[j].b*=-1;
//                a[j-1].b*=-1;
//            }
//        }
//        if(a[n-1].coor==l){
//            a[n-1].b*=-1;
//        }
//    }
//    sort(a,a+n,cmp2);
//    for(int i=0;i<n;i++){
//        if(i!=0){
//            printf(" ");
//        }
//        printf("%d",a[i].coor);
//    }
//} 

#include<iostream>
#include<algorithm>
using namespace std;
struct Dot{
	int no;
	int sp;
	int dir;
}dot[101];

bool cmp1(Dot A,Dot B){
	return A.sp<B.sp;
}
bool cmp2(Dot A,Dot B){
	return A.no<B.no;
}
int main(){
	int n,L,t;
	scanf("%d %d %d",&n,&L,&t);
	for(int i=0;i<n;i++){
		int temp;
		scanf("%d",&temp);
		dot[i].no = i;
		dot[i].sp = temp;
		dot[i].dir = 1;
	}
	sort(dot,dot+n,cmp1);
	while(t--){
		dot[0].sp = dot[0].sp+dot[0].dir;
		if(dot[0].sp==0) dot[0].dir*=-1;
		for(int i=1;i<n;i++){
			dot[i].sp = dot[i].sp + dot[i].dir;
			if(dot[i].sp == dot[i-1].sp){
				dot[i].dir*=-1;
				dot[i-1].dir*=-1;
			}
		}
		if(dot[n-1].sp == L){
				dot[n-1].dir*=-1;
			}
	}
	sort(dot,dot+n,cmp2);
    for(int i=0;i<n;i++){
        if(i!=0){
            printf(" ");
        }
        printf("%d",dot[i].sp);
    }
    printf("\n");
    return 0;
	
}
