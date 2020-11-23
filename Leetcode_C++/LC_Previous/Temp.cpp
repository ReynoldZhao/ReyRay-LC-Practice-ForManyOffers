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
using namespace std;
typedef pair<int,int> pa

class Solution {
public:
    bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
        bool pre[numCourses];
        for(int i=0;i<numCourses;i++) pre[i] = 1;
        for(int i=0;i<prerequisites.size();i++){
            pre[prerequisites[i].first] = false;
        }
        for(int i=0;i<numCourses;i++){
            if(!checnk(i,pre,prerequisites)) return false;
        }
        return true;
    }
    bool check(int i, bool &pre, vector<pair<int, int>>& prerequisites){
        if(pre[i]) return true;
        else{
            for(int j=0;j<prerequisites.size();j++){
                if(prerequisites[j].first==i){
                    if(check(prerequisites[j].second,pre,prerequisites)){
                        pre[i] = true;
                        return true;
                    }
                    else return false;
                }
            }
        }
    }
};

class Solution {
public:
    bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
        vector<vector<int> > graph(numCourses,vector<int>(0));
        vector<int> in(numCourses,0);
        for(auto a:prerequisites){
            graph[a[1]].push_back(a[0]);
            ++in[a[0]];
        }
        queue<int> q;
        for(int i=0;i<numCourses;i++){
            if(in[i]==0) q.push(i);
        }
        while(!q.empty()){
            int temp = q.front();
            q.pop();
            for(auto t:graph[temp]){
                --in[t];
            }
        }
        for(int i=0;i<numCourses;i++){
            
        }
    }
}
