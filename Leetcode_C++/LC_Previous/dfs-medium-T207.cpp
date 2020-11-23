#include<iostream>
#include<vector>
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
typedef pair<int,int> pa; 

class Solution {
public:
    bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
        bool pre[numCourses];
        for(int i=0;i<numCourses;i++) pre[i] = 1;
        for(int i=0;i<prerequisites.size();i++){
            pre[prerequisites[i].first] = false;
        }
        for(int i=0;i<numCourses;i++){
            if(!check(i,pre,prerequisites)) return false;
        }
        return true;
    }
    bool check(int i, bool pre[], vector<pair<int, int>>& prerequisites){
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

class Solution1 {
public:
    bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
        vector<vector<int> > graph(numCourses,vector<int>(0));
        vector<int> in(numCourses,0);
        for(auto a:prerequisites){
            graph[a.second].push_back(a.first);
            ++in[a.first];
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
                if(in[t]==0) q.push(t);
            }
        }
        for(int i=0;i<numCourses;i++){
            if(in[i]!=0) return false;
        }
        return true;
    }
};

class SolutionDFS{
	public:
		bool canFinish(int numCourses, vector<vector<int> >& prerequisites) {
			vector<vector<int> > graph(numCourse,vector<int>(0));
			vector<int> visit(numCourses,0);
        	for(auto a:prerequisites){
	            graph[a.second].push_back(a.first);
        	}
        	for(int i=0;i<numCourse;i++){
        		if(!canFinishDFS(graph,visit,i)); return false;
			}
			return true;
		}
		bool canFinishDFS(vector<vector<int>> graph, vector<int> visit, int i){
			if(visit[i]==-1) return false;
			if(visit[i]==1) return true;
			visit[i] = -1;
			for(auto a:graph[i]){
				if(!canFinish(graph,visit,a)) return false;
			}
			visit[i] = 1;
			return true;
		}
	
};
