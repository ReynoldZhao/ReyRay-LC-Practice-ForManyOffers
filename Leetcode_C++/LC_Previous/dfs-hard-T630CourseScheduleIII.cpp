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

class Solution {
public:
    int scheduleCourse(vector<vector<int>>& courses) {
        priority_queue<int> q;
        sort(courses.begin(), courses.end(), [](vector<int>& a, vector<int>& b) {return a[1] < b[1];});
		sort(courses.begin(),courses.end(),[](vector<int> a,vector<int> b){return a[1]<b[1]});
		int curtime = 0;
		for(auto course:courses){
			curtime += course[0];
			q.push(course[0]);
			if(curtime>course[1]){
				curtime-=q.top();
				q.pop();
			}
		}
		return q.size(); 
    }
};
