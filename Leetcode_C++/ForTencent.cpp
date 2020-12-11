#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <map>
#include <set>
#include <stack>
#include <utility>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <hash_map>
#include <deque>
using namespace std;


struct errInfo
{
    int st;
    int ed;
    string info;
    errInfo() {};
    errInfo(int _st, int _ed, string _info) : st(_ed), ed(_ed), info(_info){};
};

string getInfo(vector<errInfo> &arr,int errCode)
{
    int len=arr.size();
    for(int i=0;i<len;i++)
    {
        if(errCode>=arr[i].st&&errCode<=arr[i].ed)
        {
            return arr[i].info;
        }
    }
    return "错误码不存在";
}