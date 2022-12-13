#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<stack>
#include<list>
#include<utility>
#include<cstring>
#include<string>
#include<unordered_map>
#include<unordered_set>
#include<ext/hash_map>
#include<deque>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <algorithm>
#include <list>
#include <map>
#include <vector>
#include <queue>
#include <stack>
#include <cmath>
using namespace std;

int const maxn = 1000010;
int const smaxn = 2005; //smaxn = sqrt(maxn)
char ch[maxn];
int num; //大块的个数
//累加和 知道之前的每个block里有多少东西
int sum[maxn];
int n, m;

struct ListBlock { //every block in blocklist
    int size; //size of every block
    char data[smaxn];
    void push_back(char c) {
        size++;
        data[size] = c;
    }
    void insert(int pos, char c) {
        for (int i = size + 1; i > pos; i--){
            data[i] = data[i - 1];
        }
        data[pos] = c;
        size++;
    }
    char getData(int pos) {
        return data[pos];
    }
    void del(int pos) {
        for (int i = pos; i < size; i++) {
            data[i] = data[i + 1];
        }
        size--;
    }
};
ListBlock block[smaxn];
void maintain() {
    for (int i = 1; i <= num; i++) {
        sum[i] = sum[i - 1] + block[i].size;
    }
}

//初始化字符串
void init() {
    num = sqrt((n + m) * 1.0) + 1;
    for (int i = 0; i < n; i++) {
        block[i / (num + 1)].push_back(ch[i]);
    }
    maintain();
}

//参数已经处理为block查询的参数
char query(int pos, int number) {
    return block[number].getData((pos));
}

//参数已经处理为block insert的参数
void insert(int pos, int number, char c)
{
    block[number].insert(pos, c);
}

//参数已经处理为block delete的参数
void del(int pos, int number) {
    block[number].del(pos);
}


char myQuery(int pos)
{
    //二分查找 应该插入到哪个block里
    int p = lower_bound(sum + 1,sum + 1 + num, pos) - sum;
    return query(pos - sum[p - 1], p);
}

void myInsert(char c, int pos)
{
    //二分查找 应该插入到哪个block里
    int p = lower_bound(sum + 1, sum + 1 + num,pos) - sum;
    insert(pos - sum[p - 1] , p, c);
    maintain();
}
