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

int main() {
    int m = 69;
    int n = 20;
    int* p = &m;
    //定义时，如果*的前面是数据类型，表明这个变量是个指针变量
    //即变量是p，p是指针变量，他的值是个地址
    *p = 20;
    *p = n;
    //是一样的，都是改变所指向内存地址（变量）的值
    //这里的*是解引用操作，表示指针变量指向的值
    p = &n;
    //改变指向
    *p = &n;
    //把n的地址作为m的值


    int a[6][6];
    *(*(a+1)+1)
    // 数组名代表第一个存储单元的地址，在这里 a + 1直接加了一个一维数组的地址量
    // *(a+1) 即 取到当前行的数组  == a[i]，(可以把*(a+1)理解成退化为一维数组的数组名a)，
    // 在此基础上+n，即在一位数组的初始地址上+n 

    int** ppa = &p;
    int*** pppa = &ppa;

    *pa is p
    **pa is m
    //指向指针的指针 int** 可以理解为数据类型 p是指针变量， &p，取得指针变量的地址

    char* arr[5];//指针数组，数组里装的指针
    char* (*pa)[5];//数组指针，指向数组的指针，+1 地址变化是5个量

    int array[5];
    int *parr1[10];// = (int*) pa[10] 是个指针数组 int(* parr3[10])[5] 方块优先级高 首先是个数组，数组之外的部分
    //int(*)[] 是个数组指针，是数组元素的类型，即为 存放数组指针的数组
    int (*pa)[5];
    pa = array;//放的是首元素（存储单元）的地址
    pa = &array;//取出的是整个数组的地址，但是暴露的还是首地址，+1会加整个数组量 int (*pa)[5]; pa一样的
                //int (*)[]
}


