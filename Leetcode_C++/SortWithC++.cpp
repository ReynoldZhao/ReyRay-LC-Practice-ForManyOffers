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
#include<hash_map>
#include<deque>
using namespace std;

void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
    return ;
}

void bubbleSort(vector<int> &arr) {
    for (int i = 0; i < arr.size() - 1; i++) {
        for (int j = i; j  < arr.size() - 1; j++) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }
    }
}

void selectionSort(vector<int> &a)
{
    int len = a.size();
    for (int i = 0, minIndex; i < len - 1; i++) //需要循环次数
    {
        minIndex = i;                     //最小下标
        for (int j = i + 1; j < len; j++) //访问未排序的元素
        {
            if (a[j] < a[minIndex])
                minIndex = j; //找到最小的
        }
        swap(a[i], a[minIndex]);
    }
}

void insertSort(vector<int> &arr) {
    for (int i = 0; i < arr.size(); i++) {
        int j = i;
        int temp = a[i+1];
        while (j >= 0 && a[j] > temp) {
            a[j + 1] = a[j];
            j--
        }
        a[j+1] = temp;
    }
}

// QuickSort
int partition(vector<int> &arr, int l, int r) {
    int pivot = arr[r];
    int i = l - 1;
    for (int j = l; j < r; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i+1], arr[r]);
    return i+1;
}
void quickSort(vector<int> &arr, int l, int r) {
    if (l >= r) return;
    int mid = partition(arr, l, r);
    quickSort(arr, l, mid - 1);
    quickSort(arr, mid + 1, r);
}
void qSort(vector<int> &arr) {
    return quickSort(arr, 0, arr.size()-1);
}


// MergeSort 
void mergeSort(vector<int> &a, vector<int> &T, int left, int right)
{
    if (right - left == 1)
        return;
    int mid = left + right >> 1, tmid = left + right >> 1, tleft = left, i = left;
    mergeSort(a, T, left, mid), mergeSort(a, T, mid, right);
    while (tleft < mid || tmid < right)
    {
        if (tmid >= right || (tleft < mid && a[tleft] <= a[tmid]))
        {
            T[i++] = a[tleft++];
        }
        else
        {
            T[i++] = a[tmid++];
        }
    }
    for (int i = left; i < right; i++)
        a[i] = T[i];
}
void mSort(vector<int> &a)
{
    int len = a.size();
    vector<int> T(len);
    mergeSort(a, T, 0, len);
}

// 
void adjustheap(vector<int> arr, int i, int len) {
    int l = 2 * i + 1;
    int r = 2 * i + 2;
    int maxindex = i;
    if (l < len && arr[l] > arr[maxindex]) {
        maxindex = l;
    }
    if (r <len && arr[r] > arr[maxindex]) {
        maxindex = r;
    }
    if (maxindex != i) {
        swap(arr[i], arr[maxindex]);
        adjustheap(arr, maxindex, len);
    }
}

void Sort(vector<int> arr) {
    int len = arr.size() - 1;
    for (int i = len / 2 - 1; i >= 0; i--) {
        adjustheap(arr, i, len);
    }
    for (int i = len - 1; i > 0; i--) {
        swap(arr[0], arr[i]);
        adjustheap(arr, 0, i);
    }
 }

priority_queue< int, vector<int>, greater<int> > q;  // 小顶堆
priority_queue< int, vector<int>, less<int> > q;     // 大顶堆
// 如果需要对结构体的优先级设置，有两种方法：

// 方式一：重载运算符 ‘<’

struct Node
{int adj;
 int val;
 friend  bool operator<(const Node &a,const Node &b) { return  a.val < b.val; }
};
priority_queue<Node>Q;
// 方式二：重载比较函数

struct Node
{int adj;
 int val;
};
struct cmp
{bool operator()(const Node &a,const Node &b) { return  a.val > b.val; }
};
priority_queue<Node,vector<Node>,cmp>Q;
