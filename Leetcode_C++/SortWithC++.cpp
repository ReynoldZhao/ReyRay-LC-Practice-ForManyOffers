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

// QuickSort
int partition(vector<int> &arr, int l, int r) {
    int pivot = arr[r];
    //以l-1作为指针的起始点，以后每次遍历，都只用swap(arr[++i],arr[j])
    //非常直观
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

//基于随机的快排
//这里其实是找到kth最大
class SolutionT215 {
public:
    int partition(vector<int>& nums, int l, int r) {
        int pivot = nums[r];
        int i = l - 1;
        //起始点是 l - 1
        //方便后面遍历，与swap交换，最稳妥的快排
        for (int j = l; j <= r - 1; ++j) {
            if (nums[j] <= pivot) {
                i = i + 1;
                swap(nums[i], nums[j]);
            }
        }
        swap(nums[i + 1], nums[r]);
        return i + 1;
    }
    // 基于随机的划分
    
    int randomized_partition(vector<int>& nums, int l, int r) {
        int i = rand() % (r - l + 1) + l;
        swap(nums[r], nums[i]);
        return partition(nums, l, r);
    }
    //注意 k一直指的是第k个
    void randomized_selected(vector<int>& arr, int l, int r, int k) {
        if (l >= r) return ;
        int pos = randomized_partition(arr, l, r);
        if (pos == k - 1) {
            return ;
        }
        if (pos < k - 1) {
            randomized_selected(arr, l, pos - 1, k);
        } else {
            randomized_selected(arr, pos + 1, r, k - pos - 1);
        }
    }
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        srand((unsigned)time(NULL));
        randomized_selected(arr, 0, (int)arr.size() - 1, k);
        vector<int> vec;
        for (int i = 0; i < k; ++i) {
            vec.push_back(arr[i]);
        }
        return vec;
    }
};

class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        int left = 0, right = nums.size() - 1;
        while (true) {
            int pos = partition(nums, left, right);
            if (pos == k - 1) return nums[pos];
            if (pos > k - 1) right = pos - 1;
            else left = pos + 1;
        }
    }
    int partition(vector<int>& nums, int left, int right) {
        int pivot = nums[left], l = left + 1, r = right;
        while (l <= r) {
            if (nums[l] < pivot && nums[r] > pivot) {
                swap(nums[l++], nums[r--]);
            }
            if (nums[l] >= pivot) ++l;
            if (nums[r] <= pivot) --r;
        }
        swap(nums[left], nums[r]);
        return r;
    }
};

//三路快排
public void partition2(int[] nums, int low, int high) {
        // write your code here
        if (nums == null ||  nums.length == 0) {
            return;
        }
        
        int l = -1, i = 0, r = nums.length;
        while (i < r) {
        	// 如果当前数字小于low，那么就将其换到左边；
        	// 由于我们已经保证了A(l, i]在low与high之间，所以交换完毕后，A[i]已经在low与high之间了，所以将i也右移
        	// 如果当前数字大于high，则将其换到右边；如果当前数字在low与high直接，就直接右移i
            if (nums[i] < low) {
                swap(nums, i++, ++l);
            } else if (nums[i] > high) {
                swap(nums, i, --r);
            } else {
                i++;
            }
        }
    }

    int partition2(int[] nums, int low, int high) {
        // write your code here
        if (nums == null ||  nums.length == 0) {
            return;
        }
        
        int l = -1, i = 0, r = nums.length;
        while (i < r) {
        	// 如果当前数字小于low，那么就将其换到左边；
        	// 由于我们已经保证了A(l, i]在low与high之间，所以交换完毕后，A[i]已经在low与high之间了，所以将i也右移
        	// 如果当前数字大于high，则将其换到右边；如果当前数字在low与high直接，就直接右移i
            if (nums[i] < low) {
                swap(nums, i++, ++l);
            } else if (nums[i] > high) {
                swap(nums, i, --r);
            } else {
                i++;
            }
        }
        return i - 1;
    }
    
    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }


class SolutionT215KthLargest {
public:
    int findKthLargest(vector<int>& nums, int k) {
        int n = nums.size(), target = n - k;
        return quickSelect(nums, 0, n-1, target);
    }

    int quickSelect(vector<int>& nums, int l, int r, int target) {
        int pos = partition(nums, l, r);
        if (pos == target) return nums[pos];
        else {
            return pos < target ? quickSelect(nums, pos + 1, r, target) : quickSelect(nums, l, pos - 1, target);
        }
    }

    //这个partition其实很绕了
    int partition(vector<int>& nums,int left, int right){
        int pivot = nums[right];
        int l = left, r = right-1;
        while (l <= r) {
            if (nums[l] > pivot && nums[r] < pivot) {
                swap(nums[l++], nums[r--]);
            }
            //这个真的不好理解
            if (nums[l] <= pivot) l++;
            if (nums[r] >= pivot) r--;
        }
        swap(nums[l], nums[right]);
        return l;
    }
};

//Three way QuickSort

void quickSort3Way(vector<int> arr, int l, int r) {
 
	if (l >= r)
		return;
 //随机选择要做比较的值
	swap(arr[l], arr[rand() % (r - l + 1) + l]);
	int v = arr[l];
	int lt = l;     // arr[l+1...lt] < v
    int rt = r + 1; // arr[rt...r] > v
    int i = l+1;    // arr[lt+1...i) == v
    //已有序游标为lt，如果pivot是最右值，游标初始值为l - 1;
    // --- lt(最后一个小于) ---- rt（第一个大于） ----
	while (i<rt) {
		if (arr[i] < v) {
			swap(arr[i], arr[lt+1]);
			lt++;
			i++;
		}
		if (arr[i] == v) {
			i++;
		}
		if (arr[i] > v) {
			swap(arr[i], arr[rt-1]);
			rt--;
            //注意这里i不变化
		}
 
	}
	swap(arr[l], arr[lt]);
	quickSort3Way(arr, l, lt-1 );
	quickSort3Way(arr, rt, r);
 
}



void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
    return ;
}

void bubbleSort(vector<int> &arr) {
    for (int i = 0; i < arr.size() - 1; i++) {
        for (int j = 0; j  < arr.size() - i - 1; j++) {
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
    for (int i = 0; i < arr.size() - 1; i++) {
        int j = i;
        int temp = a[i+1];
        while (j >= 0 && a[j] > temp) {
            a[j + 1] = a[j];
            j--
        }
        a[j+1] = temp;
    }
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




//T148 链表的归并快慢指针 有坑
void MergeSort (int arr [], int low,int high) {
    if(low>=high) { return; } // 终止递归的条件，子序列长度为1
    int mid =  low + (high - low)/2;  // 取得序列中间的元素
    MergeSort(arr,low,mid);  // 对左半边递归
    MergeSort(arr,mid+1,high);  // 对右半边递归
    merge(arr,low,mid,high);  // 合并
}

void merge(int arr[],int low,int mid,int high){
    //low为第1有序区的第1个元素，i指向第1个元素, mid为第1有序区的最后1个元素
    int i=low,j=mid+1,k=0;  //mid+1为第2有序区第1个元素，j指向第1个元素
    int *temp=new int[high-low+1]; //temp数组暂存合并的有序序列
    while(i<=mid&&j<=high){
        if(arr[i]<=arr[j]) //较小的先存入temp中
            temp[k++]=arr[i++];
        else
            temp[k++]=arr[j++];
    }
    while(i<=mid)//若比较完之后，第一个有序区仍有剩余，则直接复制到t数组中
        temp[k++]=arr[i++];
    while(j<=high)//同上
        temp[k++]=arr[j++];
    for(i=low,k=0;i<=high;i++,k++)//将排好序的存回arr中low到high这区间
	arr[i]=temp[k];
    delete []temp;//释放内存，由于指向的是数组，必须用delete []
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

//手撸堆排序
/**
 *  堆排
 */
public class HeapSort {
    public static void heapSort(int[] arr) {
        if (arr == null || arr.length < 2) {
            return;
        }
        for (int i = 0; i < arr.length; i++) {
            heapInsert(arr, i);
        }
        int size = arr.length;
        swap(arr, 0, --size);
        while (size > 0) {
            heapify(arr, 0, size);
            swap(arr, 0, --size);
        }
    }
//大顶
    public static void heapInsert(int[] arr, int index) {
        while (arr[index] > arr[(index - 1) / 2]) {
            swap(arr, index, (index - 1) / 2);
            index = (index - 1) / 2;
        }
    }
//大顶堆
    public static void heapify(int[] arr, int index, int size) {
        int left = index * 2 + 1;
        while (left < size) {
            int largest = left + 1 < size && arr[left + 1] > arr[left] ? left + 1 : left;
            largest = arr[largest] > arr[index] ? largest : index;
            if (largest == index) {
                break;
            }
            swap(arr, largest, index);
            index = largest;
            left = index * 2 + 1;
        }
    }

    public static void swap(int[] arr, int i, int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    public static void main(String[] args) {
        int []arr={1,34,4,5,76,8,9};
        heapSort(arr);
        System.out.println(Arrays.toString(arr));
    }
}

class HeapSort {
public:
    void heapSort(vector<int> &arr) {
        if (arr.empty() || arr.size() < 2) {
             return ;
        }
        int n = arr.size();
        for (int i = 0; i < arr.size(); i++) {
            heapInsert(arr, i);
        }
        swap(arr[0], arr[--n]);
        while (n > 0) {
            heapify(arr, 0, n);
            swap(arr[0], arr[--n]);
        }
    }
private:
//bottom - up
    void heapInsert(vector<int> &arr, int index) {
        while ((index-1)/2 >= 0 && arr[index] > arr[(index-1)/2]) {
            swap(arr[index], arr[(index-1)/2]);
            index = (index-1)/2;
        }
    }
//up-down
    void heapify(vector<int> &arr, int index, int size) {
        int child = index * 2 + 1;
        while (child < size) {
            int largest = left + 1 < size && arr[left + 1] > arr[left] ? left + 1 : left;
            largest = arr[largest] > arr[index] ? largest:index;
            if (largest == index) break;
            swap(arr[largest], arr[index]);
            index = largest;
            child = index*2 + 1;
        }
    }
}
