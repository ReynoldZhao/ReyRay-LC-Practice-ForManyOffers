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

https://www.cnblogs.com/grandyang/p/4465932.html
https://blog.csdn.net/zxzxy1988/article/details/8587244

class Solution {
private:
    double findKth(vector<int>& nums1, vector<int>& nums2, int start1, int len1, int start2, int len2, int k) {
        if (len1 > len2) {
            return findKth(nums2, nums1, start2, len2, start1, len1, k);
        }

        if (len1 == 0) {
            return nums2[start2 + k - 1];
        }

        if (k == 1) {
            return min(nums1[start1], nums2[start2]);
        }

        int p1 = min(k / 2, len1);
        int p2 = k - p1;
        if (nums1[start1 + p1 - 1] > nums2[start2 + p2 - 1]) {
            return findKth(nums1, nums2, start1, len1, start2 + p2, len2 - p2, k - p2);
        }
        else if(nums1[start1 + p1 - 1] < nums2[start2 + p2 - 1]){
            return findKth(nums1, nums2, start1 + p1, len1 - p1, start2, len2, k - p1);
        }
        else {
            return nums1[start1 + p1 - 1];
        }

    }

public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int len = nums1.size() + nums2.size();

        if (!(len & 0x01)) {
            return (findKth(nums1, nums2, 0, nums1.size(), 0, nums2.size(), len / 2)
                + findKth(nums1, nums2, 0, nums1.size(), 0, nums2.size(), len / 2 + 1)
                ) / 2.0f;
        }
        else {
            return findKth(nums1, nums2, 0, nums1.size(), 0, nums2.size(), len / 2 + 1);
        }
    }
};

double findkth(int a[],int m,int b[],int n,int k){
	if(m>n)
		return findkth(b,n,a,m,k);
	if(m==0) return b[k-1];
	if(k==1) return min(a[0],b[0]);
	int pa = min(m,k/2);
	int pb = k-m;
	if(a[pa]<b[pb]) return findkth(a+pa,m-pa,b,n,k-pa);
	else if(a[pa]>b[pb]) return findkth(a,m,b+pb,n-pb,k-pb);
	else return a[pa];
}

class Solution
{
public:
	double findMedianSortedArrays(int A[], int m, int B[], int n)
	{
		int total = m + n;
		if (total & 0x1)
			return findKth(A, m, B, n, total / 2 + 1);
		else
			return (findKth(A, m, B, n, total / 2)
					+ findKth(A, m, B, n, total / 2 + 1)) / 2;
	}
};
--------------------- 
作者：zxzxy1988 
来源：CSDN 
原文：https://blog.csdn.net/zxzxy1988/article/details/8587244 
版权声明：本文为博主原创文章，转载请附上博文链接！
