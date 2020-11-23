#include<iostream>
#include<vector>
using namespace std;
class Solution
{
	public:
		int binarySearch(vector<int> nums, int val){
			int low = 0,high = nums.size()-1,mid;
			while(low<high)
			{
				mid = (low+high)/2;
				if(nums[mid] = val) return mid;
				else if(nums[mid]>val){
					high = mid - 1;
				} 
				else{
					low = mid+1;
				}
			}
			return -1;
		}
 } 
