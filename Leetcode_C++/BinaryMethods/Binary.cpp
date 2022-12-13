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
#include<ext/hash_map>
#include<deque>
using namespace std;

int find(vector<int>& nums, int target) {
    int left = 0, right = nums.size();
    while (left < right) {
        int mid = (right - left) / 2 + left;
        if (nums[mid] < target) left = mid + 1;
        else if (nums[mid] <= target) right = mid;
    }
    return right;
}

class SolutionT875 {
public:
    int minEatingSpeed(vector<int>& piles, int h) {
        // int minK = INT_MAX, maxK = INT_MIN, sum = 0; 
        // for (int i = 0; i < piles.size(); i++) { 
        //     if (piles[i] < minK) minK = piles[i]; 
        //     if (piles[i] > maxK) maxK = piles[i]; 
        //     sum += piles[i]; 
        // } 
        int left = 0, right = 1e9; 
        while (left < right ) {
            int mid = (right - left) / 2 + left;
            if (mid == 0) break;
            int k = 0;
            for (auto pile : piles) {
                k += (pile + mid - 1) / mid; 
            }
            if (k > h) left = mid + 1;
            else right = mid;
        }
        return (right - left) / 2 + left == 0 ? 1:right;
    }
};

class SolutionT33 {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left < right) {
            int mid = (right - left) / 2 + left;
            if (target == nums[mid]) return mid;
            if (nums[mid] > nums[right]) {
                if (nums[left] <= target && target < nums[mid]) right = mid - 1;
                else left = mid + 1;
            } else {
                if (target >= nums[mid] && target <= nums[right]) left = mid + 1;
                else right = mid - 1;
            }
        }
        return nums[left] == target ? left : -1;
    }
};

class SolutionT302 {
public:
    int minArea(vector<vector<char> >& image, int x, int y) {
        
    }

    int binarySearch(vector<vector<char> > &image, int left, int right, int range_start, int range_end, bool opt) {
        int k = range_start;
        while (left < right) {
            int mid = (right - left) / 2 + left;
            while (image[mid][k] != 1) k++;
            if (k < range_end == opt) {
                right = mid;
            } else left = mid + 1;
        }
        return left;
    }
};

class Solution {
public:
    /**
     * @param L: Given n pieces of wood with length L[i]
     * @param k: An integer
     * @return: The maximum length of the small pieces
     */
    int woodCut(vector<int> &L, int k) {
        // write your code here
        if (L.size() == 0) return 0;
        int right = INT_MIN;
        for (auto l : L) right = max(right, l);
        if (right < INT_MAX) right += 1;
        int left = 1; 
        while (left < right) {
            int mid = (right - left) / 2 + left, sum = 0;
            for (auto l : L) {
                sum += l / mid;
            }
            if (sum >= k) {
                left = mid + 1;
            }
            else right = mid;
        }
        left -= 1;
        int temp_sum = 0;
        for (auto l : L) { temp_sum += l/left;}
        return (left > 0 && temp_sum >= k) ? left : 0;
    }
};

class SolutionT658 {
public:
    //反向思维，从数组中去除n-k个元素，肯定是从头尾去除
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        vector<int> res = arr;
        while (res.size() > k) {
            if (x - res.front() <= res.back() - x) {
                res.pop_back();
            } else {
                res.erase(res.begin());
            }
        }
        return res;
    }

    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        auto itr = lower_bound(arr.begin(), arr.end(), x);
        int index = itr - arr.begin();
        vector<int> res({arr[index]});
        int pre = index - 1 >= 0 ? index - 1:-1, next = index+1 < arr.size()?:index+1:arr.size();
        while(k > 0) {
            if (pre < 0 || next >= arr.size()) {
                res.push_back(arr[pre<0?next++:pre--]);
                k--;
            }
            if (abs(x - arr[pre]) <= abs(arr[next] - x)) {
                res.push_back(arr[pre--]);
            } else {
                res.push_back(arr[next++]);
            }
            k--;
        }
        return res;
    }

    //巧妙二分, 这个设计胎牛皮了
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        int l = 0, r = arr.size() - k;
        while (l < r) {
            int mid = l + (r - l)/2;
            if (x - arr[mid] > arr[mid] - x) l = mid + 1;
            else r = mid;
        }
        return vector<int>(arr.begin() + l, arr.begin() + l + k);
    }
};

//目标值必定在一个最大最小的范围之间
//给定一个非负整数数组和一个整数 m，
//你需要将这个数组分成 m 个非空的连续子数组。设计一个算法使得这 m 个子数组各自和的最大值最小。
class SolutionT410 {
public:
    int splitArray(vector<int>& nums, int m) {
        int left = 0, right = 0;
        for (int i = 0; i < nums.size(); i++) {
            left = max(left, nums[i]);
            right += nums[i];
        }
        int res = INT_MAX;
        while (left < right) {
            int mid = left + (right - left)/2;
            if (canSplit(nums, mid, m)) {
                res = min(res, mid);
                right = mid;
            }
            else left = mid + 1;
        }
        return right;
    }

    bool canSplit(vector<int>& nums, int tar, int m) {
        int curSum = 0, cnt = 0;
        for (int i = 0; i < nums.size(); i++) {
            curSum += nums[i];
            if(curSum > tar) {
                cnt++;
                curSum = nums[i];
            }
            if (cnt > m) return false;
        }
        return true;
    }
};

class SolutionT611 {
public:
//二分
    int triangleNumber(vector<int>& nums) {
        int n = nums.size();
        int res = 0;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int sum = nums[i] + nums[j], left = j+1, right = n;
                while (left < right) {
                    int mid = left + (right - left)/2;
                    if (nums[mid] < sum) left = mid + 1;
                    else right = mid;
                }
                res += right - 1 - j;
            }
        }
        return res;
    }
//双指针
    int triangleNumber(vector<int>& nums) {
        int n = nums.size();
        int res = 0;
        sort(nums.begin(), nums.end());
        for (int i = n-1; i >= 2; i--) {
            int right = i-1, left = 0;
            while (left < right) {
                if (nums[left] + nums[right] > nums[i]) {
                    res += right - left;
                    right--;
                } else left++;
            }
        }
        return res;
    }
};

class SolutionT16 {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        int closest = nums[0] + nums[1] + nums[2];
        int diff = abs(closest - target);
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size() - 2; ++i) {
            if (nums[i] * 3 > target) return min(closest, nums[i] + nums[i + 1] + nums[i + 2]);
            int left = i + 1, right = nums.size() - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                int newDiff = abs(sum - target);
                if (diff > newDiff) {
                    diff = newDiff;
                    closest = sum;
                }
                if (sum < target) ++left;
                else --right;
            }
        }
        return closest;
    }
};

class SolutionT374 {
public:
    int guessNumber(int n) {
        if(guess(n) == 0) return n;
        int left = 1, right = n;
        while(left < right) {
            int mid = left + (right - left), t = guess(mid);
            if (t == 0) return mid;
            if (t == -1) //mid < target
                left = mid+1;
            else right = mid-1;
        }
        return left;
    }
};

class Solution {
public:

// If x - A[mid] > A[mid + k] - x,
// it means A[mid + 1] ~ A[mid + k] is better than A[mid] ~ A[mid + k - 1],
// and we have mid smaller than the right i.
// So assign left = mid + 1.
    //这个二分不好想
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        int left = 0, right = arr.size() - k;
        while (left < right) {
            int mid = left + (right - left)/2;
            if ( x - arr[mid] > arr[mid + k] - x ) left = mid + 1;
            else right = mid;
        }
        vector<int> res(arr.begin() + left, arr.begin() + left + k);
        return res;
    }

    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        int left = 0, right = arr.size(), n = arr.size();
        while (left < right) {
            int mid = left + (right - left)/2;
            if (arr[mid] < x) left = mid + 1;
            else right = mid;
        }
        //right 是第一个大于等于x的位置
        //然后开始从左右扫描合适的放进去，用deque
        deque<int> res;
        int l = right - 1, r = right;
        while(res.size() < k) {
            if (l >= 0 && r <= n - 1) {
                if (abs(arr[l] - x) <= abs(arr[r] - x)) res.push_front(arr[l--]);
                else res.push_back(arr[r++]);
            }
            else {
                if (l >= 0) res.push_front(arr[l--]);
                else res.push_back(arr[r++]);
            }
        }
        vector<int> temp(res.begin(), res.end());
        return temp;
    }

    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        deque<int> res(arr.begin(), arr.end());
        while(res.size() > k) {
            if (abs(x - res.front()) <= abs(x - res.back())) res.pop_back();
            else res.pop_front();
        }
        return vector<int> (res.begin(), res.end());
    }
};

class SolutionT719 {
public:

//有最大最小值范围，不算特别大，桶排序
    int smallestDistancePair(vector<int>& nums, int k) {
        int n = nums.size(), N = 1000000;
        vector<int> cnt(N, 0);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                ++cnt[abs(nums[i] - nums[j])];
            }
        }
        for (int i = 0; i < N; ++i) {
            if (cnt[i] >= k) return i;
            k -= cnt[i];
        }
        return -1;
    }

    //将所求值转为二分查找的对象
    int smallestDistancePair(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end());
        int left = 0, right = nums.back() - nums[0], n = nums.size();//right就是距离差的最大范围
        while (left < right) {
            int mid = left + (right - left) / 2, cnt = 0, start = 0;
            for (int i = 0; i < n; i++) {
                while (start < n && nums[i] - nums[start] > mid) start++; //对每个i来说，有多少个距离差小于mid
                cnt += i - start;
            }
            if (cnt < k) left = mid + 1;
            else right = mid;
        }
        return right;
    }
};

class SolutionT668 {
public:
//不是在给出的确定的范围的值里寻找，而是在0 - 可能的最大值之间直接二分。
//难点是怎么比较mid，left， right怎么继续缩小，按照什么样的规则
    int findKthNumber(int m, int n, int k) {
        int left = 0, right = m * n;
        while (left < right) {
            int mid = left + (right - left) / 2, cnt = 0;
            for (int i = 1; i <= m; i++) {
                cnt += (mid > n * i) ? n : mid/i;
            }
            if (cnt < k) left = mid + 1;
            else right = mid;
        }
        return right;
    }

    int findKthNumber(int m, int n, int k) {
         int left = 1, right = m * n;
         while (left < right) {
             int mid = left + (right - left) / 2, cnt = 0, i = m, j = 1;
             while (i >= 1 && j <= n) {
                 if (i * j <= mid) {
                     cnt += i;
                     ++j;
                 } else {
                     --i;
                 }
             }
             if (cnt < k) left = mid + 1;
             else right = mid;
         }
         return right;
     }
};

class SolutionT378 {
public:
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        int m = matrix.size(), n = matrix[0].size();
        int left = matrix[0][0], right = matrix[m-1][n-1];
        while (left < right) {
            int mid = left + (right - left) / 2, cnt = 0; //小于mid的有多少个，跟k比较
            for (int i = 0; i < m; i++) {
                cnt += upper_bound(matrix[i].begin(), matrix[i].end(), mid) - matrix[i].begin();
            }
            if (cnt < k) left = mid +1;
            else right = mid;
        }
        return right;
    }
};

class SolutionT643 {
public:
    double findMaxAverage(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> sum = nums;
        for (int i = 1 ; i < n; i++) {
            sum[i] = sum[i - 1] + nums[i];
        }
        double res = (double)(sum[k-1]/k);
        for (int i = k; i < n; i++) {
            double t = sum[i];
            for (int j = i - k; j >= 0; j--) {
                t = sum[i] - sum[j];
                if (t > res * (i - j)) res = t/(i-j);
            }
        }
        return res;
    }
};

class SolutionT644 {
public:
//brute-force
    double findMaxAverage(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> sums = nums;
        for (int i = 1; i < n; ++i) {
            sums[i] = sums[i - 1] + nums[i];
        }
        double res = (double)sums[k-1] / k;
        for (int i = k; i < n - k + 1; i++) {
            double t = sums[i];
            if (t > res * (i + 1)) res = t / (i + 1);
            for (int j = i - k; j >= 0; j--) {
                t = sum[i] -sum[j];
            }
            if (t > res * (i - j)) res = t / (i - j);
        }
        return res;
    }

//平均值的最大最小范围就是数组中值的最大值和最小值
    double findMaxAverage(vector<int>& nums, int k) {
        int n = nums.size();
        vector<double> sums(n + 1, 0);
        double left = *min_element(nums.begin(), nums.end());
        double right = *max_element(nums.begin(), nums.end());
        while (right - left > 1e-5) {
            double minSum = 0, mid = left + (right - left)/2;
            bool check = false;
            for (int i = 1; i < n; i++) {
                sums[i] = sums[i-1] + nums[i-1] - mid;
                if (i >= k) minSum = min(minSum, sums[i-k]);
                if (i >= k && sums[i] - minSum> 0) {check=true; break}; //表明（每个数-中位数）的的累加和
                // > 0，即这个长度大于k的子数组，他的平均值比mid大
            }
            if (check) left = mid + 1;
            else right = mid;
        }
        return left;
    }
};

class SolutionT287 {
public:
    //有最大最小值范围，折半方法为计数
    int findDuplicate(vector<int>& nums) {
        int left = 1, right = nums.size();
        while (left < right){
            int mid = left + (right - left) / 2, cnt = 0;
            for (int num : nums) {
                if (num <= mid) ++cnt;
            }
            if (cnt <= mid) left = mid + 1;
            else right = mid;
        }    
        return right;
    }

    //快慢指针成环原理
    int findDuplicate(vector<int>& nums) {
        int slow = 0, fast = 0, t = 0;
        while(true) {
            slow = nums[slow];
            fast = nums[nums[fast]];
            if (slow == fast) break;
        }
        while(true) {
            slow = nums[slow];
            t = nums[t];
            if (slow == t) break;
        }
        return slow;
    }
};

class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        vector<int> gas_sum = gas, cost_sum = cost;
        for (int i = 1; i < gas.size(); i++) {
            gas_sum[i] += gas_sum[i-1];
            cost_sum[i] += cost_sum[i-1];
        }
        int res = gas.size() - 1;
        if (gas_sum.back() < cost_sum.back()) return -1;
        for (int i = gas.size() - 1; i >= 0; i--) {
            if (gas_sum[i] < cost_sum[i]) break;
            res = i;
        }
        return res;
    }
};

class SolutionT778 {
public:
    int swimInWater(vector<vector<int>>& grid) {

    }
};

class SolutionT875 {
public:
    int minEatingSpeed(vector<int>& piles, int H) {
        int minK = INT_MAX, maxK = INT_MIN, sum = 0;
        for (int i = 0; i < piles.size(); i++) {
            if (piles[i] < minK) minK = piles[i];
            if (piles[i] > maxK) maxK = piles[i];
            sum += piles[i];
        }
        int left = minK, right = maxK;
        while(left < right) {
            int mid = left + (right - left) / 2, cnt = 0;
            for (int pile : piles) cnt += (pile + mid - 1) / mid;
            if (cnt > H) left = mid + 1;
            else right = mid;
        }
    }
};

class SolutionT302 {
public:
    int minArea(vector<vector<char>>& image, int x, int y) {
        int m = image.size(), n = image[0].size();
        int up = binary_search(image, true, 0, x, 0, n, true);
        int down = binary_search(image, true, x + 1, m, 0, n, false);
        int left = binary_search(image, false, 0, y, up, down, true);
        int right = binary_search(image, false, y + 1, n, up, down, false);
        return (right - left) * (down - up);
    }
    //h 行还是列, i, j 二分的主范围，中心点距离各个边界的距离，上下左右，low，high是遍历距离，opt表明是否越界
    int binary_search(vector<vector<char>> &image, bool h, int i, int j, int low, int high, bool opt) {
        while (i < j) {
            int mid = i + (j - i)/2, k = low;
            while(k < high && (h ? image[mid][k]:image[k][mid]) == '0') k++;
            if (k < high == opt) j = mid;
            else i = mid + 1;
        }
        return i;
    }
};