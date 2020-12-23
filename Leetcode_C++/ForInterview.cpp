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
#include<unordered_set>
#include<hash_map>
#include<deque>
#include<list>
using namespace std;

//T215. 数组中的第K个最大元素
class SolutionT215 {
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

    int partition(vector<int>& nums,int left, int right){
        int pivotIndex = rand()%(right - left + 1) + left;
        int pivot = nums[pivotIndex];
        swap(nums[right], nums[pivotIndex]);
        int l = left, r = right-1;
        while (l <= r) {
            while(nums[l] < pivot && l < nums.size() - 1) l++;
            while(nums[r] > pivot && r > 0) r--;
            if (l < r) swap(nums[l], nums[r]);
        }
        swap(nums[l], nums[right]);
        return l;
    } //二分形式的partition（好像在处理长度为2的数组时有些问题）

    int partition(vector<int>& nums,int left, int right){
        int pivot = nums[right];
        int l = left, r = right-1;
        while (l <= r) {
            while(nums[l] < pivot && l < nums.size() - 1) l++;
            while(nums[r] > pivot && r > 0) r--;
            if (l < r) swap(nums[l], nums[r]);
        }
        swap(nums[l], nums[right]);
        return l;
    }

 // ***************记这个模板好了 遍历快排*****************
    inline int partition(vector<int>& a, int l, int r) {
        int x = a[r], i = l - 1;
        for (int j = l; j < r; ++j) {
            if (a[j] <= x) {
                swap(a[++i], a[j]);
            }
        }
        swap(a[i + 1], a[r]);
        return i + 1;
    } //一遍遍历的partition 最稳妥的
 // ***************记这个模板好了 遍历快排*****************
    
    int findKthLargest(vector<int>& nums, int k) {
        int left = 0, right = nums.size() - 1;
        while (true) {
            int pos = partition(nums, left, right);
            if (pos == k - 1) return nums[pos];
            if (pos > k - 1) right = pos - 1;
            else left = pos + 1;
        }
    }
    // ***************记这个模板好了 二分快排*****************
    //大到小
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
    //小到大
    int partition(vector<int>& nums,int left, int right){
        int pivot = nums[right];
        int l = left, r = right-1;
        while (l <= r) {
            if (nums[l] > pivot && nums[r] < pivot) {
                swap(nums[l++], nums[r--]);
            }
            if (nums[l] <= pivot) l++;
            if (nums[r] >= pivot) r--;
        }
        swap(nums[l], nums[right]);
        return l;
    }
    // ***************记这个模板好了*****************

    //这个二分版本就是错的
    int partition(vector<int>& nums, int left, int right) {
        int pivot = nums[left], l = left + 1, r = right;
        while (l <= r) {
            while(nums[l] < pivot && l < nums.size() - 1) l++;
            while(nums[r] > pivot && r > 0) r--;
            if (l <= r) swap(nums[l], nums[r]);
        }
        swap(nums[left], nums[r]);
        return r;
    }

    //小顶堆实现
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int, vector<int>, greater<int>> minHeap;
        for (int num : nums) {
            if (minHeap.size() < k) {
                minHeap.push(num);
            } else if (minHeap.top() < num) {
                minHeap.pop();
                minHeap.push(num);
            }
        }
        return minHeap.top();
    }

    //荷兰国旗
    vector<int> partition(vector<int> nums, int l, int r, int pivot) {
        int left = l - 1;
        int right = r + 1;
        while(l < right) {
            if (nums[l] < pivot) {
                swap(nums[++left], nums[l++]);
            } else if (nums[l] > pivot) {
                swap(nums[--right], nums[l]);
            } else {
                l++;
            }
        }
        return vector<int> ({left+1, right-1});
    }
};

//T206 翻转链表
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {} 
};

class SolutionT206 {
public:
    ListNode* reverseList(ListNode* head) {
        if (!head->next) return head;
        ListNode* dummy = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return dummy;
    }
};

//滑动窗口
class SolutionT3 {
public:
    int lengthOfLongestSubstring(string s) {
        vector<int> bucket(128, -1);
        int start = 0, end = 0, mx = 0;
        while (end < s.size()) {
            if (bucket[s[end]]==-1) {
                bucket[s[end]] = end;
                end++;
            } else {
                int repeatPos = bucket[s[end]];
                while(start <= repeatPos) {
                    bucket[s[start]] = -1;
                    start++;
                }
                bucket[s[end]] = end;
                end++;
            }
            mx = max(mx, end - start);
        }
        return mx;
    }

    //直接用map记录位置，不用清零，只要大于当前left即可
    int lengthOfLongestSubstring(string s) {
        int res = 0, left = -1, n = s.size();
        unordered_map<int, int> m;
        for (int i = 0; i < n; ++i) {
            if (m.count(s[i]) && m[s[i]] > left) {
                left = m[s[i]];  
            }
            m[s[i]] = i;
            res = max(res, i - left);            
        }
        return res;
    }

    int lengthOfLongestSubstring(string s) {
        int res = 0, left = -1, n = s.size();
        vector<int> map(128, -1);
        for (int i = 0; i < s.size(); i++) {
            if (map[s[i]] > 0 && map[s[i]] >left) {
                left = map[s[i]];
            }
            map[s[i]] = i;
            res = max(res,i - left);
        }
        return res;
    }
};

//链表
class SolutionT25 {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        if (!head) return nullptr;
        if (k==1) return head;
        ListNode *cur = nullptr, *pre = head;
        for (int i = 1; i <= k; i++) {
            ListNode* temp = pre->next;
            pre->next = cur;
            cur = pre;
            pre = temp;
            if(!pre) break;
        }
        head->next = reverseKGroup(pre, k);
        return cur;
    }
};

//系统设计
//哈希+双向链表实现
class LRUCacheT146 {
public:
    LRUCache(int capacity) {
        cap = capacity;
    }
    
    int get(int key) {
        if (map.find(key) == map.end()) return -1;
        l.splice(l.begin(), l, map[key]);
        return map[key]->second;
    }
    
    void put(int key, int value) {
        if (map.find(key) != map.end()) {
            l.erase(map[key]);
        }
        l.push_front(make_pair(key,value));
        map[key] = l.begin();
        if (map.size() > cap) {
            int k = l.rbegin()->first;
            map.erase(k);
            l.pop_back();
        }
    }
private:
    int cap;
    list<pair<int,int>> l; //list里面放的key-value pair
    unordered_map<int, list<pair<int, int>>::iterator> map;//放的key {key-value} =》双向链表的node
};

//java 使用linkedHashMap
// class LRUCache extends LinkedHashMap<Integer, Integer>{
//     private int capacity;
    
//     public LRUCache(int capacity) {
//         super(capacity, 0.75F, true);
//         this.capacity = capacity;
//     }

//     public int get(int key) {
//         return super.getOrDefault(key, -1);
//     }

//     public void put(int key, int value) {
//         super.put(key, value);
//     }

//     @Override
//     protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
//         return size() > capacity; 
//     }
// }

//python 使用OrderedDict
// class LRUCache(collections.OrderedDict):

//     def __init__(self, capacity: int):
//         super().__init__()
//         self.capacity = capacity


//     def get(self, key: int) -> int:
//         if key not in self:
//             return -1
//         self.move_to_end(key)
//         return self[key]

//     def put(self, key: int, value: int) -> None:
//         if key in self:
//             self.move_to_end(key)
//         self[key] = value
//         if len(self) > self.capacity:
//             self.popitem(last=False)

//2 Sum
class SolutionT1 {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> res;
        unordered_map<int, int> map;
        for (int i = 0; i < nums.size(); i++) {
            map[nums[i]] = i; 
        }
        for (int i = 0; i < nums.size(); i++) {
            if (map.find(target - nums[i]) != map.end()) {
                return vector<int> ({i, map[target - nums[i]]});
            }
        }
        return res;
    }
};

//3sum
class SolutionT15 {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> res;
        if (nums.size() <= 2) return res;
        int sum = 0;
        for (int i = 0; i <= nums.size()-3; i++) {
            if (nums[i] > 0) break;
            if (i > 0 && nums[i] == nums[i-1]) continue;
            int temp_target = 0 - nums[i], left = i+1, right = nums.size() - 1;
            while (left < right) {
                //while (left > i + 1 && nums[left] == nums[left-1]) left++;
                if (left > i+1 && nums[left] == nums[left - 1]) {
                    left++;
                    continue;
                }
                sum = nums[left] + nums[right];
                if (sum == temp_target) {
                    res.push_back({nums[i], nums[left], nums[right]});
                    left++;
                }
                else if (nums[left] + nums[right] > temp_target) right--;
                else left++;
            }
        }
        return res;
    }
};

//中序遍历
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> s;
        TreeNode* p = root;
        while(p || !s.empty()) {
            while(p) {
                s.push(p);
                p = p->left;
            }
            p = s.top(); s.pop();
            res.push_back(p->val);
            p = p->right;
            //左子必处理完了，右子处理完为空，top为新的
        }
    }

    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> s;
        TreeNode* p = root;
        while(p || !s.empty()) {
            while(p) {
                res.push_back(p->val);
                s.push(p);
                p = p->left;
            }
            p = s.top(); s.pop();
            p = p->right;
        }
    }

    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> s({root});
        TreeNode* p = root;
        while(!s.empty()) {
            while (p) {
                res.insert(res.begin(), p->val);
                p = p->right;
            }
            p = s.top(); s.pop();
            p = p->left;
        }
    }
};

//环形链表的规律，快慢指针相遇后，从头与慢指针同速一起走
// class Solution(object):
//     def detectCycle(self, head):
//         fast, slow = head, head
//         while True:
//             if not (fast and fast.next): return
//             fast, slow = fast.next.next, slow.next
//             if fast == slow: break
//         fast = head
//         while fast != slow:
//             fast, slow = fast.next, slow.next
//         return fast

//递归
class SolutionT236 {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root || root == p || root == q) return root;
        TreeNode* left = lowestCommonAncestor(root->left, p, q);
        TreeNode* right = lowestCommonAncestor(root->right, p, q);
        if (p && q) return root;
        return left?left:right;
    }
};

class SolutionT121 {
public:
    int maxProfit(vector<int>& prices) {
        int buy = INT_MAX, maxProfit = 0;
        for (int i = 0; i < prices.size(); i++) {
            if (prices[i] < buy) {
                buy = prices[i];
                continue;
            } else {
                maxProfit = max(maxProfit, prices[i] - buy);
            }
        }
        return maxProfit;
    }
};

class SolutionT199 {
public:
    vector<int> rightSideView(TreeNode* root) {
        queue<TreeNode*> q({root});
        vector<int> res;
        if (!root) return res;
        while(!q.empty()) {
            int size = q.size();
            for (int i = 0 ; i < size; i++) {
                TreeNode* temp = q.front(); q.pop();
                if (i == size - 1) res.push_back(temp->val);
                if (temp->left) q.push(temp->left);
                if (temp->right) q.push(temp->right);
            }
        }
        return res;
    }
};

//相交链表
class SolutionT160 {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode* startA = headB;
        ListNode* startB = headA;
        if (!headA || !headB) return nullptr;
        while(startA != startB) {
            startA = (startA != nullptr) ? startA->next:headA;
            startB = (startB != nullptr) ? startB->next:headB;
        }
        return startA;
    }
};

class SolutionT92 {
public:
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        ListNode *cur = head, *pre = nullptr, *end = nullptr;
        for (int i = 1; i < m; i++) {
            pre = cur;
            cur = cur->next;
        }
        ListNode* tail = cur;
        for (int i = m; i < n; i++) {
            ListNode*  nextNode= tail->next;
            tail->next = nextNode->next;
            nextNode->next = pre->next;
            pre->next = nextNode;
        }
        return head;
    }
};

class SolutionT543 {
public:
    int diameterOfBinaryTree(TreeNode* root) {
        int maxD = INT_MIN;
        helper(root, maxD);
        return maxD;
    }

    int helper(TreeNode* root, int &maxD) {
        if(!root) return 0;
        int leftD = helper(root->left, maxD);
        int rightD = helper(root->right, maxD);
        maxD = max(maxD, leftD + rightD + 1);
        return max(leftD+1, rightD+1);
    }
};

class HeapSort {
public:
    void heapSort(vector<int> &arr) {
        if (arr.size() < 2) return ;
        for (int i = 0; i < arr.size(); i++) {
            heapInsert(arr, i);
        }
        int size = arr.size()
        swap(arr[0], arr[--size]);
        //每次把最后一个元素找对位置
        while (size > 0) {
            heapify(arr, 0, size);
            swap(arr[0], arr[--size]);
        }
    }

    void heapInsert(vector<int> &arr, int index) {
        int father = (index - 1) / 2;
        while (arr[index] > arr[father] && index >= 0) {
            swap(arr[index], arr[father]);
            index = father;
        }
    }

    void heapify(vector<int> &arr, int index, int size) {
        int child = index*2 + 1;
        while(child < size) {
            int larger = child + 1 < size ? (arr[child +1] > arr[child] ? (child + 1):child):child;
            larger = arr[index] > arr[larger] ? index : larger;
            if (larger == index) {
                break;
            }
            swap(arr[larger], arr[index]);
            index = larger;
            child = index * 2 + 1;
        }
    }
};

//滑动窗口

//最大值
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;
        deque<int> dq;
        for (int i = 0; i < k ;i ++) {
            while (!dq.empty() && nums[i] >= nums[dq.back()]) dq.pop_back();
            if (dq.empty() || nums[i] > nums[dq.back()]) dq.push_back(i);
        }
        res.push_back(nums[dq.front()]);
        for (int i = k; i < nums.size(); i++) {
            while(!dq.empty() && nums[i] >= nums[dq.back()]) dq.pop_back();
            if (dq.front() <= i - k) dq.pop_front();
            if (dq.empty() || nums[i] > nums[dq.back()]) dq.push_back(i);
            res.push_back(nums[dq.front()]);
        }
        return res;
    }
};

//中位数

class SolutionT480 {
public:
    vector<double> medianSlidingWindow(vector<int>& nums, int k) {
        vector<double> res;
        multiset<double> set(nums.begin(), nums.begin() + k);
        auto mid = next(nums.begin(), k / 2); //永远指向后一个的那个 3, 0 + 1 = 1 // 4, 0 + 2 = 2(1,2)
        for (int i = k; i < nums.size(); i++) {
            res.push_back((*mid + *prev(mid, 1 - k%2)/2));
            if (i == nums.size()) return res;
            if (nums[i] < *mid) --mid;
            if (nums[i - k] <= *mid) ++mid;
            set.erase(set.lower_bound(nums[i-k]));
        }
    }

    vector<double> medianSlidingWindow(vector<int>& nums, int k) {
        vector<double> res;
        multiset<int> small, large;
        for (int i = 0; i < nums.size(); ++i) {
            if (i >= k) {
                if (small.count(nums[i - k])) small.erase(small.find(nums[i - k]));
                else if (large.count(nums[i - k])) large.erase(large.find(nums[i - k]));
            }
                if (small.size() <= large.size()) {
                    if (large.empty() || nums[i] <= *large.begin()) small.insert(nums[i]);
                    else {
                        small.insert(*large.begin());
                        large.erase(large.begin());
                        large.insert(nums[i]);
                    }
                } else {
                    if (nums[i] >= *small.rbegin()) large.insert(nums[i]);
                    else {
                        large.insert(*small.rbegin());
                        small.erase(--small.end());
                        small.insert(nums[i]);
                    }
                }

            if (i >= (k - 1)) {
                if (k % 2) res.push_back(*small.rbegin());
                else res.push_back(((double)*small.rbegin() + *large.begin()) / 2);
            }
        }
    }
};

//T295 数据流的中位数
class MedianFinder {
public:
    /** initialize your data structure here. */
    MedianFinder() {

    }
    
    void addNum(int num) {
        maxHeap.push(num);
        minHeap.push(maxHeap.top());
        maxHeap.pop();
        if (maxHeap.size() < minHeap.size()) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
    }
    
    double findMedian() {
        return maxHeap.size() == minHeap.size()?0.5*(maxHeap.top()+minHeap.top()):minHeap.top();
    }
private:
    priority_queue<int, vector<int>, less<int>> maxHeap;
    priority_queue<int, vector<int>, greater<int>> minHeap;
};

//骰子概率
class Solution {
public:
    vector<double> twoSum(int n) {
        vector<double> pre = {double(1/6),double(1/6),double(1/6),double(1/6),double(1/6),double(1/6)};
        for (int i = 2; i <= n; i++) {
            vector<double> temp(5*i+1);
            for (int j = 0; j <pre.size(); j++) {
                for (int x = 0; x < 6; x++) {
                    temp[j + x] += pre[j]/6;
                }
                pre = temp;
            }
        }
        return pre;
    }
};

class Solution {
public:
    int lastRemaining(int n, int m) {
        vector<int> v;
        for (int i = 0; i < n; i++) {
            v.push_back(i);
        }
        int index = 0;
        while(n > 1) {
            index = (index + m - 1)%n;
            v.erase(v.begin() + index);
            n--;
        }
        return v.back();
    }
};

class SolutionT03 {
public:
    int findRepeatNumber(vector<int>& nums) {
        int temp;
        for (int i = 0; i < nums.size(); i++) {
            while(nums[i]!=i) {
                if(nums[i] == nums[nums[i]]) return nums[i];
                temp = nums[i];
                nums[i] = nums[temp];
                nums[temp] = temp;
            }
        }
    }
};


class Solution {
    public int findRepeatNumber(int[] nums) {
        int temp;
        for(int i=0;i<nums.length;i++){
            while (nums[i]!=i){
                if(nums[i]==nums[nums[i]]){
                    return nums[i];
                }
                temp=nums[i];
                nums[i]=nums[temp];
                nums[temp]=temp;
            }
        }
        return -1;
    }
}

class Node {
public:
    int val;
    Node* left;
    Node* right;

    Node() {}

    Node(int _val) {
        val = _val;
        left = NULL;
        right = NULL;
    }

    Node(int _val, Node* _left, Node* _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};

class SolutionTOffer36 {
public:

    Node* head = nullptr;
    Node* pre = nullptr;
    Node* treeToDoublyList(Node* root) {
        if(!root) return nullptr;
        dfs(root);
        head->left = pre;
        pre->right = head;
    }

    void dfs(Node* cur) {
        if(!cur) return;
        dfs(cur->left);
        if(pre) {
            pre->right = cur;
            cur->left = pre;
        } else head = cur;
        pre = cur;
        dfs(cur->right);
    }
};

//字符串排列，两个高效的剪枝
class Solution {
public:
    vector<string> permutation(string s) {
        vector<string> res;

        dfs(res,s,0);
       
        return res;
    }

    void  dfs(vector<string> &res,string &s,int pos){
        if(pos == s.size())
            res.push_back(s);

        for(int i=pos;i<s.size();i++){
            bool flag = true;
            for(int j = pos;j<i;j++)//字母相同时，等效，剪枝
                if(s[j] == s[i])
                    flag = false;
            if(flag){
                swap(s[pos],s[i]);
                dfs(res,s,pos+1);
                swap(s[pos],s[i]);

            }
        }
    }
};

class Solution {
public:
    vector<string> res;
    vector<string> permutation(string s) {
        vector<char> temp;
        for(int i = 0;i < s.length();i++)
            temp.push_back(s[i]);
        sort(temp.begin(),temp.end(),compare);
        dfs(temp,0);
        return res;
    }
    void dfs(vector<char> temp,int left){
        if(left == temp.size()-1){
            string s;
            for(int i = 0;i < temp.size();i++)
                s += temp[i];
            res.emplace_back(s);
            return;
        }
        for(int i = left;i < temp.size();i++){
            if(i != left && temp[left] == temp[i])
                continue;
            swap(temp[left],temp[i]);
            dfs(temp,left+1);
        }
    }
    static bool compare(const char& a,const char& b){
        return a <= b;
    }
};

//my 回溯
class Solution {
public:
    vector<string> permutation(string s) {
        vector<string> res;
        helper(s, 0, res);
        return res;
    }

    void helper(string &s, int index, vector<string> &res) {
        if (index == s.size() - 1) {
            res.push_back(s);
            return;
        }
        for (int i = index; i < s.size(); i++) {
            if (i!=index && s[index] == s[i]) continue;
            swap(s[i], s[index]);
            helper(s, index+1, res);
            swap(s[i], s[index]);
        }
    }
};

//第N位数字，这个n-1太妙了
class Solution {
public:
    int findNthDigit(int n) {
        long long len = 1, cnt = 9, start = 1;
        while (n > len * cnt) {
            n -= len*cnt;
            len++;
            cnt*=10;
            start*=10;
        }
        start += (n - 1)/len;
        string s = to_string(start);
        return s[(n-1)%len] - '0';
    }
};

class SolutionT92 {
public:
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        ListNode *cur = head, *end = nullptr;
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* pre = dummy;
        for (int i = 1; i < m; i++) {
            pre = cur;
            cur = cur->next;
        }
        ListNode* tail = cur;
        for (int i = m; i < n; i++) {
            ListNode* nextNode = tail->next;
            tail->next = nextNode->next;
            nextNode->next = pre->next;
            pre->next = nextNode;
        }
        return dummy->next;
    }
};

class Solution {
public:
    vector<string> generateParenthesis(int n) {
        int left = n, right = n;
        vector<string> res;
        string temp = "";
        helper(temp, left, right, res);
        return res;
    }

    void helper(string& temp, int left, int right, vector<string> &res) {
        if (left == 0 && right ==0) {
            res.push_back(temp);
            return;
        }
        if (left > right) return ;
        if (left == right) {
            helper(temp+'(', left-1, right, res);
        }
        else {
            helper(temp+'(',-1 right, res);
            helper(temp+')', left, right-1, res);
        }
        return ;
    }
};

class SolutionT55 {
public:
    bool canJump(vector<int>& nums) {
        int maxJump = 0;
        for (auto num : nums) {
            if (maxJump >= n - 1|| i > maxJump) break;
            maxJump = max(maxJump, num[i] + i);
        }
        return ;
    }
};

class SolutionT45 {
public:
    int jump(vector<int>& nums) {
        int res = 0, n = nums.size(), i = 0, cur = 0;
        while (cur < n - 1) {
            ++res;
            int pre = cur;
            for (; i <= pre; ++i) {
                cur = max(cur, i + nums[i]);
            }
            if (pre == cur) return -1; // May not need this
        }
        return res;
    }
};

//接雨水
class SolutionT42 {
public:
    int trap(vector<int>& height) {
        int res = 0, mx = 0, n = height.size();
        vector<int> dp(n, 0);
        for (int i = 0; i < n; i++) {
            dp[i] = mx;
            mx = max(mx, height[i]);
        }
        mx = 0;
        for (int i = n - 1; i >= 0; i--) {
            dp[i] = min(dp[i], mx)
            mx = max(mx, height[i]);
            if (dp[i] > height[i]) res+=dp[i] - height[i];
        }
        return res;
    }
};

//局部峰值，找到数组的形式规律
class SolutionT84 {
public:
    int largestRectangleArea(vector<int> &height) {
        int res = 0;
        for (int i = 0; i < height.size(); ++i) {
            if (i + 1 < height.size() && height[i] <= height[i + 1]) {
                continue;
            }
            int minH = height[i];
            for (int j = i; j >= 0; --j) {
                minH = min(minH, height[j]);
                int area = minH * (i - j + 1);
                res = max(res, area);
            }
        }
        return res;
    }
};


class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size(), n = obstacleGrid[0].size();
        vector<vector<int>> dp(m + 1, vector<int> (n+1, 0)); dp[0][1] = 1;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (obstacleGrid[i-1][j-1] == 1) dp[i][j] = 0;
                else {
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
                }
            }
        }
        return dp[m][n];
    }
};

