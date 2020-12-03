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
