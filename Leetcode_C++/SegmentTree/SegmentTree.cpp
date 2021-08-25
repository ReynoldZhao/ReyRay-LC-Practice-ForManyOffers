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
#include<deque>
using namespace std;

//T307
class NumArray {
public:
    NumArray(vector<int> nums) {
        n = nums.size();
        tree.resize(n * 2);
        buildTree(nums);
    }

    void buildTree(vector<int>& nums) {
        for (int i = n; i < n * 2; ++i) {
            tree[i] = nums[i - n];
        }
        for (int i = n - 1; i > 0; --i) {
            tree[i] = tree[i * 2] + tree[i * 2 + 1];
        }
    }

    void update(int i, int val) {
        tree[i += n] = val;
        while (i > 0) {
            tree[i / 2] = tree[i] + tree[i ^ 1];
            i /= 2;
        }
    }

    int sumRange(int i, int j) {
        int sum = 0;
        for (i += n, j += n; i <= j; i /= 2, j /= 2) {
            if ((i & 1) == 1) sum += tree[i++];
            if ((j & 1) == 0) sum += tree[j--];
        }
        return sum;
    }    

private:
    int n;
    vector<int> tree;
};

struct SegNode {
    int lo, hi, add;
    SegNode *lchild, *rchild;
    SegNode(int left, int right): lo(left), hi(right), add(0), lchild(nullptr), rchild(nullptr) {}
};

class Solution {
public:
    SegNode* build(int left, int right) {
        SegNode *node = new SegNode(left, right);
        if (left == right) {
            return node;
        }
        int mid = (right - left) / 2 + left;
        node->lchild = build(left, mid);
        node->rchild = build(mid + 1, right);
        return node;
    }

    void insert(SegNode* root, int val) {
        root->add++;
        if (root->lo == root->hi) {
            return ;
        }
        int mid = (root->lo + root->hi) / 2;
        if (val <= mid) {
            insert(root->lchild, val);
        } else {
            insert(root->rchild, val);
        }
    }

    int count(SegNode* root, int left, int right) const {
        if (left > root->hi || right < root->lo) {
            return 0;
        }
        if (left <= root->lo && root->hi <= right) {
            return root->add;
        }
        return count(root->lchild, left, right) + count(root->rchild, left, right);
    }

    int countRangeSum(vector<int>& nums, int lower, int upper) {
        long long sum = 0;
        vector<long long> preSum = {0};
        for (int v : nums) {
            sum += v;
            preSum.push_back(sum);
        }

        set<long long> allNumbers;
        for (auto x : preSum) {
            allNumbers.insert(x);
            allNumbers.insert(x - lower);
            allNumbers.insert(x - upper);
        }
        unordered_map<long long, int> values;
        int idx = 0;
        for (auto x : allNumbers) {
            values[x] = idx;
            idx++;
        }

        SegNode* root = build(0, values.size() - 1);
        int ret = 0;
        for (long long x : preSum) {
            int left = values[x - upper], right = values[x - lower];
            ret += count(root, left, right);
            insert(root, values[x]);
        }
        return ret;
    }
};