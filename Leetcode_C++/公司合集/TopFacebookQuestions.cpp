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
#include<hash_map>
#include<deque>
using namespace std;

class SolutionT1249 {
public:
    string minRemoveToMakeValid(string s) {
        int left = 0, right = 0, target = 0, expect_left = 0, cur_right = 0;
        for (auto a : s) {
            if (a == '(') left++;
            else if (a == ')') right++;
        }
        target = min(left, right);
        stack<char> st;
        string res = "";
        for (int i = 0; i < s.size(); i++) {
            if (isalpha(s[i])) {
                res.push_back(s[i]);
            }
            if (s[i] == '(') {
                target -=1;
                if (target >= 0 && cur_right < right) {
                    expect_left++;
                    if (right - cur_right >= expect_left) {
                        st.push('(');
                        res.push_back(s[i]);
                    }
                }
            } else if (s[i] == ')') {
                cur_right++;
                if (!st.empty()) {
                    expect_left--;
                    st.pop();
                    res.push_back(s[i]);
                }
            }
        }
        return res;
    }
};

class SolutionT953 {
public:
    bool isAlienSorted(vector<string>& words, string order) {
        unordered_map<char, int> dict;
        for (int i = 0 ; i < order.size(); i++) {
            dict[order[i]] = i;
        }
        for (int j = 0 ; j < words.size() - 1; j++) {
            string word1 = words[j], word2 = words[j + 1];
            int size = min(word1.size(), word2.size()), k = 0, equal = 0;
            for (; k < size; k++) {
                if (dict[word1[k]] < dict[word2[k]]) break;
                if (dict[word1[k]] > dict[word2[k]]) return false;
                if (dict[word1[k]] == dict[word2[k]]) equal++;
            }
            if (equal == size && size < word1.size()) return false;
        }
        return true;
    }
};

class SolutionT680 {
public:
    bool validPalindrome(string s) {
        int start = 0, end = s.size() - 1;
        while (start < end) {
            if (s[start] == s[end]) {
                start++;
                end--;
            } else {
                return isPalindrome(s, start+1, end) || isPalindrome(s, start, end-1);
            }
        }
        return true;
    }

    bool isPalindrome(string s, int start, int end) {
        while (start < end) {
            if (s[start++] != s[end--]) return false;
        }
        return true;
    }
};

class SolutionT1762 {
public:
    vector<int> findBuildings(vector<int>& heights) {
        deque<int> inc;
        for (int i = 0; i < heights.size(); i++) {
            if (inc.empty()) inc.push_back(i);
            while(!inc.empty() && inc.back() != i && heights[i] >= heights[inc.back()]) {
                inc.pop_back();
            }
            inc.push_back(i);
        }
        vector<int> res(inc.begin(), inc.end());
        return res;
    }
};

class SparseVector {
 public:
  SparseVector(vector<int>& nums) {
    for (int i = 0; i < nums.size(); ++i)
      if (nums[i])
        v.push_back({i, nums[i]});
  }

  // Return the dotProduct of two sparse vectors
  int dotProduct(SparseVector& vec) {
    int ans = 0;

    for (int i = 0, j = 0; i < v.size() && j < vec.v.size();)
      if (v[i].first == vec.v[j].first)
        ans += v[i++].second * vec.v[j++].second;
      else if (v[i].first < vec.v[j].first)
        ++i;
      else
        ++j;

    return ans;
  }

 private:
  vector<pair<int, int>> v;  // {index, num}
};

class SolutionT42 {
public:
    int trap(vector<int>& height) {
        int left = 0, right = height.size() - 1, res = 0;
        while (left < right) {
            int mn = min(height[left], height[right]);
            if (mn == height[left]) {
                left++;
                while (left < right && height[left] < mn) {
                    res += mn - height[left];
                }
            } else {
                right--;
                while(left < right && height[right] < mn) {
                    res += mn - height[right];
                }
            }
        }
        return res;
    }

    int trap(vector<int>& height) {
        stack<int> st;
        int i = 0, res = 0;
        while (i < height.size()) {
            if (st.empty() || height[i] < height[st.top()]) {
                st.push(i++);
            } else {
                int t = st.top(); st.pop();
                if (st.empty()) continue;
                res += ((min(height[i], height[st.top()]) - height[t]) * (i - st.top() - 1));
            }
        }
        return res;
    }
};

class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        stack<int> st;
        int res = INT_MIN;
        for (int i = 0; i < heights.size(); i++) {
            if (st.empty() || heights[i] < heights[st.top()]) {
                st.push(i);
                res = max(res, heights[i] * 2);
            } else {
                while (heights[i] > heights[st.top()]) {
                    int temp = heights[st.top()] * (i - st.top() + 1);
                    st.pop();
                    res = max(res, temp);
                }
                st.push(i);
            }
        }
        return res;
    }
};

class SolutionT739 {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        vector<int> res;
        vector<int> temp;
        for (int i = temperatures.size() - 1; i >= 0; i--) {
            if (temp.empty() || temperatures[i] < temperatures[temp.back()]) {
                if (!temp.empty()) res.push_back(temp.back() - i);
                else res.push_back(0);
                temp.push_back(i);
            } else {
                while (!temp.empty() && temperatures[i] >= temperatures[temp.back()]) {
                    temp.pop_back();
                }
                if (temp.empty()) res.push_back(0);
                else res.push_back(temp.back() - i);
                temp.push_back(i);
            }
        }
        reverse(res.begin(), res.end());
        return res;
    }

    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        vector<int> res{n, 0};
        stack<int> st;
        for (int i = 0; i < temperatures.size(); i++) {
            while (!st.empty() && temperatures[i] > temperatures[st.top()]) {
                auto t = st.top(); st.pop();
                res[t] = i - t;
            }
            st.push(i);
        }
        return res;
    }
};

class SolutionT907 {
public:
    int sumSubarrayMins(vector<int>& arr) {
        long res = 0;
        int m = 1e9 + 7;
        for (int i = 0; i < arr.size(); i++) {
            // vector<int> st;
            int min_val = INT_MAX;
            for (int j = i; j < arr.size(); j++) {
                min_val = min(min_val, arr[j]);
                res += min_val;
            }
        }
        return res % m;
    }

    int sumSubarrayMins(vector<int>& arr) {
        int n = arr.size(), res = 0, M = 1e9+7;
        stack<int> st{{-1}};
        vector<int> dp(n+1); 
        //dp[i] 表示以数字 arr[i-1] 结尾的所有子数组最小值之和
        // 所以dp[i] 是 i - 1个子数组，他们的最小值，的和
        for (int i = 0; i < arr.size(); i++) {
            while (st.top() != -1 && arr[i] <= st.top()) st.pop();
            //经过这一步操作，栈顶st.top()一定是第一个比当前元素小的元素
            // dp[st.top() + 1] 是以st.top结尾的所有子数组的最小元素和
            // (i - st.top()) * arr[i]
            //是从 st.top() --- i, 这一段，以arr[i]结尾的子数组的最小值的和
            dp[i + 1] = (dp[st.top() + 1] + (i - st.top()) * arr[i]) % M;
            st.push(i);
            res = (res + dp[i + 1]) % M;
        }
        return res;
    }
};

class SolutionT503 {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        int n = nums.size();
        vector<int> temp(2 * n, 0);
        vector<int> res(n, -1);
        for (int i = 0; i < 2 * n; i++) {
            temp[i] = nums[i % n];
        }
        stack<int> st;
        for (int j = 0; j < 2 * n; j++) {
            while (!st.empty() && temp[j] > temp[st.top()]) {
                int cur_pos = st.top(); st.pop();
                if (res[cur_pos % n] != -1) {
                    res[cur_pos % n] = temp[j];
                }
            }
            st.push(j);
        }
        return res;
    }
};

class StockSpanner {
public:
    StockSpanner() {
        
    }
    
    int next(int price) {
        int cur_span = 0;
        while(!st.empty() && price >= st.top().first) {
            int pre_span = st.top().second; st.pop();
            cur_span += pre_span;
        }
        st.push(make_pair(price, cur_span));
        return cur_span;
    }
private:
    stack<pair<int, int>> st;
};

class SolutionT321 {
public:
    vector<int> maxNumber(vector<int>& nums1, vector<int>& nums2, int k) {
        
    }
};

class BinaryMatrix {
  public:
    int get(int row, int col);
    vector<int> dimensions();
};

class SolutionT1428 {
public:
    int leftMostColumnWithOne(BinaryMatrix &binaryMatrix) {
        int n = binaryMatrix.dimensions()[0];
        int m = binaryMatrix.dimensions()[1];

        int check = m - 1;
        for (int i = 0; i < n; i++) {
            while (check >= 0 && binaryMatrix.get(i, check) == 1) {
                check--;
            }
        }
        return check == m - 1 ? -1 : check + 1;
    }
};

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

class Solution {
public:
    Node* treeToDoublyList(Node* root) {
        Node* head = new Node(0);
        stack<Node*> st;
        Node *cur = root, *pre = nullptr;
        while (cur || !st.empty()) {
            while(cur) {
                st.push(cur);
                cur = cur->left;
            }
            cur = st.top(); st.pop();
            if (!pre) {
                head = cur;
                pre = cur;
            } else {
                pre->right = cur;
                cur->left = pre;
                pre = pre->right;
            }
            cur = cur->right;
        }
        pre->right = head;
        head->left = pre;
        return head;
    }
};