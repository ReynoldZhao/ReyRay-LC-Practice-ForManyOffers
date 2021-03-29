    string largestNumber(vector<int>& nums) {
        string res;
        sort(nums.begin(), nums.end(), [](int a, int b) {
           return to_string(a) + to_string(b) > to_string(b) + to_string(a); 
        });
        for (int i = 0; i < nums.size(); ++i) {
            res += to_string(nums[i]);
        }
        return res[0] == '0' ? "0" : res;
    }

class SolutionT373 {
public:
    vector<pair<int, int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        vector<pair<int, int>> res;
        for (int i = 0; i < min((int)nums1.size(), k); ++i) {
            for (int j = 0; j < min((int)nums2.size(), k); ++j) {
                res.push_back({nums1[i], nums2[j]});
            }
        }
        sort(res.begin(), res.end(), [](pair<int, int> &a, pair<int, int> &b){return a.first + a.second < b.first + b.second;});
        if (res.size() > k) res.erase(res.begin() + k, res.end());
        return res;
    }
};

vector<vector<int>> dp;
int dfs(vector<vector<int>>& e, int i, int k) {
    if (k == 0 || i >= e.size())
        return 0;
    if (dp[i][k] != -1) 
        return dp[i][k];
    auto j = upper_bound(begin(e) + i, end(e), e[i][1], 
        [](int t, const vector<int> &v) {return v[0] > t;}) - begin(e);
    return dp[i][k] = max(e[i][2] + dfs(e, j, k - 1), dfs(e, i + 1, k));
}
int maxValue(vector<vector<int>>& events, int k) {
    dp = vector<vector<int>>(events.size(), vector<int>(k + 1, -1));
    sort(begin(events), end(events));
    return dfs(events, 0, k);
}