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
using namespace std;

class Solution {
public:
	void replaceSpace(char *str,int length) {
        int blankcount;
        int i = 0;
        int pre_len = 0, cur_len = 0;
        while (str[i] != '\0'){
            pre_len++;
            if (str[i] == ' ') {
                blankcount++;
                cur_len = pre_len + 2;
        }
        if (cur_len > length) return;
        int former_len = pre_len, new_len = cur_len;
        while(former_len >= 0 && new_len >= pre_len){
            if (str[former_len] == ' '){
                str[new_len--] = '0';
                str[new_len--] = '2';
                str[new_len--] = '%';
                former_len--;
            }
            else str[new_len--] = str[former_len];
        }
	}

struct ListNode {
      int val;
      struct ListNode *next;
      ListNode(int x) :
            val(x), next(NULL) {
      }
};

public:
    vector<int> printListFromTailToHead(ListNode* head) {
        ListNode dummy = ListNode(0);
        ListNode cur = dummy;
        while(head){
            cur = head->next;
            head->next = dummy->next;
            dummy->next = head;
            head = cur;
        }
    }
};



struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
 
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        if (pre.size()<= 1) return TreeNode* t = new TreeNode(pre[0]);
        int root = pre[0];
        int i = 0;
        while (i < pre.size() && pre[i] != root) i++;
        int left_size = i + 1, right_size = pre.size() - 1 - left_size;
        TreeNode* head = new TreeNode(root);
        vector<int> left_pre;
        left_pre.assign(pre.begin() + 1), pre.begin() + 1 + left_size())
        vector<int> left_vin = vin.assign(vin.begin(), vin.begin()+left_size) 
        head->left = reConstructBinaryTree(left_pre)
    }

public:
    void push(int node) {
        stack1.push(node);
    }

    int pop() {
        int temp;
        while(!stack1.empty()){
            temp = stack1.top();
            stack2.push(temp);
            stack1.pop();
        }
        int res =stack2.top();
        stack2.pop();
        while(!stack2.empty()){
            temp = stack2.top();
            stack1.push(temp);
            stack2.pop();
        }
    }

    int minNumberInRotateArray(vector<int> rotateArray) {
        int left = 0, right = rotateArray.size();
        if (rotateArray[left] <= rotateArray[right]) return rotateArray[left];
        int mid = left + (right - left) / 2;
        while (left < right){
            if (rotateArray[mid] > rotateArray[right]) {
                left = mid + 1;
            }
            else if (rotateArray[mid] < rotateArray[right]) {
                right = mid ;
            }
            else return rotateArray[right];
        }
    }
    
    int Fibonacci(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1
        return Fibonacci(n - 1) + Fibonacci(n - 2);

        int f = 0, g = 1;
        while (n--) {
            g = g + f;
            f = g - f
        }
    }
    
    int jumpFloor(int number) {
        if (number <= 0) return 0;
        if (number == 1) return 1;
        if (number == 2) return 2;
        return jumpFloor(number - 1) + jumpFloor(number - 2);
    }

    int jumpFloorII(int number) {

    }

    void reOrderArray(vector<int> &array) {
        int i = array.size();
        for (int i = 0; i < array.size(); i++) {
            for (int j = i + 1; j < array.size(); j++) {
                if (array[j]%2 == 1 && array[j - 1]%2 == 0) swap(array[j], array[j-1]);
            }
        }
    }
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        if (pRoot1 == NULL || pRoot2 == NULL) return false;
        return HasSubtree(pRoot1->left, pRoot2) || HasSubtree(pRoot1->right, pRoot2) || IsSubtree(pRoot1, pRoot2);
    }
    bool IsSubtree(TreeNode* n1, TreeNode* n2) {
        if (n2 == NULL) return true;
        if (n1 == NULL) return false;

    }
    void Mirror(TreeNode *pRoot) {
        if (!pRoot) return;
        TreeNode* templ = pRoot->left;
        TreeNode* tempr = pRoot->right;
        pRoot->left = tempr;
        pRoot->right = templ;
        Mirror(pRoot->left);
        Mirror(pRoot->right);
        return;
    }
    vector<int> printMatrix(vector<vector<int> > matrix) {
        int height = matrix.size();
        int width = matrix[0].size();
        int start_row = 0, start_col = 0;
        while (start <= 2 && )
        for (int i = 0;i < width - 1; i++) printf(matrix[start_row][start_col + i]);
        start_col = width - 1;
        for (int j = 0;j < height - 1; j++) printf(matrix[start_row + j][start_col];
        start_row = height - 1;
        for (int k = 0;k < width - 1; k++) printf

        width -=2;
        hei-=2;
        )
    }

    stack<int> st;
    stack<int> min;
    void push(int value) {
        st.push(value);
        if (min.empty()) min.push(value);
        else {
            int temp = min.top();
            if (value <= temp) min.push(value);
        }
    }
    void pop() {
        int temp = st.top();
        st.pop();
        if(temp = min.top()) min.pop();
    }
    int top() {
        int top = st.top();
        return top;
    }
    int min() {
        int m = min.top();
        return m
    }
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        stack<int> help;
        for (int i = 0, j = 0; i < pushV.size(); i++) {
            help.push(pushV[i]);
            while (help.back() == popV[j]) {
                help.pop();
                j++;
            }
        }
    }
    bool VerifySquenceOfBST(vector<int> sequence) {
        if (sequence.empty()) return true;
        int l = 0, r = sequence.size() - 1;
        return helpsequence(sequence, l, r);
    }

    bool helpsequence(vector<int> sequence, int left, int right) {
        if (left >= right) return true;
        int i;
        for (i = 0; i < right; i++) {
            if (sequence[i] > sequence[right]) break;
        }
        for (int j = i; j < right; j++) {
            if (sequence[j] < sequence[right]) return false;
        }
        return helpsequence(sequence, left, i-1) && helpsequence(sequence, i, right - 1);
    }
    
    vector<vector<int>> res;
    vector<int> temp;

    vector<vector<int> > FindPath(TreeNode* root,int expectNumber) {       
        if (!root) return res;
        temp.push_back(root->val);
        expectNumber = expectNumber - root->val;
        if (expectNumber == 0 && !root->left && root->left)
            res.push_back(temp);
        
    }

    void help(vector<vector<int>> &res, vector<int> &temp, TreeNode* root, int expectNumber) {
        if(root->left == NULL && root->right == NULL) {
            if (root->val = expectNumber) {
                res.push_back(temp);
                return;
            }
            else return;
        }
        temp.push_back(root->val);
        expectNumber = expectNumber - root->val;
        if (root->left) help(res, temp, root->left, expectNumber);
        if (root->right) help(res, temp, root->right, expectNumber);
        temp.pop_back();
    }

    struct RandomListNode {
        int label;
        struct RandomListNode *next, *random;
        RandomListNode(int x) :
            label(x), next(NULL), random(NULL) {
        }
    };   

    RandomListNode* Clone(RandomListNode* pHead)
    {
        if (pHead == NULL) return NULL;
        RandomListNode* current_node = pHead;
        while (current_node) {
            RandomListNode* temp = new RandomListNode(current_node->label);
            temp->next = current_node->next;
            current_node->next = temp;
            current_node = temp->next;
        }
        RandomListNode* c = pHead;
        while (c) {
            c->next->random = c->random->next;
            c = c->next;
        }
        currentNode = pHead;
        RandomListNode pCloneHead = pHead.next;
        while(currentNode != null) {
            RandomListNode cloneNode = currentNode.next;
            currentNode.next = cloneNode.next;
            cloneNode.next = cloneNode.next==null?null:cloneNode.next.next;
            currentNode = currentNode.next;
        }
    }

    static bool cmp(int a, int b) {
        string A, B;
        A+=to_string(a);
        A+=to_string(b);
        B+=to_string(a);
        B+=to_string(b);
        return A<B;
    }
    string PrintMinNumber(vector<int> numbers) {
        string res = "";
        sort(numbers.begin(), numbers.end(), cmp);
        
    }

    int GetUglyNumber_Solution(int index) {
        if (index < 7) return index;
        int p2 = 0, p3 = 0, p5 = 0, newnum = 1;
        vector<int> v(index);
        v.push_back(newnum);
        for (int i = 1;i<index;i++){
            newnum = min(v[p2]*2, v[p3]*3, v[p5]*5);
            v[i] = newnum;
            if (newnum==v[p2]*2) p2++;
            if (newnum==v[p3]*3) p3++;
            if (newnum==v[p5]*5) p5++;
        }
        return v.back();
    }

    int FirstNotRepeatingChar(string str) {
        vector<int> c(256,0);
        hash_map<char, int> map;
        for (int i=0;i<str.size();i++){
            c[str[i]] += 1;
            if (c[str[i]] == 1) {
                map.insert(pair(c[str[i]], i));
            }
            else if (c[str[i]]!=1 && map.find(c[str[i]])) {
                iter = map.find(c[str[i]]);
                map.erase(iter);
            }
        }
    }

    int InversePairs(vector<int> data) {
        int length=data.size();
        if(length<=0)
            return 0;
        vector<int> copy;
        for (int i=0;i<data.size();i++) {
            copy.push_back(data[i]);
        }
        long long int count = merge(copy, data, 0, data.size()-1);
        return count;
    }

    long long int merge(vector<int> copy, vector<int> data, int start, int end) {
        if (start == end) {
            copy[start] = data[start];
            return 0;
        }
        int length = (end - start + 1)/2;
        long long int left = merge(copy, data, start, start + length);
        long long int right = merge(copy,data, start + length +1, end);
        int i = start + length, j = end;
        int copyindex = end;
        long long int count = 0;
        for(;i>=start;i--) {
            if (data[i]>data[j]) {
                copy[copyindex--] = data[i];
                count = count + j - start - length;
            }
            else copy[j--] = data[j--];
        }
       for(;i>=start;i--)
           copy[indexcopy--]=data[i];
       for(;j>=start+length+1;j--)
           copy[indexcopy--]=data[j];       
       return left+right+count;

    }
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        if (pHead1==NULL || pHead2==NULL) return NULL;   
        ListNode* p1 = pHead1;
        ListNode* p2 = pHead2;
        while (p1!=p2) {
            p1 = (p1==NULL) ? pHead2:p1->next;
            p2 = (p2==NULL) ? pHead1:p2->next;
        } 
    }
    
    int GetNumberOfK(vector<int> data ,int k) {
        int start = 0, end = data.size()-1;
        int mid = (start + end)/2;
        
    }
    int findpos(vector<int> data, float k) {
        int start = 0, end = data.size()-1;
        int mid = (start + end)/2;
        while (start < end) {
            if (data[mid]<k) start = mid+1;
            else end = mid - 1;
            mid = (start + end)/2;
        }
        return start;
    }
    void swap(int a, int b) {
        a = a^b;
        b = a^b;
        a = a^b;
    }
    void Reverse(string &str, int start, int end) {
        while (s>e) {
            swap(str[start],str[end]);
        }
    }
    string ReverseSentence(string str) {
        Reverse(str, 0, str.size()-1);
        int i=0;
        while (i<str.size()) {
            while(i<str.size()&&str[i]==' ') i++;
            int s = i, e = i;
            Reverse(str, s, e-1);
            while (str[e]!=' ') {
                e++;
                i++;
            }
            Reverse(str, s, e-1);
        }
    }

    int LastRemaining_Solution(int n, int m)
    {   
        if(n==0||m==0) return -1;
        if(n==1) return 1;
        vector<int> v;
        for(int i=0;i<n;i++) v.push_back(i);
        while (v.size()>1&&n>1) {
            int count = (m-1)%n;
            v.erase(v.begin()+count);
            n--;
        }
        return v.back();
    }

    int LastRemaining_Solution(int n, int m){
        if (n==0||m==0) return -1;
        vector<int> v(n+1,0);
        while (n>1) {
            int count = m%n;
            int i;
            for (i=1; i<=n; i++){
                if (v[i]==0) {
                    count--;
                    if (count==0) break;
                }
                else continue;
            }
            v[i]=-1;
            n--;
        }
        for (int i=1;i<=n;i++) {
            if (v[i]==0)
                return i;
        }
    }
    int StrToInt(string str) {
        long long res = 0;
        int mark = 1;
        if (str[0]=='-') mark = -1;
        for (int i=(str[0]=='-'||str[0]=='+')?1:0;i<str.size();i++) {
            if (str[i]>='0'&&str[i]<='9') {
                res = res*10 + (str[i]-'0');
                if(((mark>0) && (num > 0x7FFFFFFF)) ||((mark<0) && (num > 0x80000000)))
                    
            }
            else return 0;
        }
        return mark*res;
    }
    bool isNumeric(char* string)
    {
        bool hasE = false, sign = false, decimal = false;
        for (int i=0;i<strlen(string);i++) {
            if(string[i]=='e'||string[i]=='E'){
                if(hasE) return false;
                if(i==strlen(str)-1) return false;
                hasE = true;
            }
            else if(string[i]=='+'||string[i]=='-'){
                if(sign&&string[i-1]!='e'&&string[i-1]!='E') return false;
                if(!sign&&i>0&&string[i-1]!='e'&&string[i-1]!='E') return false;
                sign = true;
            }
            else if(string[i]=='.'){
                if(decimal||hasE) return false;
                decimal = true;
            }
            else if (str[i] < '0' || str[i] > '9') // 不合法字符
                return false;
        }
    }

    void Insert(char ch)
    {
        count[ch-'0']++;
        if (count[ch-'0']==1) {
            q.push(ch);
        }
    }
  //return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
        while (!q.empty()) {
            if (count[q.front()]==1) {
                return q.front();
            }
            else q.pop();
        }
        if (q.empty) return '#';
    }
    TreeLinkNode* GetNext(TreeLinkNode* pNode)
    {
        if (pNode==null) return null;
        if(pNode->right!=null) {
            TreeLinkNode* temp = pNode->right;
            while(temp->left) {
                temp = temp->left;
            }
            return temp;
        }
        TreeLinkNode* root = pNode->father;
        TreeLinkNode* temp = pNode;
        while(temp != root->left)
            temp = root；
            root = root->father;
    }

    bool isSymmetrical(TreeNode* pRoot)
    {
        queue<TreeNode*>
    }

    bool isSymmetrical(TreeNode* pRoot)
    {
        if(pRoot==NULL) return true;
        if(pRoot->left&&pRoot->right){
            return help(pRoot->left,pRoot->right);
        }
        else if(!pRoot->left&&!pRoot->right) {
            return true;
        }
        else return false;
    }
    bool help(TreeNode* left, TreeNode* right) {
        if(left == null) return right==null;
        if(right == null) return false;
        if(left->val!=right->val) return false;
        else {
            return help(left->left, right->right)&&help(left->right, right->left);
        }
    }
    public ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {

    }
    char* Serialize(TreeNode *root) {    
        string str;
        if(!root) return NULL;
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()){
            int n = q.size();
            for(int i=0;i<n;i++){
                TreeNode* temp = q.front();
                if(temp!=NULL){
                    q.push(temp->left);
                    q.push(temp->right);
                    str+=to_string(temp->val)+',';
                    q.pop();
                }
                else {
                    str+="#,";
                }
            }
        }
        char* chr = strdup(str.c_str());
        return chr
    }
    TreeNode* Deserialize(char *str) {
        if(!str) return nullptr
        int k = 0;
        auto res = nextNode(str, k);
        deque<TreeNode*> q;
        q.push_back(res);
        while(!q.empty()){
            int n = q.size();
            for(int i=0;i<n;i++){
                q.front()->left = nextNode(str, k);
                q.front()->right = nextNode(str, k);
                if (q.front()->left)
                    q.push_back(q.front()->left);
                if (q.front()->right)
                    q.push_back(q.front()->right);
                q.pop_front();               
            }
        }
    }
    TreeNode* nextNode(char* str, int &k){
        string s;
        while(str[i]!='\0'&&str[i]!=','){
            if(str[i]=='#'){
                k +=2;
                return nullptr;
            }
            s+=s[i];
            i++;
        }
        if(str[i]==',')
        i++;
        if(!s.empty())
            return new TreeNode(stoi(s));
        return nullptr;

    }

    TreeNode* KthNode(TreeNode* pRoot, int k)
    {
        if(!pRoot) return NULL;
        stack<TreeNode*> s;
        int index = 0;
        TreeNode* temp = pRoot
        while(!s.empty()&&temp!=NULL){
            if(temp!=NULL){
                s.push(temp);
                temp = temp->left;
            }
            else{
                temp = s.pop();
                index++;
                if(index==k) return temp;
                temp = temp->right;
                
            }
        }
    }
    int index = 0;
    TreeNode* KthNode(TreeNode* pRoot, int k)
    {
        if(!pRoot) return NULL;
        TreeNode* temp = KthNode(pRoot->left,k);
        if(temp!=NULL) return temp;
        index++;
        if(index==k){
            return pRoot;
        }



    }
    vector<int> maxInWindows(const vector<int>& num, unsigned int size)
    {
        deque<int> q;
        vector<int> res;
        for(int i=0;i<num.size();i++){
            while(q.size()&&num[s.back()]<num[i]) q.pop_back();
            while(q.size()&&i-s.front()+1>size) q.pop_front();
            q.push_back(i);
            if(size&&i+1>=size){
                res.push_back(q.front());
            }
        }
    public bool mark = false;
    bool hasPath(char* matrix, int rows, int cols, char* str)
    {
        char visited[rows][cols];
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                if(matrix[i][j]==str[0]){
                    dfs()
                }
            }
        }
    }
    void dfs(char* matrix, int rows, int cols, char* str, int i,int j,int k, char& visited[][]){
        if(mark) return;
        if(matrix[i][j]==str[k]) {
            if(k==strlen(str)){
                mark = true;
                return;
            }
                if(i-1>=0&&visit[i-1][j]!=1){
                    visited[i-1][j]=1;
                    dfs(matrix, rows, cols, str, i-1, j, k+1, visited);
                    visited[i-1][j]=0;
                }
            }
        }
    }
    int movingCount(int threshold, int rows, int cols)
    {
        
    }
    int getsum(int n){
        int sum = 0;
        while(n>0){
            int temp = n%10;
            n = n/10;
        }
        return sum;
    }
    int cutRope(int number) {
        
    }
    }
private:
    queue<char> q;
    unsigned count[128];
    stack<int> stack1;
    stack<int> stack2;
    

};

class SolutionT1496 {
public:
    bool isPathCrossing(string path) {
        set<pair<int,int>> visited;
        visited.insert({0,0});
        int x = 0, y = 0;
        for(char p: path){
            if (p == 'N') y++;
            else if(p == 'S')y--;
            else if(p == 'E') x++;
            else x--;
            if (visited.find({x,y})) return true;
            else visited.insert({x, y});
        }
        return false;
    }
};

class SolutionT1497 {
public:
    bool canArrange(vector<int>& arr, int k) {
        if (arr.length()%2 != 0) return false;
    }
};

class SolutionT1502 {
public:
    bool canMakeArithmeticProgression(vector<int>& arr) {
        if (arr.size() <= 2) return true;
        sort(arr.begin(), arr.end());
        int interval = arr[1] - arr[0];
        for (int i = 2; i < arr.size(); i++) {
            if(arr[i] - ar[i-1] != interval) return false;
        }
        return true;
    }
};

class SolutionT1503 {
public:
    int getLastMoment(int n, vector<int>& left, vector<int>& right) {
        sort(left.begin(), left.end());
        sort(right.begin(), right.end());
        return max(left.empty() ? 0 : left[left.size() - 1], right.empty() ? 0 : n - right[0]);
    }
};

class SolutionT1504 {
public:
    int numSubmat(vector<vector<int>>& mat) {
        
    }
};

class SolutionT1027 {
public:
    int longestArithSeqLength(vector<int>& A) {
        vector<pair<int,int>> dp(A.size()); //pair<len, dif>
        if (A.size() <= 2) return A.size();
        int N = A.size();
        vector<vector<int> > dp(N, vector<int>(20010, 1));
        int res = 1;
        for (int i = 1; i < N; i++) {
            for (int j = 0; j < i; j++) {
                int dif = A[i] - A[j];
                dif += 10000;
                dp[i][dif] = max(dp[i][dif], dp[j][dif] + 1);
                res = max(res, dp[i][dif]);
            }
        }
        return res;
    }
};

class SolutionT862 {
public:
    int shortestSubarray(vector<int>& A, int K) {
        int n = A.size(), res = INT_MAX, sum = 0;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int> > pq;
        for (int i = 0; i < A.size(); i++) {
            sum+=A[i];
            if (!pq.empty() && sum - pq.top().first < K) {
                pq.push({sum, i}); 
                continue;
            }
            while(!pq.empty() && sum - pq.top().first >= K) {
                res = min(res, i - pq.top().second);
                pq.pop(); //可以分下一个比自己大 下一个比自己小来揣测这个pop
            }
            pq.push({sum, i}); 
        }
        return res == INT_MAX? -1 : res;
    }

    int shortestSubarray(vector<int>& A, int K) {
        int n = A.size(), res = INT_MAX;
        map<int, int> sumMap;
        vector<int> sums(n + 1)
        for (int i = 1; i <= n; i++) sum[i] = sum[i-1] + A[i-1];
        for (int i = 0; i <= n; i++) {
            auto pos = sumMap.upper_bound(sum[i] - K);
            if (pos != sumMap.end()) {
                for (auto it = sumMap.begin(); it != pos; it++) {
                    res = min(res, i - it->second);
                }
                sumMap.erase(sumMap.begin(), pos);
            }
            sumMap[sum[i]] = i;
        }
        return res == INT_MAX? -1 : res;
    }

    int shortestSubarray(vector<int>& A, int K) {
        int n = A.size(), res = INT_MAX;
        deque<int> dq;
        vector<int> sums(n + 1);
        for (int i = 1; i <= n; ++i) sums[i] = sums[i - 1] + A[i - 1];
        for (int i = 0; i <= n; i++) {
            while(!dq.empty() && sum[i] - sum[dq.front()] >= K) {
                res = min(res, i - dq.front());
                dq.pop_front();
            }
            while(!dq.empty() && sum[i] <= sum[dq.back()]) dq.pop_back();
            dq.push_back(i);
        }
        return res == INT_MAX? -1 : res;
    }
};

class SolutionT128 {
public:
    //使用集合
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> set(nums.begin(), nums.end());
        int res = 0;
        for (auto num:nums) {
            if (!set.count(num)) continue;
            set.erase(num);
            int pre = num-1, next = num+1;
            while(set.count(pre)) s.erase(pre--);
            while(set.count(next)) s.erase(next++);
            res = max(res, next - pre - 1);
        }
        return res;
    }

    //dp
    int longestConsecutive(vector<int>& nums) {
        int res = 0;
        unordered_map<int, int> m;
        for (int num : nums) {
            if (m.count(num)) continue;
            int left = m.count(num - 1) ? m[num - 1] : 0;
            int right = m.count(num + 1) ? m[num + 1] : 0;
            int sum = left + right + 1;
            m[num] = sum;
            res = max(res, sum);
            m[num - left] = sum;
            m[num + right] = sum;
        }
        return res;
    }

    // public int longestConsecutive(int[] nums) {
    //     if (nums.length == 0) return 0;

    //     int n = nums.length, max = 1;
    //     Set<Integer> set = new HashSet<>();
    //     for (int v : nums) set.add(v);

    //     for (int v : nums) {
    //         // 技巧：如果比自己小的已经在了，自己便不用查了，剪枝
    //         if (set.contains(v - 1)) continue;

    //         int r = v; // r: right 表示「以 v 开头，能连续到多少」
    //         while (set.contains(r + 1)) r++; // 逐个查看
    //         max = Math.max(max, r - v + 1); // 记录区间 [v, r] 长度
    //     }
    //     return max;
    // }

    //并查集
    // public int longestConsecutive(int[] nums) {
    //     if (nums.length == 0) return 0;
        
    //     // 首次遍历，与邻居结盟
    //     UnionFind uf = new UnionFind(nums);
    //     for (int v : nums)
    //         uf.union(v, v + 1); // uf.union() 结盟

    //     // 二次遍历，记录领队距离
    //     int max = 1;
    //     for (int v : nums)
    //         max = Math.max(max, uf.find(v) - v + 1); // uf.find() 查找领队
    //     return max;
    // }

    // class UnionFind {
    //     private int count;
    //     private Map<Integer, Integer> parent; // (curr, leader)

    //     UnionFind(int[] arr) {
    //         count = arr.length;
    //         parent = new HashMap<>();
    //         for (int v : arr)
    //             parent.put(v, v); // 初始时，各自为战，自己是自己的领队
    //     }

    //     // 结盟
    //     void union(int p, int q) {
    //         // 不只是 p 与 q 结盟，而是整个 p 所在队伍 与 q 所在队伍结盟
    //         // 结盟需各领队出面，而不是小弟出面
    //         Integer rootP = find(p), rootQ = find(q);
    //         if (rootP == rootQ) return;
    //         if (rootP == null || rootQ == null) return;

    //         // 结盟
    //         parent.put(rootP, rootQ); // 谁大听谁
    //         // 应取 max，而本题已明确 p < q 才可这么写
    //         // 当前写法有损封装性，算法题可不纠结

    //         count--;
    //     }

    //     // 查找领队
    //     Integer find(int p) {
    //         if (!parent.containsKey(p))
    //             return null;

    //         // 递归向上找领队
    //         int root = p;
    //         while (root != parent.get(root))
    //             root = parent.get(root);

    //         // 路径压缩：扁平化管理，避免日后找领队层级过深
    //         while (p != parent.get(p)) {
    //             int curr = p;
    //             p = parent.get(p);
    //             parent.put(curr, root);
    //         }

    //         return root;
    //     }
    // }

};

//双指针
class SolutionT11 {
public:
    int maxArea(vector<int>& height) {
        for (int i = 0; i < height.size(); i++) {

        }
    }
};


class Solution T41{
public:
    int firstMissingPositive(vector<int>& nums) {

    }
};

class SolutionT72 {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size(), n = word2.size();
        vector<vector<int>> memo(m, vector<int>(n));
        return helper(word1, 0, word2, 0, memo);
    }
    int helper(string& word1, int i, string& word2, int j, vector<vector<int>>& memo) {
        if (i == word1.size()) return (int)word2.size() - j;
        if (j == word2.size()) return (int)word1.size() - i;
        if (memo[i][j] > 0) return memo[i][j];
        int res = 0;
        if (word1[i] == word2[j]) {
            return helper(word1, i + 1, word2, j + 1, memo);
        } else {
            int insertCnt = helper(word1, i, word2, j + 1, memo);
            int deleteCnt = helper(word1, i + 1, word2, j, memo);
            int replaceCnt = helper(word1, i + 1, word2, j + 1, memo);
            res = min(insertCnt, min(deleteCnt, replaceCnt)) + 1;
        }
        return memo[i][j] = res;
    }


    //DP dp[i][j] word1 i位置 word2 j位置 最小替换
    int minDistance(string word1, string word2) {
        int m = word1.size(), n = word2.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1));
        for (int i = 0; i <= m; ++i) dp[i][0] = i;
        for (int i = 0; i <= n; ++i) dp[0][i] = i; //初始化
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (word1[i - 1] == word2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1;
                    // dp[i][j] = min(替换，min(删除，插入))；
                }
            }
        }
        return dp[m][n];
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

//二分 DP 真的很屌
class SolutionT410 {
public:
    int splitArray(vector<int>& nums, int m) {

    }
};

class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.size(), n = text2.size();
        string lo = m > n?text1:text2;
        string sh = m > n?text2:text1;
        int i = 0, j = 0;
        int lar = m > n?m:n, sma = m > n?n:m;
        int res = 0;
        while (i < lar && j < sma) {
            if (text1[i] == text2[j]) {
                res++; i++; j++;
            }
            else i++;
        }
        return res;
    }
};

//excellent idea
class SolutionT123 {
public:
    int maxProfit(vector<int>& prices) {
        int cost1 = INT_MAX, cost2 = INT_MAX, profit1 = 0, profit2 = 0;

        for (int i = 0; i < prices.size(); i++) {
            cost1 = min(cost1, prices[i]);
            pro1 = max(pro1, prices[i] - cost1);
            cost2 = min(cost2, prices[i] - pro1); //把前一次的利润算进成本
            pro2 = max(pro2, prices[i] - cost2);
        }
        return pro2;
    }
};

class SolutionT23 {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        auto cmp = [](ListNode* A, ListNode* B) {
            return A->val > B->val;
        }
        priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);
        for (auto node:lists) {
            if (node) pq.push(node);
        }
        ListNode* dummy = new ListNode(-1), *cur = dummy;
        while (!pq.empty())
        {
            ListNode* temp = pq.top(); pq.pop();
            cur->next = temp;
            cur = cur->next;
            if (temp->next) pq.push(temp->next);
        }
        return dummy->next;
    }
};

class SolutionT209 {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int l = 0, r = 0, sum = 0, len = nums.size();
        int res = INT_MAX;
        while (r < len) {
            while(sum < s && r < len) {
                sum += nums[r++];
            }
            while(sum >= s) {
                res = min(res, r - l);
                sum -= nums[l++];
            }
        }
        return res == INT_MAX?0:res;
    }
};

class Solution {
public:
    string longestPalindrome(string s) {
        int len = s.size();
        vector<vector<int>> dp(len, vector<int>(len, 0));
        int res = 0, start = 0;
        for (int i = 0; i < len; i++) {
            dp[i][i] = 1;
            for (int j = i+1; j <len; j++) {
                if (s[i] != s[j]) continue;
                dp[i][j] = j - i <= 2?1:dp[i+1][j-1];
                if (dp[i][j] && j - i + 1 > res) {
                    res = j - i + 1;
                    start = i;
                }
            }
        }
        return s.substr(start, res);
    }
};

class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> res;
        int len = intervals.size(), i = 0;
        while (i < len) {
            vector<int> temp;
            temp.push_back(intervals[i][0]);
            int temp_end = intervals[i][1], j = i+1;
            while (j < len) {
                if (intervals[j][1] <= temp_end) {
                    j++;
                }
                else if (intervals[j][0] <= temp_end && intervals[j][1] > temp_end) {
                    temp_end = intervals[j][1];
                    j++;
                } else break;
            }
            temp.push_back(temp_end);
            i = j;
            res.push_back(temp);
        }
        return res;
    }

    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if (intervals.empty()) return {};
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> res{intervals[0]};
        for (int i = 1; i < intervals.size(); ++i) {
            if (res.back()[1] < intervals[i][0]) {
                res.push_back(intervals[i]);
            } else {
                res.back()[1] = max(res.back()[1], intervals[i][1]);
            }
        }   
        return res;
    }
};

class SolutionT98 {
public:
    bool isValidBST(TreeNode* root) {
        if (!root) return true;
        stack<TreeNode*> st;
        vector<int> v;
        TreeNode* p = root;
        while(!st.empty() || p) {
            while(p) {
                st.push(p);
                p = p->left;
            }
            p = st.front();
            v.push_back(p->val);
            p = p->right;
        }
        for (int i = 0; i < v.size()-1; i++) {
            if (v[i] > v[i+1]) return false;
        }
        return true;
    }

    bool isValidBST(TreeNode* root) {
        return helper(root, INT_MIN, INT_MAX);
    }

    bool helper(TreeNode* root, int min, int max) {
        if(!root) return true;
        if (root->val > min && root->val < max) {
            return helper(root->left, min, root->val) && helper(root->right, root->val, max);
        } else {
            return false;
        }
    }    
};

class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for (int i = 0; i < n/2; i++) {
            for (int j = i; j <= n - i - 2; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - j][i];
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j];
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i];
                matrix[j][n - 1 - i] = tmp;
            }
        }
    }
};

class SolutionT114 {
public:
    void flatten(TreeNode* root) {
        if (!root) return ;
        TreeNode* temp = helper(root);
        return ;
    }

    TreeNode* helper(TreeNode* root) {
        if (!root) return nullptr;
        TreeNode* leftNode = helper(root->left);
        TreeNode* rightNode = helper(root->right);
        TreeNode* cur = leftNode;
        while(cur && cur->right) cur = cur->right;
        if(cur) {
            cur->right = rightNode;
            root->right = leftNode;
            root->left = nullptr; 
        } else {
            root->right = rightNode;
        }
        return root;
    }

    void flatten(TreeNode* root) {
        if (!root) return ;
        if (root->left) flatten(root->left);
        if (root->right) flatten(root->right);
        TreeNode *temp = root->right, *cur = root->left;
        root->right = root->left;
        while(root->right) root = root->right;
        root->right = temp;
    }
};

class SolutionT349 {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size();
        sort(nums1.begin(), nums1.end());
        sort(nums2.begin(), nums2.end());
        int i = 0, j = 0;
        vector<int> res;
        while(i < m && j < n) {
            if (nums1[i] > nums2[j]) j++;
            else if (nums1[i] < nums2[j]) i++;
            else {
                res.push_back(nums1[i]);
                i++; j++;
            }
            while(i+1 < m && nums1[i] == nums1[i+1]) i++;
            while(j+1 < n && nums2[j] == nums2[j+1]) j++;
        }
        return res;
    }
};

class SolutionT475 {
public:
    int findRadius(vector<int>& houses, vector<int>& heaters) {
        int res = 0, n = heaters.size(), j = 0;
        for (int i = 0; i < houses.size(); i++) {
            int cur = houses[i];
            while(j < n - 1 && abs(heaters[j] - cur) > abs(heaters[j+1] - cur)) j++;
            res = max(res, abs(heaters[j] - cur));
        }
        return res;
    }
};

class SolutionT441 {
public:
    int arrangeCoins(int n) {

    }
};

class SolutionT354 {
public:
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        sort(envelopes.begin(), envelope.end());
        int res = 0, n = envelopes.size();
        for (int i = 0; i < n; i++) {

        }
    }
};

class Solution {
public:
    int maxSumSubmatrix(vector<vector<int>>& matrix, int k) {
        int m = matrix.size(), n = matrix[0],size();
        for (int j = 0; j < n; j++) { //固定左边界
            vector<int> sum(m, 0); //固定左右边界后，每一row的和
            for (int i = j; i < n; i++) { //固定右边界
                for (int p = 0; p < m; p++) {
                    sum[p] += matrix[p][j];
                } //当前左右边界的子矩阵和
                int curSum = 0;
                set<int> st{{0}};
                for (auto a:sum) {
                    curSum += a; //当前左右边界，0-m,的子矩阵的和
                    auto it = st.lower_bound(curSum - k); 
                    //找到 《= k的最大子矩阵和 sum[j,i] = sum[i] - sum[j] <= k
                    //curSum是累加到现在的sum[i], 要在集合里找到一个sum[j]
                    // sum[j] >= sum[i] - k, 第一个不小于sum[i] - k的值，就是的
                    if ( it != st.end() ) res = max(res, curSum - *it);
                    st.insert(curSum); //当前左右边界，0-m,的子矩阵的和,放入到set中
                }
            }
        }
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
//
};

class SolutionT16 {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int n = nums.size(), res = 0;
        for (int i = 0; i <= n-3; i++) {
            if (nums[i] + nums[n-2] + nums[n-1] < target) {
                int temp_sum = nums[i] + nums[n-2] + nums[n-1]
                res = abs(target - res) < abs(target - temp_sum) ? res:temp_sum;
                continue;
            } else if (nums[i] + nums[i+1] + nums[i+2] >= target) {
                int temp_sum = nums[i] + nums[i+1] + nums[i+2];
                res = abs(target - res) < abs(target - temp_sum) ? res:temp_sum;
                break;
            }
            for (int j = i + 1; j <= n-2; j++) {
                int temp_target = target - nums[i] - nums[j];
                int left = j+1, right = n;
                while(left < right) {
                    int mid = left + (right - left)/2;
                    if (nums[mid] < temp_target) left = mid+1;
                    else right = mid;
                }
                int temp;
                if (right == n) {
                    temp_sum = nums[i] + nums[j] + nums[right-1];
                    res = abs(target - res) < abs(target - temp_sum) ? res:temp_sum;
                } else {
                    temp_sum = nums[i] + nums[j] + nums[right];
                    res = abs(target - res) < abs(target - temp_sum) ? res:temp_sum;
                    if (right > j+1) {
                        temp_sum = nums[i] + nums[j] + nums[right-1];
                        res = abs(target - res) < abs(target - temp_sum) ? res:temp_sum;
                    }
                }
            }
        }
        return res;
    }
};

class SolutionT435 {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        int n = intervals.size();
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> res;
        for (int i = 0; i < n; i++) {
            if (res.empty()) {
                res.push_back(internals[i]);
                continue;
            }
            auto temp_interval = intervals[i];
            if (temp_interval[0] >= res.back()[1]) {
                res.push_back(temp_interval);
                continue;
            }
            while (temp_interval[1] < res.back()[1]) {
                res.pop_back();
                res.push_back(temp_interval);
            }
        }
        return n - res.size();
    }
};

class SolutionT86 {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode *dummy = new ListNode(-1), *dummy2 = new ListNode(-1);
        ListNode *pre = dummy2, *tail = dummy, *cur = head;
        pre->next = head;
        while(cur) {
            if (cur->val < x) {
                tail->next = cur;
                pre->next = cur->next;
                cur->next = nullptr;
                cur = pre->next;
                tail = tail->next;
            } else {
                cur = cur->next;
                pre = pre->next;
            }
        }
        tail->next = dummy2->next;
        return dummy->next;
    }
};

class SolutionT830 {
public:
    vector<vector<int>> largeGroupPositions(string s) {
        int n = s.size();
        vector<vector<int>> res;
        for (int i = 0; i <= n-3; i++) {
            int cur = i + 1;
            while (cur <= n-1 && s[i] == s[cur]) cur++;
            if (cur - i >= 3) res.push_back({i, cur-1});
            i = cur;
        }
        return res;
    }
};

class SolutionT32 {
public:
    int longestValidParentheses(string s) {
        stack<int> st;
        int res = 0, n = s.size(), start = 0;
        for (int i = 0; i < n; i++) {
            if (s[i] == '(') st.push(i);
            else {
                if (st.empty()) start = i + 1;
                else {
                    st.pop()
                    res = st.empty()?max(res, i - start + 1):max(res, i - st.top());
                }
            }
        }
        return res;
    }
};

class SolutionT76 {
public:
    string minWindow(string s, string t) {
        unordered_map<char, int> bucket;
        for (int i = 0; i < t.size(); i++) bucket[t[i]]++;
        int left = 0, minLen = INT_MAX, n = s.size(), count = 0;
        string res = "";
        for (int i = 0; i < n; i++) {
            if (--bucket(s[i]) >= 0) ++cnt;
            while (cnt == t.size()) {
                if (minLen > i - left + 1) {
                    minLen = i - left + 1;
                    res = s.substr(left, minLen);
                }
                if (++bucket[s[left]] > 0) --cnt;
                ++left;
            }
        }
        return res;
    }
};

class SolutionT849 {
public:
    int maxDistToClosest(vector<int>& seats) {
        int res = INT_MAX, left = 0, n = seats.size(), right = n - 1;
        vector<int> dist(seats.size(), 0);
        int left = 0;
        while(seats[left] != 1) left++;
        for (int i = 0; i < n; i++) {
            if (seats[i] > 0) {
                left = max(left, i);
                continue;
            }
            else {
                dist[i] = abs(left - i);
            }
        }
        int right = 0;
        while(seats[right] != 1) right--;
        for (int i = n - 1; i >= 0; i--) {
            if (seats[i] > 0) {
                right = min(right, i);
                continue;
            }
            else {
                int temp = abs(i - right);
                dist[i] = min(dist[i], temp);
                res = min(res, seats[i]);
            }
        }
        return res;
    }
};

class SolutionT560 {
public:
    int subarraySum(vector<int>& nums, int k) {
        int res = 0, n = nums.size();
        vector<int> sums = nums;
        for (int i = 1; i < n; ++i) {
            sums[i] = sums[i - 1] + nums[i];
        }
        for (int i = 0; i < n; ++i) {
            if (sums[i] == k) ++res;
            for (int j = i - 1; j >= 0; --j) {
                if (sums[i] - sums[j] == k) ++res;
            }
        }
        return res;
    }

    int subarraySum(vector<int>& nums, int k) {
        int res = 0, sum = 0, n = nums.size();
        unordered_map<int, int> map{{0,1}};
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            res += map[sum - k];
            ++m[sum];
        }
        return res;
    }
};

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class SolutionT328 {
public:
    ListNode* oddEvenList(ListNode* head) {
        ListNode *dummy = new ListNode(-1), *tailEven = dummy, *cur = head, *pre = head;
        int count = 1;
        while(cur) {
            if (count % 2 == 0) {
                ListNode* temp = cur->next;
                tailEven->next = cur;
                pre->next = cur->next;
                cur->next = nullptr;
                cur = temp;
                tailEven = tailEven->next;
            } else {
                pre = cur;
                cur = cur->next;
            }
            count++;
        }
        pre->next = dummy->next;
        return head;
    }

    ListNode* oddEvenList(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode *odd = head, *even = head->next, *even_head = even;
        while(even && even->next) {
            odd->next = even->next;
            odd = odd->next;
            even->next = odd->next;
            even = even->next;
        }
        odd->next = even_head;
        return head;
    }
};

class SolutionT973 {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        priority_queue<pair<int, int>> pq;
        for (int i = 0; i < points.size(); i++) {
            int prod = points[i][0] * points[i][1];
            pq.push(make_pair(prod, i));
        }
        vector<vector<int>> res;
        for (int i = 0; i < K; i++) {
            auto temp = pq.top(); pq.pop();
            res.push_back(points[temp.second]);
        }
        return res;
    }

     vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        priority_queue<pair<int, int>> q;
        for (int i = 0; i < K; ++i) {
            q.emplace(points[i][0] * points[i][0] + points[i][1] * points[i][1], i);
        }
        int n = points.size();
        for (int i = K; i < n; ++i) {
            int dist = points[i][0] * points[i][0] + points[i][1] * points[i][1];
            if (dist < q.top().first) {
                q.pop();
                q.emplace(dist, i);
            }
        }
        vector<vector<int>> ans;
        while (!q.empty()) {
            ans.push_back(points[q.top().second]);
            q.pop();
        }
        return ans;
    }
};

class SolutionT1024 {
public:
    int videoStitching(vector<vector<int>>& clips, int T) {
        sort(clips.begin(), clips.end(), [](vector<int> a, vector<int> b){
            return a[0] < b[0] ? true:(a[0]==b[0]?a[1] < b[1]:false);
        });
        int start = 0, res = 0;
        while(clips[start][0] == 0) start++;
        if (start > 0) start = start - 1;
        int cur_end = clips[start][1];
        for (int i = start + 1; i < clips.size(); i++) {
            if (clips[i][0] > cur_end) {
                return -1;
            }
            int temp_end = clips[i][1];
            while(clips[i][0] <= cur_end) {
                temp_end = max(temp_end, clips[i][1]);
                i++;
            }
            res++;
            cur_end = temp_end;
        }
        return res;
    }

    int videoStitching(vector<vector<int>>& clips, int T) {
        sort(clips.begin(), clips.end(), [](vector<int> a, vector<int> b){
            return a[0] < b[0] ? true:(a[0]==b[0]?a[1] < b[1]:false);
        });
        if(T==0) return 0;
        if(clips[0][0] > 0) return -1;
        //sort(clips.begin(), clips.end());
        int start = 0, res = 1;
        while(start < clips.size() && clips[start+1][0] == 0) start++;
        int cur_end = clips[start][1];
        int i = start + 1;
        for (; i < clips.size(); i++) {
            if(cur_end >= T) break;
            if (clips[i][0] > cur_end) {
                return -1;
            }
            int temp_end = clips[i][1];
            while(i < clips.size() && clips[i][0] <= cur_end) {
                temp_end = max(temp_end, clips[i][1]);
                i++;
            }
            i = i - 1;
            res++;
            cur_end = temp_end;
        }
        return (cur_end >= T) ? res:-1;
        //return cur_end;
    }

    //dp
    int videoStitching(vector<vector<int>>& clips, int T) {
        vector<int> dp(T + 1, INT_MAX - 1);
        dp[0] = 0;
        for (int i = 1; i <= T; i++) {
            for (auto &it:clips) {
                if (it[0] < T && it[1] >= T) {
                    dp[i] = min(dp[i], dp[it[0]] + 1);
                }
            }
        }
        return dp[T] == INT_MAX - 1 ? -1 : dp[T];
    }

    //用一两个变量来存储cur-end，temp-end
    //用一个数组来存储当前位置可以到达的最远
    int videoStitching(vector<vector<int>>& clips, int T) {
        vector<int> dist(T+1);
        dist[0] = 0;
        for (vector<int>& it : clips) {
            if (it[0] < T) {
                dist[it[0]] = max(dist[it[0]], it[1]);
            }
        }
        int step = 0, reach = 0, i = 0, res = 0;
        while (reach < T) {
            int pre = reach;
            for (; i <= pre; i++) {
                reach = max(reach, dist[i]);
            }
            res++;
            if (reach <= pre) return -1;
        }
        return res;
    }
};

class SolutionT778 {
public:
    int swimInWater(vector<vector<int>>& grid) {
        int res = 0, n = grid.size();
        unordered_set<int> visited{0};
        vector<vector<int>> dirs{{0, -1}, {-1, 0}, {0, 1}, {1, 0}};
        auto cmp = [](pair<int, int>& a, pair<int, int>& b) {return a.first > b.first;};
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp) > q(cmp);
        q.push({grid[0][0], 0});
        //存放的是高度，坐标
        while(!q.empty()) {
            auto temp = q.top();
            int x = temp.second / n, y = temp.second % n;
            res = max(res, grid[x][y]);
            //每次取最大，表示通过BFS遍历到当前最小位置需要等待的时间
            if (x == n-1 && y == n-1) return res;
            for (auto dir : dirs) {
                int x = x + dir[0], y = y + dir[1];
                if (x < 0 || x >= n || y < 0 || y >= n || visited.count(x * n + y)) continue;
                visited.insert(x * n + y);
                q.push({grid[x][y], x * n + y});
            }
        }
        return res;
    }
};

class SolutionT324 {
public:
    void wiggleSort(vector<int>& nums) {
        priority_queue<int, vector<int>, less<int> > maxheap; //maxheap; 存放较小的数
        priority_queue<int, vector<int>, greater<int> > minheap; //minheap; 存放较大的数
        priority_queue<int, vector<int>, less<int> > temp_maxheap;
        int n = nums.size();
        int maxheap_size = (n + 1)/2, minheap_size = n - maxheap_size;
        for (int i = 0; i < n; i++) {
            maxheap.push(nums[i]);
            minheap.push(nums[i]);
            if (maxheap.size() > maxheap_size) maxheap.pop();
            if (minheap.size() > minheap_size) minheap.pop();
        }
        while(!minheap.empty()) {
            int temp = minheap.top(); minheap.pop();
            temp_maxheap.push(temp);
        }
        for (int i = 0; i < n; i++) {
            if (i%2 == 0) {
                nums[i] = maxheap.top(); maxheap.pop();
            } else {
                nums[i] = temp_maxheap.top(); temp_maxheap.pop();
            }
        }
    }

    void wiggleSort(vector<int>& nums) {
        vector<int> tmp = nums;
        int n = nums.size(), k = (n + 1) / 2, j = n; 
        sort(tmp.begin(), tmp.end());
        for (int i = 0; i < n; ++i) {
            nums[i] = i & 1 ? tmp[--j] : tmp[--k];
        }
    }
};

class SolutionT347 {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> map;
        for (const auto &num:nums) map[num]++;
        auto cmp = [](const pair<int, int> &a, const pair<int, int> &b){
            return a.first < b.first;
        } //less maxheap
        priority_queue<pair<int, int>, vector<pair<int,int> >, decltype(cmp)> pq(cmp);
        for (auto it:map) {
            pq.push(make_pair(it.second, it.first));
        }
        vector<int> res;
        for (int i = 0; i < k; i++) {
            auto temp = pq.top(); pq.pop();
            res.push_back(temp->second);
        }
        return res;
    }
};

class SolutionT567 {
public:
    bool checkInclusion(string s1, string s2) {
        int n1 = s1.size(), n2 = s2.size();
        vector<int> m1(128), m2(128);
        for (int i = 0; i < n1; ++i) {
            ++m1[s1[i]]; ++m2[s2[i]];
        }
        for (int i = n1; i < n2; i++) {
            ++m2[s2[i]];
            --m2[s2[i - n1]];
            if (m1 == m2) return true;
        }
        return false;
    }

    bool checkInclusion(string s1, string s2) {
        int n1 = s1.size(), n2 = s2.size(), left = 0;
        vector<int> m(128);
        for (char c : s1) ++m[c];
        for (int right = 0; right < n2; ++right) {
            if (--m[s2[right]] < 0) {
                while (++m[s2[left++]] != 0) {}
            } else if (right - left + 1 == n1) return true;
        }
        return n1 == 0;
    }
    
    bool checkInclusion(string s1, string s2) {
        int n1 = s1.size(), n2 = s2.size(), left = 0, count = n1;
        unordered_map<int, int> map;
        unordered_map<int, int> temp_map;
        for (char c : s1) ++map[c];
        temp_map = map;
        for (int i = 0; i < s2; i++) {
            if (map.find(s2[i]) != map.end()) {
                map[s2[i]]--; count--;
                if (map[s2[i]] < 0) {
                    while(map[s2[i]] < 0) {
                        map[s2[left]]++;
                        left++;
                        count++;
                    }
                }
                if (count == 0) return true;
            }
            else {
                left = i + 1;
                map = temp_map;
            }
        }
        return false;
    }
};

class SolutionT1143 {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int s1 = text1.size(), s2 = text2.size();
        vector<vector<int> > dp(s1 + 1, vector<int> (s2 + 1, 0));
        for (int i = 1; i <= s1; i++) {
            for (int j = 1; j <= s2; j++) {
                if (text1[i - 1] == text2[j - 1]) dp[i][j] = dp[i-1][j-1] + 1;
                else dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
        return dp[s1][s2];
    }
};

class SolutionT617 {
public:
    TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
        if (!t1 && !t2) return nullptr;
        if (!t1 || !t2) return t1?t1:t2;
        TreeNode* newNode = new TreeNode(t1->val + t2->val);
        newNode->left = mergeTrees(t1->left, t2->left);
        newNode->right = mergeTrees(t1->right, t2->right);
        return newNode;
    }
};

class Solution662 {
public:
    int widthOfBinaryTree(TreeNode* root) {
        queue<TreeNode*> q({root});
        int res = 1;
        while(!q.empty()){
            TreeNode* node = q.front();
            while(node == nullptr) {
                q.pop(); node = q.front();
            }
            int qSize = q.size(), lastIndex = 0;
            for (int i = 0; i < qSize; i++) {
                if (node) {
                    lastIndex = i;
                    q.push(node->left);
                    q.push(node->right);
                } else {
                    q.push(nullptr);
                    q.push(nullptr);
                }
            }
            res = max(res, lastIndex + 1);
        }
        return res;
    }

    //bfs 为什么不去管空节点呢，因为父节点是空，肯定不会有字节点，即使空父节点后面
    //还有节点，但是这个空姐点的子节点都是用来充数的，非空父节点有pos，子节点有相应更大的pos
    int widthOfBinaryTree(TreeNode* root) {
        queue<PosNode*> q;
        q.push(new TreeNode(root, 0, 0));
        int left = 0, ans = 0, curDepth = 0;
        while(!q.empty()) {
            auto temp = q.front();
            if (temp->node != nullptr) {
                q.push(new PosNode(temp->left, temp->depth + 1, temp->pos*2));
                q.push(new PosNode(temp->right, temp->depth + 1, temp->pos*2 + 1));
            }
            if (curDepth != temp->depth) {
                curDepth = temp->depth;
                left = temp->pos;
            }
            ans = max(ans, temp->pos - left + 1);
        }
    }

    //dfs
    int ans = 0;
    unordered_map<int, int> map;
    int widthOfBinaryTree(TreeNode* root) {
        dfs(root, 0, 0);
    }

    void dfs(TreeNode* node, int depth, int pos) {
        if (!node) return;
        if (map.count(depth) == 0) {
            map[depth] = pos;
        }
        ans = max(ans, pos - map[depth]);
        dfs(node->left, depth + 1, 2 * pos);
        dfs(node->right, depth + 1, 2 * pos + 1);
    }
};

class PosNode{
    TreeNode node;
    int depth, pos;
    PosNode(TreeNode n, int d, int p){
        node = n;
        depth = d;
        pos = p;
    }
}

class SolutionT315 {
public:
    vector<int> countSmaller(vector<int>& nums) {
        vector<int> temp, res(nums.size());
        for (int i = nums.size() - 1; i >= 0; i++) {
            int l = 0, r = temp.size();
            while (l < r) {
                int mid = l + (r - l) / 2;
                if (temp[mid] < nums[i]) left = mid + 1;
                else right = mid;
            }
            res[i] = right; 
            temp.insert(temp.begin() + right, nums[i]);
        }
        return res;
    }
};

class SolutionT1373 {
public:
    int findMin(TreeNode* root) {
        if (!root) return INT_MAX;
        while(root->left) {
            root = root->left;
        }
        return root->val;
    }

    int findMax(TreeNode* root) {
        if (!root) return INT_MIN;
        while(root->right) {
            root = root->right;
        }
        return root->val;
    }

    bool searchBST(TreeNode* root, int &maximum, int &cursum) {
        if (!root) {
            cursum = 0;
            return true;
        }
        int leftsum = 0;
        bool isLeft = searchBST(root->left, maximum, leftsum);
        int rightsum = 0;
        bool isRight = searchBST(root->right, maximum, rightsum);
        if(isLeft && isRight){
            int minVal = findMin(root->right);
            int maxVal = findMax(root->left);
            if (root->val > maxVal && root->val < minVal) {
                cursum = root->val + leftsum + rightsum;
                maximum = max(maximum, cursum);
            }
            return true;
        }
        cursum = 0;
        return false;
    }

    int maxSumBST(TreeNode* root) {
        if (!root) return 0;
        int maxium = 0, cursum = 0;
        searchBST(root, maxium, cursum);
        return maximum;
    }
};


class SolutionT1405 {
public:
    string longestDiverseString(int a, int b, int c) {
        priority_queue<pair<int, char>> pq;
        if(a) pq.push({a,'a'});
        if(b) pq.push({b,'b'});
        if(c) pq.push({c,'c'});
        string ans="";
        while(pq.size() > 1) {
            pair<int,char> one = pq.top();pq.pop();
            pair<int,char> two = pq.top();pq.pop();
        }
        if(one.first>=2){
                ans+=one.second;
                ans+=one.second;
                one.first-=2;
            }
            else{
                ans+=one.second;
                one.first-=1;
            }
            if(two.first>=2 && two.first>=one.first){
                ans+=two.second;
                ans+=two.second;
                two.first-=2;
            }
            else{
                ans+=two.second;
                two.first-=1;
            }
            if(one.first>0)
                pq.push(one);
            if(two.first>0)
                pq.push(two);
    }
};

class SolutionT813 {
public:
    //动态规划的基础 递归
    double largestSumOfAverages(vector<int>& A, int K) {
        int n = A.size();
        double sum = 0.0, res = 0.0;
        if (K == 1) {
            for (int i = 0; i <= n - K; i++) {
                sum += A[i];
            }
            return sum/n;
        } else {
            for (int i = 0; i <= n - K; i++) {
                sum += A[i];
                double temp = sum/(i+1)+helper(A, i + 1, K - 1); 
                res = max(temp, res);
            }
        }
        return res;
    }

    double helper(vector<int>& A, int index, int k) {
        int n = A.size();
        if (index >= n || k == 0) return 0;
        if (k == 1 && index < n) {
            double tempSum = 0.0;
            for (int i = index; i < n; i++) {
                tempSum += A[i];
            }
            return tempSum/(n - index);
        }
        double sum = 0.0, res = 0.0;
        for (int i = index; i <= n - k; i++) {
            sum += A[i];
            double temp = sum/(i-index+1)+helper(A, i + 1, k - 1); 
            res = max(temp, res);
        }
        return res;
    }

    //记忆化的递归，动态规划
    //dp[i][k]表示范围是[i, n-1]的子数组分成k组的最大得分
    double largestSumOfAverages(vector<int>& A, int K) {
        int n = A.size();
        vector<vector<double>> dp(n, vector<double>(K,0));
        vector<double> sum(n+1);
        for (int i = 1; i < n; i++) {
            sum[i]  += (sum[i - 1] + A[i]);
        }
        for (int i = 0; i < n; i++) {
            dp[i][0] = (sum[n] - sum[i]) / (n-i);
        }
        for (int k = 1; k < K; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    dp[i][k] = max(dp[i][k], (sum[j] - sum[i]) / (j - i) + dp[j][k-1]);
                }
            }
        }
        return dp[0][K-1];
    }
};


class SolutionT163 {
public:
    int maximumGap(vector<int>& nums) {
        
    }
};

class SolutionT409 {
public:
    int longestPalindrome(string s) {
        int res = 0;
        bool mid = false;
        unordered_map<char, int> m;
        for (char c : s) ++m[c];
        for (auto it = m.begin(); it != m.end(); ++it) {
            res += it->second;
            if (it->second % 2 == 1) {
                res -= 1;
                mid = true;
            } 
        }
        return mid ? res + 1 : res;
    }
};

class SolutionT207 {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> graph(numCourses, vector<int>());
        vector<int> in(numCourses);
        for (auto a : prerequisites) {
            graph[a[1]].push_back(a[0]);
            ++in[a[0]];
        }
        queue<int> q;
        for (int i = 0; i < numCourses; ++i) {
            if (in[i] == 0) q.push(i);
        }
        while (!q.empty()) {
            int t = q.front(); q.pop();
            for (auto a : graph[t]) {
                --in[a];
                if (in[a] == 0) q.push(a);
            }
        }
        for (int i = 0; i < numCourses; ++i) {
            if (in[i] != 0) return false;
        }
        return true;
    }
};

class SolutionT210 {
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> graph(numCourses, vector<int>());
        vector<int> in(numCourses);
        vector<int> path;
        for (auto a : prerequisites) {
            graph[a[1]].push_back(a[0]);
            ++in[a[0]];
        }
        queue<int> q;
        for (int i = 0; i < numCourses; ++i) {
            if (in[i] == 0) q.push(i);
        }
        while (!q.empty()) {
            int t = q.front(); q.pop();
            path.push(t);
            for (auto a : graph[t]) {
                --in[a];
                if (in[a] == 0) q.push(a);
            }
        }
        for (int i = 0; i < numCourses; ++i) {
            if (in[i] != 0) return {};
        }
        return path;
    }
};

class SolutionT417 {
public:
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& matrix) {
        vector<vector<int>> res;
        if (matrix.empty() || matrix[0].empty()) return res;
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<bool> > pacific(m, vector<bool> (n, false)), atlantic = pacific;
        queue<vector<int>> q1, q2;
        for (int i = 0; i < m; i++) {
            q1.push({i, 0});
            q2.push({i, n-1});
            pacific[i][0] = true;
            atlantic[i][n-1] = true;
        }
        for (int i = 0; i < n; ++i) {
            q1.push({0, i});
            q2.push({m - 1, i});
            pacific[0][i] = true;
            atlantic[m - 1][i] = true;
        }
        bfs(matrix, pacific, q1);
        bfs(matrix, atlantic, q2);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (pacific[i][j] && atlantic[i][j]) {
                    res.push_back({i, j});
                }
            }
        }
    }
    void bfs(vector<vector<int>>& matrix, vector<vector<bool>>& visited, queue<pair<int, int>>& q) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}};
        while (!q.empty()) {
            auto t = q.front(); q.pop();
            for (auto dir : dirs) {
                int x = t[0] + dir[0], y = t[1] + dir[1];
                if (x < 0 || x >= m || y < 0 || y >= n || visited[x][y] || matrix[x][y] < matrix[t[0]][t[1]]) continue;
                visited[x][y] = true;
                q.push({x, y});
            }
        }
    }
};

class SolutionT329 {
public:
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size(), res = INT_MIN;
        vector<vector<int>> visited(m, vector<int> (n, 1));
        //visited[i][j]以i，j为终点的最长长度
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j ++) {
                //if (visited[i][j] > 1) continue;
                dfs(matrix, i, j, visited, res);
            }
        }
        return res;
    }

    void dfs(vector<vector<int>>& matrix, int i, int j, vector<vector<int>>& visited, int& res) {
        int m = matrix.size(), n = matrix[0].size(), curLen = visited[i][j];
        vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}};
        for (auto dir:dirs){
            int x = i + dir[0], y = j + dir[1];
            if (x < 0 || x >= m || y < 0 || y >= n || matrix[x][y] <= matrix[i][j]) continue ;
            visited[x][y] = max(curLen+1, visited[x][y]);
            res = max(visited[x][y], res);
            dfs(matrix, x, y, visited, res);
        }
        return ;
    }
    //超时dfs 没有记忆化

    int longestIncreasingPath(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size(), res = 1;
        vector<vector<int>> dp(m, vector<int> (n, 0));
        //visited[i][j]以i，j为起点的最长长度
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j ++) {
                res = max(res, dfs(matrix, i, j, dp));
            }
        }
        return res;
    }

    int dfs(vector<vector<int>>& matrix, int i, int j, vector<vector<int>>& dp) {
        int m = matrix.size(), n = matrix[0].size(), mx = 1;
        if (dp[i][j]) return dp[i][j];
        vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}};
        for (auto dir:dirs){
            int x = i + dir[0], y = j + dir[1];
            if (x < 0 || x >= m || y < 0 || y >= n || matrix[x][y] <= matrix[i][j]) continue ;
            int len = 1 + dfs(matrix, x, y, dp);
            mx = max(mx, len);
        }
        dp[i][j] = mx;
        return mx;
    }

//BFS 超时
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) return 0;
        int m = matrix.size(), n = matrix[0].size(), res = 1;
        vector<vector<int>> dirs{{0,-1},{-1,0},{0,1},{1,0}};
        vector<vector<int>> dp(m, vector<int>(n, 0));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j ) {
                if (dp[i][j] > 0) continue;
                int cnt = 1;
                queue<pair<int, int>> q({i, j});
                while(!q.empty()) {
                    //每一层遍历
                    cnt++; //到下一层的长度
                    int len = q.size();
                    for (int i = 0; i < len; i++) {
                        auto t = q.front(); q.pop();
                        //该层到下一层的遍历
                        for (auto dir : dirs) {
                            int x = t.first + dir[0], y = t.second + dir[1];
                            if (x < 0 || x >= m || y < 0 || y >= n || matrix[x][y] <= matrix[t.first][t.second] || cnt <= dp[x][y]) continue;
                            dp[x][y] = max(dp[x][y], cnt);
                            res = max(res, cnt);
                            q.push({x, y});
                        }
                    }

                }
            }
        }
    }
};

class SolutionT491 {
public:
    vector<vector<int>> findSubsequences(vector<int>& nums) {
        set<vector<int>> res;
        vector<int> temp;
        helper(nums, 0, temp, res);
        return vector<vector<int>> (res.begin(), res.end());
    }

    void helper(vector<int>& nums, int start, vector<int>& out, set<vector<int>>& res) {
        if (out.size() >= 2) res.insert(out);
        for (int i = start; i < nums.size(); i++) {
            if (!out.empty() && nums[i] < out.back()) continue;
            out.push_back(nums[i]);
            helper(nums, i+1, out, res);
            out.pop_back();
        }
    }
};

class SolutionT1752 {
public:
    bool check(vector<int>& nums) {
        int minIndex = 0, minVal = nums[0];
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] > minVal) {
                minIndex = i;
                minVal = nums[i];
                //point to the first index if duplicates
            }
        }
        for (int i = minIndex + 1; i < nums.size(); i++) {
            if (nums[i] < nums[i-1]) return false;
        }
        for (int i = 0; i < minIdex; i++) {
            if (i > 0) {
                if (nums[i] < nums[i-1]) return false;
            }
        }
        if (nums[0] < nums.back()) return false;
        return true;
    }

    bool check(vector<int>& nums) {
        vector<int> temp = nums;
        sort(temp.begin(), temp.end());
        int minIndex = 0, minVal = nums[0];
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] < minVal) {
                minIndex = i;
                minVal = nums[i];
                //point to the first index if duplicates
            }
        }
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] == minVal) {
                if (checkArr(temp, nums, i)) return true;
            }
        }
        return false;
    }

    bool checkArr(vector<int> temp, vector<int> nums, int i) {
        int step = 0;
        while(i + step < nums.size()) {
            if (temp[step] != nums[i + step]) return false;
            step++;
        }
        int start = 0;
        while(step < nums.size() && start < i) {
            if (temp[step] != nums[start]) return false;
            step++; start++;
        }
        return true;
    }
};

class SolutionT1753 {
public:
    int maximumScore(int a, int b, int c) {
        vector<int> nums({a, b, c});
        sort(nums.begin(), nums.end());
        a = nums[0], b = nums[1], c = nums[2];
        if (a + b <= c) return a + b;
        else return a + b + C;
    }
};

class SolutionT1754
{
public:
    string largestMerge(string word1, string word2) 
    {
        string res;
        int i=0,    j = 0;
        while (i<word1.size() && j<word2.size())
        {   
            if (word1.substr(i) > word2.substr(j))  //c++自带的字符串比较功能
                res += word1[i++];
            else
                res += word2[j++];
        }
        if (i < word1.size())
            res += word1.substr(i);
        if (j < word2.size())
            res += word2.substr(j);
        return res;
    }
};

class SolutionT1755 {
public:
    vector<int> make(vector<int> nums) {
        vector<int> ans(1 << nums.size());
        for (int i = 0; i < nums.size(); i++) {
            for (int j = 0; j < (1 << i); j++) {
                ans[j + (1 << i)] = ans[j] + nums[i];
            }
        }
        return ans;
    }

    int minAbsDifference(vector<int>& nums, int goal) {
        int n = nums.size();
        vector<int> left = make({nums.begin(), nums.begin() +n/2});
        vector<int> right = make({nums.begin()+n/2, nums.end()});
        sort(left.begin(), left.end());
        sort(right.rbegin(), right.rend());
        int ans = INT_MAX, i = 0, j = 0;
        while(i < left.size() && j < right.size()) {
            int temp = left[i] + right[j];
            ans = min(ans, abs(goal - temp));
            if (t > goal) j++;
            else if(t < goal) i++;
            else return 0;
        }
        return ans;
    }
};

class SolutionT22 {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> res;
        string temp = "";
        dfs(n , n, res, temp);
        return res;
    }

    void dfs(int left, int right, vector<string>& res, string& temp) {
        if (left > right || left < 0 || right < 0) return;
        if (left == 0 && right == 0) {
            res.push_back(temp);
        }
        dfs(left-1, right, res, temp+"(");
        dfs(left, right-1, res, temp+")");
        return ;
    }
};


class SolutionT1748 {
public:
    int sumOfUnique(vector<int>& nums) {
        int cnt[101] = {}, res = 0;
        for (auto n : nums)
            res += ++cnt[n] == 1 ? n : cnt[n] == 2 ? - n : 0;
        return res;
    }
};

class SolutionT1749 {
public:
    int maxAbsoluteSum(vector<int>& nums) {
        int res = INT_MIN, tempSum = 0;
        int sumFlag = 1;
        for (int i = 0; i < nums.size(); i++) {
            tempSum = 0;
            int sumFlag = 1;
            for (int j = i; j < nums.size(); j++) {
                if (tempSum == 0) {
                    tempSum = tempSum + nums[j];
                    sumFlag = tempSum > 0 ? 1 : -1;
                    res = max(res, abs(tempSum));
                    continue;
                }
                int temp = tempSum + nums[j];
                if (sumFlag > 0 && temp <= 0) break;
                if (sumFlag < 0 && temp >= 0) break;
                tempSum = temp;
                res = max(res, abs(tempSum));
            }
        }
        return res;
    }
};

class SolutionT767 {
public:
    string reorganizeString(string S) {
        string res = "";
        unordered_map<char, int> m;
        priority_queue<pair<int, char>> q;
        for (char c : S) ++m[c];
        for (auto a : m) {
            if (a.second > (S.size() + 1) / 2) return "";
            //只要满足了这个条件，就一定可以成功
            q.push({a.second, a.first});
        }
        while (q.size() >= 2) {
            auto t1 = q.top(); q.pop();
            auto t2 = q.top(); q.pop();
            res.push_back(t1.second);
            res.push_back(t2.second);
            if (--t1.first > 0) q.push(t1);
            if (--t2.first > 0) q.push(t2);
            //因为堆重新排列的结构，即使个数相同，后来的元素也在后面，排除了你那种，如果-1后和之后的字母个数相同，
            //要不要判断会不会和之前的那个字母重复
        }
        if (q.size() > 0) res.push_back(q.top().second);
        return res;
    }
};

class SolutionT368 {
public:
    vector<int> largestDivisibleSubset(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<int> dp(nums.size(), 0), parent(nums.size(), 0), res;
        int mx = 0, mx_idx = 0;
        for (int i = nums.size() - 1; i >= 0; --i) {
            for (int j = i; j < nums.size(); ++j) {
                if (nums[j] % nums[i] == 0 && dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                    parent[i] = j;
                    if (mx < dp[i]) {
                        mx = dp[i];
                        mx_idx = i;
                    }
                }
            }
        }
        for (int i = 0; i < mx; ++i) {
            res.push_back(nums[mx_idx]);
            mx_idx = parent[mx_idx];
        }
    }
     

    vector<int> largestDivisibleSubset(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<int> dp(nums.size(), 0), parent(nums.size(), 0);
        int mx = 0, mxIdx = 0;
        for (int i = 0 i < nums.size; i++) {
            for (int j = i; j < nums.size(); j++) {
                if (nums[j] % nums[i] == 0 && dp[j] < dp[i] + 1) {
                    dp[j] = dp[i] + 1;
                    parent[j] = i;
                    if (dp[j] > mx) {
                        mx = dp[i];
                        mxIdx = j;
                    }
                }
            }
        }
        vector<int> res;
        for (int i = 0; i < mx; i++) {
            res.push_back(nums[mxIdx]);
            mxIdx = parent[mxIdx];
        }
        return res;
    }
};

class SolutionT516 {
public:
    int longestPalindromeSubseq(string s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n));
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {

            }
        }
    }
};

class SolutionT995 {
public:
    int minKBitFlips(vector<int>& A, int K) {

    }
};

class SolutionT1534 {
public:
    int countGoodTriplets(vector<int>& arr, int a, int b, int c) {

    }
};

class SolutionT1751 {
public:
    vector<vector<int>> dp;

    int dfs(vector<vector<int>>& events, int pos, int k) {
        if (pos >= events.size() || k == 0) return 0;
        if (dp[pos][k] != -1) return dp[pos][k];
        auto j = upper_bound(events.begin() + pos, events.end(), events[pos][1], [](int t, const vector<int>
        &v){v[0] > t;} - events.begin()) //因为找的是绝对位置，也就是相对于第一个位置的偏移量
        //要这个 不要这个 比大小
        return dp[i][k] = max(e[i][2] + dfs(e, j, k - 1), dfs(e, i + 1, k));
        //e[i][2]是当前值
    }

    int maxValue(vector<vector<int>>& events, int k) {
        sort(events.begin(), events.end());
        dp = vector<vector<int>>(events.size(), vector<int>(k + 1, -1));
        return dfs(events, 0, k);
    }
};

class SolutionT1756 {
public:
    string modifyString(string s) {
        for (int i = 0; i < s.size(); i++);
    }
};