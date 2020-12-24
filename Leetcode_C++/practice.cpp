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
