#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <map>
#include <set>
#include <stack>
#include <utility>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <hash_map>
#include <deque>
using namespace std;

//反转链表
class SolutionT206 {
public:
//递归
    ListNode* reverseList(ListNode* head) {
        if (!head||!head->next) return head;
        ListNode* dummy = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return dummy;
    }
//迭代
    //你自己的插入
    //or
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr;
        ListNode* cur = head;
        while(head!=nullptr) {
            ListNode* nextTemp = cur->next;
            cur->next = pre;
            pre = cur;
            cur = nextTemp;
        }
        return pre;
    }
};

//K个翻转链表
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

    // public ListNode reverseKGroup(ListNode head, int k) {
    //     if (head == null || head.next == null) {
    //         return head;
    //     }
    //     ListNode tail = head;
    //     for (int i = 0; i < k; i++) {
    //         //剩余数量小于k的话，则不需要反转。
    //         if (tail == null) {
    //             return head;
    //         }
    //         tail = tail.next;
    //     }
    //     // 反转前 k 个元素
    //     ListNode newHead = reverse(head, tail);
    //     //下一轮的开始的地方就是tail
    //     head.next = reverseKGroup(tail, k);

    //     return newHead;
    // }

    // /*
    // 左闭又开区间
    //  */
    // private ListNode reverse(ListNode head, ListNode tail) {
    //     ListNode pre = null;
    //     ListNode next = null;
    //     while (head != tail) {
    //         next = head.next;
    //         head.next = pre;
    //         pre = head;
    //         head = next;
    //     }
    //     return pre;

    // }
};

class LRUCache{
public:
    LRUCache(int capacity) {
        cap = capacity;
    }
    
    int get(int key) {
        auto it = m.find(key);
        if (it != m.end()) {
            l.splice(l.begin(), l, it->second);
            return it->second->second;
        } else {
            return -1;
        }
    }
    
    void put(int key, int value) {
        auto it = m.find(key);
        if (it != m.end()) {
            l.erase(it->second);
        }
        l.push_frond(make_pair(key, value));
        m[key] = l.begin();
        if (l.size() > cap) {
            int k = l.rbegin()->first;
            m.erase(k);
            l.pop_back();
        }
    }
    
private:
    int cap;
    list<pair<int, int>> l;
    unordered_map<int, pair<int,int>> m;
};

//最长无重复子串
class SolutionT3 {
public:
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
};

//完全二叉树
class SolutionT958 {
public:
//具体到写法就是先把根结点放入到队列中，然后进行循环，条件是队首结点不为空。在循环中取出队首结点，
//然后将其左右子结点加入队列中，这里不必判断子结点是否为空，为空照样加入队列，因为一旦取出空结点，
//循环就会停止。然后再用个循环将队首所有的空结点都移除，这样若是完全二叉树的话，队列中所有还剩的
//结点都应该是空结点，且都会被移除，若队列中存在非空结点，说明不是完全二叉树，最后只要判断队列是否为空即可，
    bool isCompleteTree(TreeNode* root) {
        queue<TreeNode*> q{{root}};
        while (q.front() != NULL) {
            TreeNode *cur = q.front(); q.pop();
            q.push(cur->left);
            q.push(cur->right);
        }
        while (!q.empty() && q.front() == NULL) {
            q.pop();
        }
        return q.empty();
    }
};

class SolutionT34 {
public:
    void reverseWords(string &s) {
        int storeIndex = 0, n = s.size();
        reverse(s.begin(), s.end());
        for (int i = 0; i < n; ++i) {
            if (s[i] != ' ') {
                if (storeIndex != 0) s[storeIndex++] = ' ';
                int j = i;
                while (j < n && s[j] != ' ') s[storeIndex++] = s[j++];
                reverse(s.begin() + storeIndex - (j - i), s.begin() + storeIndex);
                i = j;
            }
        }
        s.resize(storeIndex);
    }
};

//字符串相加 大数相加
class Solution {
public:
    string addStrings(string num1, string num2) {
        string res = "";
        int m = num1.size(), n = num2.size(), i = m - 1, j = n - 1;
        int sum = 0, carry = 0;
        while (i >= 0 || j >= 0) {
            int a = i >= 0 ? num1[i] - '0' : 0;
            int b = j >= 0 ? num2[j] - '0' : 0;
            sum = a + b + carry;
            carry = sum/10;
            res.insert(res.begin(), sum%10 + '0');

        }
        return carry == 1? "1" + res:res;
    }
};

class SolutionT4 {
public:
//若 m+n 为奇数的话，那么其实 (m+n+1) / 2 和 (m+n+2) / 2 的值相等，相当于两个相同的数字相加再除以2，还是其本身
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size(), left = (m + n + 1)/2, right = (m + n + 2)/2;
        return (findKth(nums1, 0, nums2, 0, left) + findKth(nums1, 0, nums2, 0, right)) / 2.0;
    }

    int findKth(vector<int>& nums1, int i, vector<int>& nums2, int j, int k) {
        if (i >= nums1.size()) return nums2[j + k - 1];
        if (j >= nums2.size()) return nums1[i + k - 1];
        if (k == 1) return min(nums1[0], nums2[0]);
        int midVal1 = (i + k / 2 - 1 < nums1.size())?nums1[i + k/2 - 1]:INT_MAX;
        int midVal2 = (j + k / 2 - 1 < nums2.size()) ? nums2[j + k / 2 - 1] : INT_MAX;
        if (midVal1 < midVal2) {
            return findKth(nums1, i + k / 2, nums2, j, k - k/2);
        } else {
            return findKth(nums1, i, nums2, j + k / 2, k - k / 2);
        }
    }
};

class SolutionT394 {
public:
    string decodeString(string s) {
        return decode(s, 0);
    }

    string decode(string s, int index) {
        string res = "";
        int n = s.size();
        while (index < n) {
            if (s[index] < '0' || s[index] > '9') {
                res.push_back(s[index++]);
            } else {
                int cnt = 0;
                while (s[index] >= '0' && s[index] <= '9') {
                    cnt = cnt*10 + s[index++] - '0';
                }
                i++;//左括号
                string temp = decode(s, index);
                i++;//右括号 这两个i++就是精髓
                while(cnt>0) {
                    cnt--;
                    res.append(temp);
                }
            }
        }
        return res;
    }
};

class SolutionT33 {
public:
    int search(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1;
        while (l < r) {
            int mid = l + (r - l)/2;
            if (nums[mid] == target) return mid;
            if (nums[mid] < nums[right]) {
                if (nums[mid] < target && nums[right] >= target) left = mid + 1;
                else right = mid - 1; 
            } else {
                if (nums[left] <= target || nums[mid] > target) right = mid - 1;
                else left = mid + 1;
            }
        }
        return -1;
    }
};

class SolutionT19 {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode *pre = head, *cur = head;
        for (int i = 0; i < n; i++) {
            cur = cur->next;
        }
        if (!cur) return pre->next;
        while (cur->next) {
            cur = cur->next;
            pre = pre->next;
        }
        pre->next = pre->next->next;
        return head;
    }
};

class SolutionT739 {
public:
    vector<int> dailyTemperatures(vector<int>& T) {
        vector<int> res(T.size(), 0);
        stack<int> st;
        for (int i = 0; i < T.size(); i++) {
            while(!st.empty() && nums[i] > nums[st.top()]) {
                auto temp = st.top(); st.pop();
                res[temp] = i - temp;
            }
            st.push(i);
        }
        return res;
    }
};

//桶排序,这个桶排序 有点牛逼子
class SolutionT164 {
public:
    int maximumGap(vector<int>& nums) {
        if (nums.size() < 2) return 0;
        if (nums.size() <= 1) return 0;
        int mx = INT_MIN, mn = INT_MAX, n = nums.size(), pre = 0, res = 0;
        for (int num : nums) {
            mx = max(mx, num);
            mn = min(mn, num);
        }
        int size = (mx - mn) / n + 1;//最大差距/个数, 每个桶容纳的数据范围
        int cnt = (mx - mn) / n + 1; //最大差距/桶个容量，桶的数量
        vector<int> bucket_min(cnt, INT_MAX), bucket_max(cnt, INT_MIN);
        for ( int num : nums) {
            int idx = (num - mn) / size;
            bucket_min[idx] = min(bucket_min[idx], num);
            bucket_max[idx] = max(bucket_max[idx], num);
        }
        for (int i = 1; i < cnt; ++i) {
            if (bucket_min[i] == INT_MAX || bucket_max[i] == INT_MIN) continue;
            res = max(res, bucket_min[i] - bucket_max[pre]);
            pre = i;
        }
        return res;
    }
};

class SolutionT315 {
public:
    vector<int> countSmaller(vector<int>& nums) {
        vector<int> temp, res(nums.size());
        for (int i = nums.size() - 1; i >= 0; i--) {
            int left = 0, right = temp.size();
            while(left < right) {
                int mid = left + (right - l)/2;
                if (temp[mid] < nums[i]) left = mid + 1;
                else right = mid;
            }
            res[i] = right;
            temp.insert(temp.begin() + right, nums[i]);
        }
        return res;
    }
};

//懒汉模式 直到我第一次调用getInstance才会实例化一个对象
class Singleton {
public:
    static Singleton* getInstance() {
        if (p == nullptr) {
            p = new Singleton();
        }
        return p;
    }
private:
    Singleton() = default;
    static Singleton* p;
};

Singleton* Singleton::p = nullptr;

//线程安全的懒汉
class Singleton {
public:
    static Singleton* getInstance() { 
        if (p == nullptr) {
            pthread_mutex_lock(&mutex);
            if (p == nullptr) {
                p = new Singleton();
            }
            pthread_mutex_unlock(&mutex);
        }

    }
private:
    Singleton() {}
    static Singleton* p;
    static pthread_mutex_t mutex;
}


//饿汉模式 static成员初始化的时候即实例化
class Singleton {
public:
    static Singleton* getInstance() {
        return p;
    }
private:
    Singleton() {}
    static Singleton* p;
}

Singleton* Singleton:: p = new Singleton();


//int a[][] 用new分配和释放一下内存
int **a;
a = new int*[M];    
for(int i = 0;i < M; i++)    
    p[i] = new int[N];    

//释放
for(int i=0;i<M;i++) 
    delete [] a[i]; 
delete [] a;
//栈分配内存的时候直接移动栈顶指针，堆分配内存的时候需要先找到合适的内存再进行分配。

作者：未必就是我
链接：https://www.nowcoder.com/discuss/541068
来源：牛客网

class a{
    char ch;
    int b;

public:
    virtual void func(){
        cout << "func" << endl;
    }
    void func1() {
        cout << "func1" << endl;
    }
    void func2(){
        cout << "func2" << endl;
        cout << b << endl;
    }
};
a *p = nullptr;

p->func(); // 异常，没有构造对象也就没有办法获取到指向虚表的指针，那么就没有办法得到虚函数的地址
p->func1(); // 正常运行
p->func2(); // 异常，类的成员变量没有初始化
// 作者：未必就是我
// 链接：https://www.nowcoder.com/discuss/541068
// 来源：牛客网

class Solution {
public:
    void reverseWords(string &s) {
        int storeIndex = 0, n = s.size();
        reverse(s.begin(), s.end());
        for (int i = 0; i < n; i++) {
            if (s[i] != '.') {
                if (i != 0) s[storeIndex] = '.';
                int j = i;
                while(j < n && s[j] != '.') s[storeIndex++] = s[j++];
                reverse(s.begin() + storeIndex, s.begin() + storeIndex - (j - i));
                i = j;
            }
        }
        s.resize(storeIndex);
    }
};

class SolutionT316 {
public:
    string removeDuplicateLetters(string s) {
        int m[256] = {0}, visited[256] = {0};
        string res = "0";
        for (auto a : s) ++m[a];
        for (auto a : s) {
            --m[a];
            if (visited[a]) continue;
            while (a < res.back() && m[res.back()] >0) {
                visited[res.back()] = 0;
                res.pop_back();
            }
            res+=a;
            visited[a] = 1;
        }
        return s.substr(1);
    }
};

//大端小端
#include <iostream>
using namespace std;

int main()
{
	int i = 0x12345678;

	if (*((char*)&i) == 0x12)
		cout << "大端" << endl;
	else	
		cout << "小端" << endl;

	return 0;
}

class RoleOperation
{
public:
    virtual std::string Op() = 0; // 纯虚函数
    virtual ~RoleOperation() {} // 虚析构函数
};

class RootAdminRole : public RoleOperation {
public:
    RootAdminRole(const std::string &roleName)
            : m_RoleName(roleName) {}

    std::string Op() {
        return m_RoleName + " has A permission";
    }

private:
    std::string m_RoleName;
};


// 订单管理员(有 B 操作权限)
class OrderAdminRole : public RoleOperation {
public:
    OrderAdminRole(const std::string &roleName)
            : m_RoleName(roleName) {}

    std::string Op() {
        return m_RoleName + " has B permission";
    }

private:
    std::string m_RoleName;
};

// 普通用户(有 C 操作权限)
class NormalRole : public RoleOperation {
public:
    NormalRole(const std::string &roleName)
            : m_RoleName(roleName) {}

    std::string Op() {
        return m_RoleName + " has C permission";
    }

private:
    std::string m_RoleName;
};

// 角色工厂
class RoleFactory {
public:
    // 获取工厂单例，工厂的实例是唯一的
    static RoleFactory& Instance() {
        static RoleFactory instance; // C++11 以上线程安全
        return instance;
    }

    // 把指针对象注册到工厂
    void RegisterRole(const std::string& name, RoleOperation* registrar) {
        m_RoleRegistry[name] = registrar;
    }

    // 根据名字name，获取对应的角色指针对象
    RoleOperation* GetRole(const std::string& name) {

        std::map<std::string, RoleOperation*>::iterator it;

        // 从map找到已经注册过的角色，并返回角色指针对象
        it = m_RoleRegistry.find(name);
        if (it != m_RoleRegistry.end()) {
            return it->second;
        }

        return nullptr; // 未注册该角色，则返回空指针
    }

private:
    // 禁止外部构造和虚构
    RoleFactory() {}
    ~RoleFactory() {}

    // 禁止外部拷贝和赋值操作
    RoleFactory(const RoleFactory &);
    const RoleFactory &operator=(const RoleFactory &);

    // 保存注册过的角色，key:角色名称 , value:角色指针对象
    std::map<std::string, RoleOperation *> m_RoleRegistry;
};

void InitializeRole() // 初始化角色到工厂
{
    static bool bInitialized = false;

    if (bInitialized == false) {
        // 注册系统管理员
        RoleFactory::Instance().RegisterRole("ROLE_ROOT_ADMIN", new RootAdminRole("ROLE_ROOT_ADMIN"));
        // 注册订单管理员
        RoleFactory::Instance().RegisterRole("ROLE_ORDER_ADMIN", new OrderAdminRole("ROLE_ORDER_ADMIN"));
        // 注册普通用户
        RoleFactory::Instance().RegisterRole("ROLE_NORMAL", new NormalRole("ROLE_NORMAL"));
        bInitialized = true;
    }
}

class JudgeRole {
public:
    std::string Judge(const std::string &roleName) {
        return RoleFactory::Instance().GetRole(roleName)->Op();
    }
};

int main() {
    InitializeRole(); // 优先初始化所有角色到工厂

    JudgeRole judgeRole;

    std::cout << judgeRole.Judge("ROLE_ROOT_ADMIN") << std::endl;
    std::cout << judgeRole.Judge("ROLE_ORDER_ADMIN") << std::endl;
    std::cout << judgeRole.Judge("ROLE_NORMAL") << std::endl;
}


#include <iostream>
using namespace std;
enum PRODUCTTYPE {SFJ,XSL,NAS};
class soapBase
{
	public:
	virtual ~soapBase(){};
	virtual void show() = 0;
};
 
class SFJSoap:public soapBase
{
	public:
	void show() {cout<<"SFJ Soap!"<<endl;}
};
 
class XSLSoap:public soapBase
{
	public:
	void show() {cout<<"XSL Soap!"<<endl;}
};
 
class NASSoap:public soapBase
{
	public:
	void show() {cout<<"NAS Soap!"<<endl;}
};
 
class Factory
{
	public:
	soapBase * creatSoap(PRODUCTTYPE type)
	{
		switch(type)
		{
			case SFJ: 
				return new SFJSoap();
				break;
			case XSL:
				return new XSLSoap();
				break;
			case NAS:
				return new NASSoap();
				break;
			default:break;
		}
		
	}
};
 
int main()
{
	Factory factory;
	soapBase* pSoap1 = factory.creatSoap(SFJ);
	pSoap1->show();
	soapBase* pSoap2 = factory.creatSoap(XSL);
	pSoap2->show();
	soapBase* pSoap3 = factory.creatSoap(NAS);
	pSoap3->show();
	delete pSoap1;
	delete pSoap2;
	delete pSoap3;
	return 0;
}

class Solution {
public:
    string decodeString(string s) {
        int i = 0;
        return decode(s, i);
    }
    string decode(string s, int& i) {
        string res = "";
        int n = s.size();
        while (i < n && s[i] != ']') {
            if (s[i] < '0' || s[i] > '9') {
                res += s[i++];
            } else {
                int cnt = 0;
                while (s[i] >= '0' && s[i] <= '9') {
                    cnt = cnt * 10 + s[i++] - '0';
                }
                ++i;
                string t = decode(s, i);
                ++i;
                while (cnt-- > 0) {
                    res += t;
                }
            }
        }
        return res;
    }
};

//手写死锁
//单线程
#include <iostream>
#include <thread>
#include <mutex>
#include <unistd.h>

using namespace std;

int data = 1;
mutex mt1,mt2;

void a2() {
	data = data * data;
	mt1.lock();  //第二次申请对mt1上锁，但是上不上去
	cout<<data<<endl;
	mt1.unlock();
}
void a1() {
	mt1.lock();  //第一次对mt1上锁
	data = data+1;
	a2();
	cout<<data<<endl;
	mt1.unlock();
}

int main() {
	thread t1(a1);
	t1.join();
	cout<<"main here"<<endl;
	return 0;
}

//双线程
#include <iostream>
#include <thread>
#include <mutex>
#include <unistd.h>

using namespace std;

int data = 1;
mutex mt1,mt2;

void a2() {
	mt2.lock();
	sleep(1);
	data = data * data;
	mt1.lock();  //此时a1已经对mt1上锁，所以要等待
	cout<<data<<endl;
	mt1.unlock();
	mt2.unlock();
}
void a1() {
	mt1.lock();
	sleep(1);
	data = data+1;
	mt2.lock();  //此时a2已经对mt2上锁，所以要等待
	cout<<data<<endl;
	mt2.unlock();
	mt1.unlock();
}

int main() {
	thread t2(a2);
	thread t1(a1);
	
	t1.join();
	t2.join();
	cout<<"main here"<<endl;  //要t1线程、t2线程都执行完毕后才会执行
	return 0;
}