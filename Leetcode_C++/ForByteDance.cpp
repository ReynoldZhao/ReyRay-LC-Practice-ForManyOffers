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
#include <memory>
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
        while(cur!=nullptr) {
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

class LRUCache{
public:
    LRUCache(int capacity) {
        cap = capacity;
    }
    
    int get(int key) {
        auto it = map.find(key);
        if (it == map.end()) return -1;
        l.splice(l.begin(), l, map[key]);
        return map[key].second;
    }
    
    void put(int key, int value) {
        auto it = map.find(key);
        if (it!=map.end()) l.erase(map[key]);
        l.push_front(make_pair<key, val>);
        m[key] = l.begin();
        if (m.size() > cap) {
            int k = l.rebegin()->first;
            l.pop_back();
            map.erase(k);
        }

    }
    
private:
    int cap;
    list<pair<int, int>> l; //k-v
    unordered_map<int, list<pair<int, int> >::iterator> map;
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
        int i = 0;
        return decode(s, i);
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
                index++;//左括号
                string temp = decode(s, index);
                index++;//右括号 这两个i++就是精髓
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
class Singleton{
public:
	static Singleton* getInstance() {
		if (p == nullptr) {
			pthread_mutex_lock(&mutex);
			if (p == nullptr)
				p = new Singleton();
			ptread_mutex_unlock(&mutex);
		}
		return p;
	}
private:
	Singleton() {
		pthread_mutex_init(&mutex);
	}
	static Singleton *p;
	static pthread_mutex_t mutex;
};
Singleton* Singleton::p = nullptr;
pthread_mutex_t Singleton::mutex;

class Singleton{
public:
    static Singleton* getInstance() {
        if (p == nullptr) {
            if (p == nullptr) {
                p = new Singleton();
            }
        }
        return p;
    }
private:
    Singleton() {
        pthread_mutex_init(&mutex);
    }
    static Singleton *p;
    static pthread_mutex_t mutex;
}
class Singleton{
public:
    static Singleton* getInstance() {
        if (p == nullptr) {
            lock(mutex);
            if (p == nullptr) {
                p = new Singleton();
            }
            unlock(mutex);
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
    a[i] = new int[N];    

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

struct ListNode {
      int val;
      struct ListNode *next;
      ListNode(int x) :
            val(x), next(NULL) {
      }
};

//智能指针

*(*(a+m)+n)
void bar(unique_ptr<ListNode> li){

}

void foo() {
    auto e = std::make_unique<
}

void swap(int &a, int &b) {
    a = a + b;
    b = a - b;
    a = a - b;

    a = a ^ b;
    b = a ^ b;
    a = a ^ b;
}

class Solution {
public:
    Node* head = NULL;
    Node* pre = NULL;
    Node* treeToDoublyList(Node* root) {
        if (!root) return nullptr;
        stack<Node*> st;
        Node* root = p;
        while(p || !st.empty()) {
            while(p) {
                st.push_back(p);
                p = p->left;
            }
            p = st.top(); st.pop();
            if (!pre) {
                pre = st.top();
                head = pre;
            }
            else {
                pre->right = p;
                p->left = pre;
                pre = pre->right;
            }
            p = p->right;
        }
    }
};

class Date
{
public:
	Date(int year = 2017, int month = 9, int day = 10)
		: _year(year)
		, _month(month)
		, _day(day)
	{}
	void Display();    //显示函数
	void SetDate();    //获取日期函数
	void AddDate();    //日期加一函数
	void SubDate();    //日期减一函数
	
 
	Date& operator=(const Date& d);    //赋值
	Date& operator++();      // 前置++ 
	Date operator++(int);    // 后置++ 
	Date& operator--();      //前置--
	Date operator--(int);    //后置-- 
	Date operator+(int days);  //days天之后的日期  
	Date operator-(int days);  // days天之前的日期
 
	int operator-( Date& d);                    // 两个日期之间的距离 （方式一）
	friend int SubDateDays(Date &x,Date &y);    // (方式二)两个日期之间的距离 
 
	bool operator==(const Date& d);
	bool operator!=(const Date& d);
	bool operator>(const Date& d);
	bool operator<(const Date& d);
 
public:
	int _year;
	int _month;
	int _day;
};


//智能指针的线程安全问题
// 1.演示引用计数线程安全问题，就把AddRefCount和SubRefCount中的锁去掉
// 2.演示可能不出现线程安全问题，因为线程安全问题是偶现性问题，main函数的n改大一些概率就变大了，就容易出现了。
// 3.下面代码我们使用SharedPtr演示，是为了方便演示引用计数的线程安全问题，将代码中的SharedPtr换成sshared_ptr
//进行测试，可以验证库的shared_ptr，发现结论是一样的。
void SharePtrFunc(shared_ptr<Date>& sp, size_t n) {
    cout << sp.get() << endl;
    for (size_t i = 0; i < n; ++i) {
        shared_ptr<Date> copyPtr(sp);
        copyPtr->_year++;
        copyPtr->_month++;
        copyPtr->_day++;
    }
}

int main() {
    shared_ptr<Date> p(new Date);
    cout << p.get() << endl;
    const size_t n = 100;
    thread t1(SharePtrFunc, p, n);
    thread t2(SharePtrFunc, p, n);
    t1.join();
    t2.join();
    cout << p->_year << endl;
	cout << p->_month << endl;
	cout << p->_day << endl;
	return 0;
}

//shared_ptr 循环引用
class B;
class A {
public:
    shared_ptr<B> ptr; 
   // weak_ptr<B> ptr;
};

class B {
public:
    shared_ptr<A> ptr;
    //weak_ptr<A> ptr;
};

int main() {
    while (true)
    {   // A aObject = new A();
        // pa 指向 aObject pb->ptr 指向aObject
        // B bObject = new B();
        // pb 指向 aObject pa->ptr 指向aObject
        shared_ptr<A> pa(new A());
        shared_ptr<B> pb(new B());
        pa->ptr = pb;
        pb->ptr = pa;
    }
// class A和class B的对象各自被两个智能指针管理，也就是A object和B object引用计数都为2，为什么是2？

// 分析class A对象的引用情况，该对象被main函数中的pa和class B对象中的ptr管理，因此A object引用计数是2，
// B object同理。

// 在这种情况下，在main函数中一个while循环结束的时候，pa和pb的析构函数被调用，但是class A对象和class B对象
// 仍然被一个智能指针管理，
// A object和B object引用计数变成1，于是这两个对象的内存无法被释放，造成内存泄漏，
}

//循环引用
struct ListNode{
    int _data;
    shared_ptr<ListNode> _prev;
    shared_ptr<ListNode> _next;
    ~ListNode(){ cout << "~ListNode()" << endl; }
}

int main() {
    shared_ptr<ListNode> node1(new ListNode);
    shared_ptr<ListNode> node2(new ListNode);
    node1->_next = node2;
	node2->_prev = node1;
// node1和node2两个智能指针对象指向两个节点，引用计数变为1，我们不需要手动delete
// node1的_next指向node2，node2的_prev指向node1，引用计数变成2
// node1和node2析构，引用计数减到1，但是_next还指向下一个节点，_prev指向上一个节点
// 也就是说_next析构了，node2释放了
// 也就是说_prev析构了，node1释放了
// 但是_next属于node的成员，node1释放了，_next才会析构，而node1由_prev管理，_prev属于node2成员，
// 所以这就叫循环引用，谁都不会释放
}

#include <iostream>       // std::cout
#include <chrono>         // std::chrono::milliseconds
#include <thread>         // std::thread
#include <mutex>          // std::mutex

std::mutex foo, bar;

void task_a() {
    std::lock(foo, bar);
    std::unique_lock<std::mutex> lck1(foo, std::adopt_lock);
    std::unique_lock<std::mutex> lck2(bar, std::adopt_lock);
    std::cout << "task a\n";
}

void task_b() {
	// foo.lock(); bar.lock(); // replaced by:
	std::unique_lock<std::mutex> lck1, lck2;
	lck1 = std::unique_lock<std::mutex>(bar, std::defer_lock);
	lck2 = std::unique_lock<std::mutex>(foo, std::defer_lock);
	std::lock(lck1, lck2);       // simultaneous lock (prevents deadlock)
	std::cout << "task b\n";
	// (unlocked automatically on destruction of lck1 and lck2)
}

struct bank_account{
    explicit bank_account(string name, int money)
    {
        sName = name;
        iMoney = money;
    }

    string sName;
    int iMoney;
    mutex mMutex;//账户都有一个锁mutex 
};

void transfer(bank_account &from, bank_account &to, int amount)//这里缺少一个from==to的条件判断个人觉得  
{
    unique_lock<mutex> lock1(from.mMutex, defer_lock);//defer_lock表示延迟加锁，此处只管理mutex  
    unique_lock<mutex> lock2(to.mMutex, defer_lock);
    lock(lock1, lock2);//lock一次性锁住多个mutex防止deadlock,这个是关键  
    from.iMoney -= amount;
    to.iMoney += amount;
    cout << "Transfer " << amount << " from " << from.sName << " to " << to.sName << endl;
}

void main()
{
    bank_account Account1("User1", 100);
    bank_account Account2("User2", 50);
    thread t1([&](){ transfer(Account1, Account2, 10)}; );
    thread t2([&]() { transfer(Account2, Account1, 5); });
    t1.join();
    t2.join();

    system("pause");
}



//生产者消费者
#include <iostream>           
#include <queue>
#include <thread>             
#include <mutex>              
#include <condition_variable> 
using namespace std;

mutex mtx;
condition_variable produce, consume;  // 条件变量是一种同步机制，要和mutex以及lock一起使用
queue<int> q;     // shared value by producers and consumers, which is the critical section
int maxSize = 20;

void consumer() 
{
    while (true)
    {
        this_thread::sleep_for(chrono::milliseconds(1000));
        unique_lock<mutex> lck(mtx);                        
        // RAII，程序运行到此block的外面（进入下一个while循环之前），资源（内存）自动释放
        consume.wait(lck, [] {return q.size() != 0; });     
        // wait(block) consumer until q.size() != 0 is true 如果queue为空就阻塞 否则不阻塞
        cout << "consumer " << this_thread::get_id() << ": ";
        q.pop();
        cout << q.size() << '\n';
        produce.notify_all();                              
        // nodity(wake up) producer when q.size() != maxSize is true 当queue不是满的时候通知
    }
}

void producer(int id)
{
    while (true)
    {
        this_thread::sleep_for(chrono::milliseconds(900));     
        // producer is a little faster than consumer  
        unique_lock<mutex> lck(mtx);
        produce.wait(lck, [] {return q.size() != maxSize; });   
        // wait(block) producer until q.size() != maxSize is true
        cout << "-> producer " << this_thread::get_id() << ": ";
        q.push(id);
        cout << q.size() << '\n';
        consume.notify_all();                                   
        // notify(wake up) consumer when q.size() != 0 is true
    }
}

int main()
{
    thread consumers[2], producers[2];
    // spawn 2 consumers and 2 producers:
    for (int i = 0; i < 2; ++i)
    {
        consumers[i] = thread(consumer);
        producers[i] = thread(producer, i + 1);
    }
    // join them back: (in this program, never join...)
    for (int i = 0; i < 2; ++i)
    {
        producers[i].join();
        consumers[i].join();
    }
    system("pause");
    return 0;
}


class SolutionT41 {
public:
    int firstMissingPositive(vector<int>& nums) {
        for (int i = 0; i < nums.size(); i++) {
            while(nums[i] != i+1) {
                if(nums[i] <= 0 || nums[i] > nums.size() || nums[i] == nums[nums[i] - 1]) break; //两个相等
                int temp = nums[i] - 1;
                nums[i] = nums[temp];
                nums[temp] = temp;
            }
        }
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] != i+1) return i+1;
        }
        return nums.size() + 1;
    }
};


class SolutionT82 {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode *dummy = new ListNode(-1), *pre = dummy;
        dummy->next = head;
        ListNode *cur = head;
        while(cur) {
            if (cur->next) {
                if (cur->val == cur->next->val) {
                    while(cur && cur->next && cur->val == cur->next->val) {
                        cur = cur->next;
                    }
                    cur = cur->next;
                } else {
                    pre->next = cur;
                    cur = cur->next;
                    pre->next->next = nullptr;
                    pre = pre->next;
                }
            } else {
                pre->next = cur;
                cur = cur->next;
                pre->next->next = nullptr;
                pre = pre->next;
            }
        }
        return dummy->next;
    }
    while(pre->next) {
        ListNode* cur = pre->next;
        while(cur->next && cur->next->val == cur->val) cur = cur->next;
        if (cur != pre->next) pre->next = cur->next;
        else pre = pre->next;
    }
};

class SolutionT83 {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if(!head) return head;
        ListNode *cur = head, *nextNode = head->next;
        while(nextNode){
            while(nextNode && cur->val == nextNode->val){
                nextNode = nextNode->next;
            }
            cur->next = nextNode;
            cur = cur->next;
            if(nextNode) nextNode = nextNode->next;
        }
        return head;
    }
};

class SolutionT80 {
public:
    int removeDuplicates(vector<int>& nums) {
        int curIndex = 0, n = nums.size();
        for (int i = 0; i < nums.size(); i++) {
            int tempIndex = i, tempVal = nums[i];
            while(tempIndex < n && nums[tempIndex] == nums[i]) tempIndex++;
            if (tempIndex - i >= 2) {
                nums[curIndex++] = tempVal;
                nums[curIndex++] = tempVal;
            } else {
                nums[curIndex++] = tempVal;
            }
            i = tempIndex - 1;
        }
        return curIndex;
    }
};

class SolutionT394 {
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

class complex{
public:
    complex();
    complex(double real, double imag);
public:
    complex operator+(const complex &A);
private:
    double m_real;
    double m_img;
}
complex::complex(): m_real(0.0), m_img(0.0){}
complex::complex(double real, double imag): m_real(real), m_img(imag) {}
//运算符重载
complex complex()::operator+(const complex &A) {
    complex B;
    B.m_real = this->m_real + A.m_real;
    B.m_imag = this->m_imag + A.m_imag;
    return B;
}
