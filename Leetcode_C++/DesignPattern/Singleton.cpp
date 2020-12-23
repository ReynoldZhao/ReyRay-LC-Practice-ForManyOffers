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
};


//饿汉模式 static成员初始化的时候即实例化
class Singleton {
public:
    static Singleton* getInstance() {
        return p;
    }
private:
    Singleton() {}
    static Singleton* p;
};

Singleton* Singleton:: p = new Singleton();