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

enum PRODUCTTYPE {A, B};

class soapBase{
public:
    virtual ~soapBase(){};
    virtual void show()=0;
};

class soapA: public soapBase{
public:
    void show() {
        cout << "A soap" << endl;
    }
};

class soapB: public soapBase{
public:
    void show() {
        cout << "A soap" << endl;
    }
};

class Factory {
public:
    soapBase *creatSoap(PRODUCTTYPE type) {
        switch(type)
        {
            case "A":
                return new soapA();
                break;
            case B:
                return new soapB();
                break;
            default: break;
        }
    }
};

// 基类
class RoleOperation
{
public:
    virtual std::string Op() = 0; // 纯虚函数
    virtual ~RoleOperation() {} // 虚析构函数
};

// 系统管理员(有 A 操作权限)
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

class RoleFactory {
public:
    static RoleFactory& Instance() {
        static RoleFactory  instance;
        return instance;
    }

    void RegisterRole(const string& name, RoleOperation* registrar) {
        m_RoleRegistry[name] = registrar;
    }

    RoleOperation* GetRole(const string& name) {
        map<string, RoleOperation*>::iterator it;
        it = m_RoleRegistry.find(name);
        if (it != m_RoleRegistry.end()) {
            return it->second;
        }
        return nullptr;
    }
private:
    RoleFactory() {}
    ~RoleFactory() {}

    RoleFactory(const RoleFactory&);
    const RoleFactory &operator=(const RoleFactory &);

    map<string, RoleOperation*> m_RoleRegistry;
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
