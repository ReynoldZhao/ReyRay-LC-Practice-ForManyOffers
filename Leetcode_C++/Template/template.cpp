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

template<typename T>
void mySwap(T &a, T &b) {
    
}
class Person{

}

template<> bool myCompare(Person &p1, Person &p2);

int main(){
    int a = 10, b = 10;
    mySwap<int>(a, b);
}

template<class NameType, class AgeType>
class Person{
public:
    Person(NameType name, AgeType age) {
        this->m_Age = age;
        this->m_Name = name;
    }
    NameType m_Name;
    AgeType m_Age;
}

Person<string, int> p1("name", 999);