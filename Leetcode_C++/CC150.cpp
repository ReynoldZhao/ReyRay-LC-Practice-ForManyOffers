/***
    如果我是面试官，我会希望看到什么？

    可能有点偏题，毕竟这里主要就是做题的地方。。

    如果我是面试官，会考虑主要考察什么，就我的工作经验看，大多数主要是招聘工程师的，面试者如果什么问题都没有，直接写个二重循环搞定，会首先给个50分，如
    果能写点判断字符串是否为null的，60分。

    直接上手什么bitset，什么位运算的，我会先问他，题目中有没有交代字符串的字符一定是26个英文字母？如果是unicode环境，你是不是要准备2^16/8个字节的空间？
    在实际项目中，风险可控，结果可期更重要，绝大多数时候不在乎那点时间和资源。

    所以我期望面试者不要急于解答，我希望他先问我问题：

    字符串的字符范围，如果我告诉他，26个小写英文字母，那可能一开头直接判断如果字符长度>26, 直接返回False，做到这一点的，80分
    如果我告诉他ascii字符集，然后他的代码里有边界检查，并且针对不同的范围有不同的侧重点，比如说ascii字符集，那也就是128个可能性，16个字节的位运算比较好
    如果我告诉他是unicode，没有字符范围，老老实实排序再判断是比较符合我对工程师的要求的，因为算法性能稳定，没有额外资源要求，一眼看出没什么不可预见的风险，100分。
    就是说，有些东西，没想到或者一时没想到根本不是问题，日常工作中稍微提示一下即可，但是缜密的思维对于程序员来说更重要。
***/

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

class SolutionT0103
{
public:
    string replaceSpaces(string Str, int length)
    {
        int temp_len = length;
        string S = Str;
        for (int i = 0; i < length; i++)
        {
            if (S[i] == ' ')
                temp_len += 2;
        }
        int pre_end = length - 1, cur_end = temp_len - 1;
        if (pre_end < 0)
            return Str;
        for (int k = pre_end; k >= 0; k--)
        {
            if (S[k] == ' ')
            {
                S[cur_end--] = '0';
                S[cur_end--] = '2';
                S[cur_end--] = '%';
            }
            else
            {
                S[cur_end--] = S[k];
            }
        }
        return S;
    }
};

class Solution0104
{
public:
    bool canPermutePalindrome(const string &s)
    {
        bitset<128> flags;
        for (auto ch : s)
        {
            flags.flip(ch);
        }
        return flags.count() < 2; //出现奇数次的字符少于2个
    }
};

class SolutionT0105
{
public:
    bool oneEditAway(string first, string second)
    {
        if (abs(first.size() - second.size()) >= 2) return false;
        int len1 = first.size(), len2 = second.size();
        string s1 = first.size() >= second.size() ? first:second;
        string s2 = first.size() >= second.size() ? second:first;
        int a = 0, b = 0, dif = 0;
        while (a < len1 && b < len2) {
            if (s1[a] != s2[b]) {
                dif++;
                if (s1.size() == s2.size()) {
                    a++; b++;
                }
                else {
                    a++;
                }
                a++; b++;
                if (dif >= 2) break
            }
        }
        return !(dif>=2);
    }
};

//翻转数组
//对于矩阵中第 ii 行的第 jj 个元素，在旋转后，它出现在倒数第 ii 列的第 jj 个位置。
//由于矩阵中的行列从 00 开始计数，
//matrix[row][col] 翻转之后在matrix[col][n-row-1]

class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        // C++ 这里的 = 拷贝是值拷贝，会得到一个新的数组
        auto matrix_new = matrix;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                matrix_new[j][n - i - 1] = matrix[i][j];
            }
        }
        // 这里也是值拷贝
        matrix = matrix_new;
    }

    void rotate(vector<vector<int>>& matrix) {
        int m = matrix.size();
        for (int i = 0; i <= (m-1)/2; i++) {
            for (int j = i; j <= m - i - 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[m - 1 -j][i];
                matrix[m-1-j][i] = matrix[m-1-i][m-1-j];
                matrix[m-1-i][m-1-j] = matrix[j][m-1-i];
                matrix[j][n - 1 - i] = temp; 
            }
        }
    }
//首先以从对角线为轴翻转，然后再以x轴中线上下翻转即可得到结果，如下图所示(其中蓝色数字表示翻转轴)：
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for (int i = 0; i < n - 1; ++i) {
            for (int j = 0; j < n - i; ++j) {
                swap(matrix[i][j], matrix[n - 1- j][n - 1 - i]);
            }
        }
        reverse(matrix.begin(), matrix.end());
    }
};

//在 C++ 中，res += s 和 res = res + s 的含义是不一样的。
//前者是直接在 res 后面添加字符串；后者是用一个临时对象计算 res + s，会消耗很多时间和内存

//在 Java 中，要使用 StringBuilder，而不能直接用字符串相加。
//字符串压缩 可以用个双指针

class SolutionT0203 {
public:
    void deleteNode(ListNode* node) {
        node->val = node->next->val;
        node->next = node->next->next;
        //杀不掉你就变成你
    }
};

class SolutionT0109 {
public:
    bool isFlipedString(string s1, string s2) {
        return s1.size() == s2.size() && (s2 + s2).find(s1) != string::npos;
    }
    //java
    // if (s1.length()!=s2.length()) return false;
	//     String ss = s2+s2;
	//     return ss.contains(s1);    
    // }
};

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class SolutionT0201 {
public:
    ListNode* removeDuplicateNodes(ListNode* head) {
        ListNode* res = new ListNode(0);
        ListNode* curhead = head;
        res->next = head;
        while(curhead) {
            ListNode *dumhead = curhead->next, *pre = curhead;
            while(dumhead) {
                if (curhead->val == dumhead->val) {
                    pre->next = pre->next->next;
                    dumhead = pre->next;
                } else {
                    dumhead = dumhead->next;
                    pre = pre->next;
                }
            }
            curhead = curhead->next;
        }
        return res->next;
    }
};

class SolutionT1715 {
    //也可以用字典前缀树做
public:
    string longestWord(vector<string>& words) {
        unordered_set<string> allWords(words.begin(), words.end());
        string ans;
        for (auto word:allWords) {
            auto tempCollects = allWords;
            tempCollects.erase(word);
            if (isCombined(word, tempCollects)) {
                if (word.size() > ans.size())ans = word;
                if (word.size() == ans.size())ans = min(ans, word);
            }
        }
        return res;
    }

private:
    bool isCombined(string word, unordered_set<string> &words) {
        if (word.size() == 0) return true;
        for (int i = 1; i <= word.size(); i++) {
            if (words.count(word.substr(0, i)) && isCombined(word.substr(i), words)) return true;
        }
        return false;
    }
};