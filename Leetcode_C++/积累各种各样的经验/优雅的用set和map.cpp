    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> wordSet(wordList.begin(), wordList.end());
        if (!wordSet.count(endWord)) return 0;
        unordered_map<string, int> pathCnt{{{beginWord, 1}}};
        queue<string> q{{beginWord}};
        while (!q.empty()) {
            string word = q.front(); q.pop();
            for (int i = 0; i < word.size(); ++i) {
                string newWord = word;
                for (char ch = 'a'; ch <= 'z'; ++ch) {
                    newWord[i] = ch;
                    if (wordSet.count(newWord) && newWord == endWord) return pathCnt[word] + 1;
                    if (wordSet.count(newWord) && !pathCnt.count(newWord)) {
                        q.push(newWord);
                        pathCnt[newWord] = pathCnt[word] + 1;
                    }   
                }
            }
        }
        return 0;
    }

    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> wordSet(wordList.begin(), wordList.end());
        if (!wordSet.count(endWord)) return 0;
        queue<string> q{{beginWord}};
        int res = 0;
        while (!q.empty()) {
            for (int k = q.size(); k > 0; --k) {
                string word = q.front(); q.pop();
                if (word == endWord) return res + 1;
                for (int i = 0; i < word.size(); ++i) {
                    string newWord = word;
                    for (char ch = 'a'; ch <= 'z'; ++ch) {
                        newWord[i] = ch;
                        if (wordSet.count(newWord) && newWord != word) {
                            q.push(newWord);
                            wordSet.erase(newWord);
                        }   
                    }
                }
            }
            ++res;
        }
        return 0;
    }

class Solution {
public:
  string destCity(vector<vector<string>>& paths) {
        unordered_map<string, int> degreeMap;        
        for(auto& e: paths){
            degreeMap[e[0]] += 1;
            degreeMap[e[1]] += 0;
        }
        
        for (auto& [k, v]: degreeMap)
            if (v == 0)
                return k;
        return ""; // Note1:
    }
};

//有序升列multiset，迭代器
class SolutionT1438 {
public:
    int longestSubarray(vector<int>& A, int limit) {
        multiset<int> set;
        int i = 0, j = 0;
        for (j = 0; j < A.size(); j++) {
            set.insert(A[j]);
            if(*set.rbegin() - *set.begin() > limit) {
                set.remove(set.find(A[i]));
            }
        }
    }
};