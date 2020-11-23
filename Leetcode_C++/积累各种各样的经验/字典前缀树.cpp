class TrieNode { 
public:
     TrieNode *child[26];
     bool isWord;
     TrieNode(): isWord(false){
         for(auto &node:child) node = nullptr;
     }

 };

class Trie {
public:
    /** Initialize your data structure here. */
    Trie() {
        root = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        int length = word.size();
        TrieNode* cur = root;
        int index = 0;
        for(int i=0;i<length;i++){
            index = word[i] - 'a';
            if(cur->child[index]==NULL){
                cur->child[index] = new TrieNode();
            }
            cur = cur->child[index];
        }
        cur->isWord = true;
    }
    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        TrieNode* cur = root;
        int index = 0;
        for(auto a:word){
            index = a-'a';
            if(cur->child[index]==NULL) return false;
            cur = cur->child[index];
        }
        return cur->isWord;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        TrieNode *p = root;
        for (auto &a : prefix) {
            int i = a - 'a';
            if (!p->child[i]) return false;
            p = p->child[i];
        }
        return true;
    }
private:
    TrieNode* root;
};