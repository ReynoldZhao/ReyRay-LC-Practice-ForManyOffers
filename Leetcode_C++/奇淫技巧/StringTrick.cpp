to_string(); 
//可以将int 转化为string

string str;
char* chr = strdup(str.c_str());

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

    TreeNode* Deserialize(char **str){//由于递归时，会不断的向后读取字符串
        if(**str == '#'){  //所以一定要用**str,
            ++(*str);         //以保证得到递归后指针str指向未被读取的字符
            return NULL;
        }
        int num = 0;
        while(**str != '\0' && **str != ','){
            num = num*10 + ((**str) - '0');
            ++(*str);
        }
        TreeNode *root = new TreeNode(num);
        if(**str == '\0')
            return root;
        else
            (*str)++;
        root->left = Deserialize(str);
        root->right = Deserialize(str);
        return root;
    }

    //dog cat cat dog这种输入
    bool wordPattern(string pattern, string str) {
        unordered_map<char, string> m;
        istringstream in(str);
        int i = 0, n = pattern.size();
        //妙啊！ 这个for循环
        for (string word; in >> word; ++i) {
            if (i >= n) continue;
            if (m.count(pattern[i])) {
                if (m[pattern[i]] != word) return false;
            } else {
                for (auto a : m) {
                    if (a.second == word) return false;
                }
                m[pattern[i]] = word;
            }
        }
        return i == n;
    }
