import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.lang.Integer;
import java.lang.Character;
import java.lang.String;
import java.lang.reflect.Array;
import java.lang.Math;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import javafx.print.Collation;
import javafx.util.Pair;
import javax.swing.tree.TreeNode;
import sun.misc.Queue;
import sun.swing.SwingAccessor;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode() {}
    TreeNode(int val) { this.val = val; }
    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
}

class SolutionT96 {
    public int numTrees(int n) {
        int[] dp = new int[n+1];
        dp[0] = 1, dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j <= i-1; j++) dp[i] += dp[j] * dp[i - 1 - j]
        }
        return dp[n];
    }
}

class SolutionT95 {
    public List<TreeNode> generateTrees(int n) {
        List<TreeNode> res = new ArrayList();
        if (n == 0) return res;
        return helper(1, n);
    }

// 递归
    public List<TreeNode> helper(int start, int end) {
        List<TreeNode> res = new ArrayList();
        if (start > end) {
            res.add(null);
            return res;
        };
        for (int i = start; i <= end; i++) {
            List<TreeNode> left = helper(start, i-1);
            List<TreeNode> right = helper(i+1, end);
            for (TreeNode a : left) {
                for (TreeNode b : right) {
                    TreeNode temp = new TreeNode(i);
                    temp.left = a;
                    temp.right = b;
                    res.add(temp);
                }
            }
        }
        return res;
    }

// 记忆化递归

    public List<TreeNode> generateTrees(int n) {
        List<TreeNode> res = new ArrayList();
        if (n == 0) return res;
        List<List<List<TreeNode>>> memo= new ArrayList<List<List<TreeNode>>>;
        return helper(1, n, memo);
    }

    public List<TreeNode> helper(int start, int end) {
        List<TreeNode> res = new ArrayList();
        if (start > end) {
            res.add(null);
            return res;
        };
        for (int i = start; i <= end; i++) {
            List<TreeNode> left = helper(start, i-1);
            List<TreeNode> right = helper(i+1, end);
            for (TreeNode a : left) {
                for (TreeNode b : right) {
                    TreeNode temp = new TreeNode(i);
                    temp.left = a;
                    temp.right = b;
                    res.add(temp);
                }
            }
        }
        return res;
    }
}

class SolutionT98 {
    public boolean isValidBST(TreeNode root) {
        return valid(TreeNode root, Long.MIN_VALUE, Long.MAX_VALUE;
    }

    public boolean valid(TreeNode root, long mn, long mx) {
        if (root == null) return true;
        if (root.val <= mn || root.val >= mx ) return false;
        return valid(root->left, mn, root.val) && valid(root->right, root.val, mx);
    }

    public boolean isValidBST(TreeNode root) {
        TreeNode pre = null;
        return inorder(root, pre);
    }
    
    public boolean inorder(TreeNode node, TreeNode pre) {
        if (node == null) return true;
        boolean res = inorder(node.left, pre);
        if (!res) return false;
        if (pre) {
            if (node.val > pre.val) return false;
        }
        else {
            pre = node;
            return inorder(node.right, pre);
        }
    } 

    public boolean isValidBST(TreeNode root) {
        Stack<TreeNode> s = new Stack<TreeNode>;
        TreeNode p = root, pre = null;
        while (p != null || !s.empty()) {
            if (p) {
                s.push(p);
                p = p.left;
            }
            else {
                p = s.top();
                s.pop();
                if (pre != null && p.val <= pre. val) return false;
                pre = p;
                p = p.right; 
            }
        }
        return true;
    }

    public boolean isValidBST(TreeNode root) {
        if (root == null) return true;
        TreeNode cur = root;
        TreeNode pre = null;
        while ( cur ) {
            if (cur.left != null) {
                pre = cur.left;
                while (pre.right != null && pre.right != cur) pre = pre.right;
                if (pre.right == null) {
                    pre.right = cur;
                    cur = cur.left;
                }
                else {
                    pre.right = null;
                    if (cur.val <= pre.val) return false;
                    pre = cur;
                    cur = cur.right;
                }
            }
            else {
                if (pre != null && cur.val <= pre.val) return false;
                pre = cur;
                cur = cur.right;
            }
        }
    }
}

// 恢复二叉搜索树
class SolutionT99 {
    public void recoverTree(TreeNode root) {
        TreeNode p = root, pre = null, first = null, second = null;
        Stack<TreeNode> st = new Stack<TreeNode>();
        while (p != null || !st.empty()) {
            while (p!=null) {
                st.push(p);
                p = p.left;
            }
            p = st.peek();
            st.pop();
            if (pre != null && pre.val > p.val) {
                if( first == null ) first = pre;
                second = p;
            }
            pre = p;
            p = p.right;
        }
        int temp = first.val;
        first.val = second.val;
        second.val = temp;
    }

    // Morris
    public void recoverTree(TreeNode root) {
        TreeNode cur = root, pre = null, first = null, second = null;
        while (cur != null ） {
            if (cur.left != null) {
                TreeNode p = cur.left;
                while (p.right != null && p.right != cur) p = p.right;
                if (p.right == null) {
                    p.right = cur;
                    cur = cur.left;
                    continue;
                }
                else {
                    p.right = null;
                }
            }
            if (pre!= null && pre.val > cur.val) {
                if (first == null) first = pre;
                second = cur;
            }
            pre = cur;
            cur = cur.right;
        }
        swap
    }
}

// 相同的树
class SolutionT100 {
    // 前序
    public boolean isSameTree(TreeNode p, TreeNode q) {
        Stack<TreeNode> st = new Stack<TreeNode>();
        st.push(p);
        st.push(q);
        while (st.empty() != null) {
            p = st.top();
            st.pop();
            q = st.top();
            st.pop();
            if (p == null && q== null) return true; 
            if ((p!=null && q==null) || (p==null && q!=null) || p.val != q.val) return false;
            st.push(p.right); st.push(q.right);
            st.push(p.left); st.push(q.left);
        }
    }

    // 中序
    public boolean isSameTree(TreeNode p, TreeNode q) {
        Stack<TreeNode> st = new Stack<TreeNode>();
        while (p!=null || q!=null || st.empty() != null) {
            while (p!== null || q!=null) {
                if ((p!=null && q==null) || (p==null && q!=null) || p.val != q.val) return false;
                st.push(p);
                st.push(q);
                p = p.left;
                q = q.left;
            }
            p = st.peek(); st.pop();
            q = st.peek(); st.pop();
            p = p.right;
            q= = q.right;
        }
    }
}

// 对称二叉树
class SolutionT101 {
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return isEqual(root.left, root.right);
    }

    public boolean isEqual(TreeNode p, TreeNode q) {
        if ((p!=null && q==null) || (p==null && q!=null) || p.val != q.val) return false;
        return isEqual(p.left, q.right) && isEqual(p.right, q.left);
    }

    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        Queue<TreeNode> q1 = new Queue<TreeNode>();
        q1.push(root.left);
        Queue<TreeNode> q2 = new Queue<TreeNode>();
        q2.push(root.right);
        while (q1.empty() != null && q2.empty() != null) {
            p = q1.peek(); q1.pop();
            q = q2.peek(); q2.pop()；
            if ((p!=null && q==null) || (p==null && q!=null) || p.val != q.val) return false;
            if (p==null && q==null) continue;
            q1.push(p.left); q1.push(p.right);
            q2.push(q.right); q2.push(q.left);
        }
        return true;
    }
}

class SolutionT102 {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if (root==null) return res;
        Queue<TreeNode> q = new Queue<TreeNode>();
        q.push(root);
        while (q.empty() != null) {
            int size = q.size();
            List<Integer> subList = new LinkedList<Integer>();
            for (int i = 0; i < size; i++) {
                TreeNode temp = q.peek();
                subList.add(temp.val);
                q.pop();
                if (temp.left != null) q.push(temp.left);
                if (temp.right != null) q.push(temp.right);
            }
            res.add(subList);
        }
        return res;
    }
}

// 124. 二叉树中的最大路径和
class SolutionT124 {
    private int maxSum;

    public int maxSumHelper(TreeNode node) {
        if (node == null) return 0;
        int left = maxSumHelper(node->left);
        int right = maxSumHelper(node->right);
        int maxBranch = Math.max(left, right);
        maxSum = Math.max(maxSum, Math.Max(node.val, node.val + left + right));
        return Math.max(node.val, node.val + maxBranch);

    }

    public int maxPathSum(TreeNode root) {
        maxSum = Integer.MIN_VALUE;
        maxSumHelper(root);
        return maxSum;
    }
}

public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root) return "#";
        return root.val + "," + serialize(root.left) + "," + serialize(root.right);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Queue<String> queue = new LinkedList<>(Arrays.asList(data.split(",")));
        return helper(queue);
    }

    public TreeNode helper(Queue<String> queue) {
        String s = queue.poll();
        if (s.equals("#")) return null;
        TreeNode root = new TreeNode(Integer.valueOf(s));
        root.left = helper(queue);
        root.right = helper(queue);
        return root;
    }
}

public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        return serial(new StringBuilder(), root);
    }

    private StringBuilder serial(StringBuilder str, TreeNode root) {
        if (root == null) return "#";
        str.append(root.val).append(',');
        serial(str, root->left).append(',');
        serial(str, root->right);
        return str;
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        return deserial(new LinkedList(Arrays.asList(data.split(','))));
    }

    private TreeNode deserial(Queue<String> q) {
        String val = q.poll();
        if (val.equals('#')) return null;
        TreeNode root = new TreeNode(Integer.valueOf(val));
        root.left = deserial(q);
        root.right = deserial(q);
        return root;
    }
}


class SolutionT337 {
    public int rob(TreeNode root) {
        return Math.max(robInclude(root), robExclude(root));
    }

    public int robInclude(TreeNode node) {
        if (node == null) return 0;
        return node.val + robExclude(node.left) + robExclude(node.right);
    }

    public int robExclude(TreeNode node) {
        if (node == null) return 0;
        return rob(node.left) + rob(node.right);
    }

    public int rob(TreeNode root) {
        return robSub(root, new HashMap<>());
    }

    public int robSub(TreeNode root, Map<TreeNode, Integer> map) {
        if (root == null) return 0;
        if (map.containKeys(root)) return map.get(root);
        int val = 0
        if (root.left != null) {
        val += robSub(root.left.left, map) + robSub(root.left.right, map);
        }
        
        if (root.right != null) {
            val += robSub(root.right.left, map) + robSub(root.right.right, map);
        }
        
        val = Math.max(val + root.val, robSub(root.left, map) + robSub(root.right, map));
        map.put(root, val);
        
        return val;

    }

    public int rob(TreeNode root) {
        int[] res = robSub(root);
        return Math.max(res[0], res[1]);
    }

    public int[] robSub(TreeNode root) {
        if (!root) return new int[2];
        int[] left = robSub(root.left);
        int[] right = robSub(root.right);
        int[] res = new int[2];
        res[0] = root.val + left[0] + right[0];
        res[1] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        return res;
    }
}

class Solution {
    int pathSum(TreeNode* root, int sum) {
        int res = 0;
        vector<TreeNode*> out;
        helper(root, sum, 0, out, res);
        return res;
    }
    void helper(TreeNode* node, int sum, int curSum, vector<TreeNode*>& out, int& res) {
        if (!node) return;
        curSum += node->val;
        out.push_back(node);
        if (curSum == sum) ++res;
        int t = curSum;
        for (int i = 0; i < out.size() - 1; ++i) {
            t -= out[i]->val;
            if (t == sum) ++res;
        }
        helper(node->left, sum, curSum, out, res);
        helper(node->right, sum, curSum, out, res);
        out.pop_back();
    }
}

class Solution {
    public int pathSum(TreeNode root, int sum) {
        if(root == null)
            return 0;
        return findPath(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
    }
    
    public int findPath(TreeNode root, int sum){
        int res = 0;
        if(root == null)
            return res;
        if(sum == root.val)
            res++;
        res += findPath(root.left, sum - root.val);
        res += findPath(root.right, sum - root.val);
        return res;
    }

    public int pathSum(TreeNode root, int sum) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);  //Default sum = 0 has one count
        return backtrack(root, 0, sum, map); 
    }

    public int backtrack(TreeNode root, int curSum, int sum, Map<Integer, Integer> map) {
        if (root == null) return 0;
        curSum = curSum + root.val;
        int res = map.getOrDefault(curSum - sum, 0);
        map.put(curSum, map.getOrDefault(curSum, 0) + 1);
        res += backtrack(root.left, sum, target, map) + backtrack(root.right, sum, target, map);
        map.put(curSum, map.get(curSum) - 1);
        return res;
    }

}

class SolutionT429 {
    public List<List<Integer>> levelOrder(Node root) {
        Queue<TreeNode> q = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        q.add(root);
        while (!q.isEmpty()) {
            int size = q.size();
            List<Integer> temp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode p = q.peek();
                temp.add(p.val);
                q.addAll(q.poll().children);
            }
        }
        res.add(temp);
    }
    return res;
}

public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder str = new StringBuilder();
        Queue<TreeNode> q = new LinkedList<>();

    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Queue<String> q = new LinkedList<>(Arrays.asList(data.split(",")));
        String val = q.poll();
        if (val.equals("#")) return null;
        TreeNode root = new TreeNode(Integer.valueOf())；
        
    }
}

// 501. 二叉搜索树中的众数
class SolutionT501 {
    Integer pre = null;
    int count = 1;
    int max = 0;
    public int[] findMode(TreeNode root) {
        if (root == null) return new int[0];
        List<Integer> list = new ArrayList<>();
        traverse(root, list);
        int[] res = new int[list.size()];
        for (int i = 0; i < list.size(); i++) res[i] = list.get(i);
        return res;
    }

    public void traverse(TreeNode node, List<Integer> list) {
        if (node == null) return ;
        traverse(node.left, list);
        if (pre!=null) {
            if (node.val == pre) count++;
            else count = 1;
        }
        if (count == max) {
            list.add(node.val);
        }
        else if (count > max) {
            max = count;
            list.clear();
            list.add(node.val);
        }
        pre = node.val;
        traverse(node.right, list);
    }
}

class SolutionT450 {
    public TreeNode deleteNode(TreeNode root, int key) {
        TreeNode p = root, pre = null;
        while (p != null) {
            if (p.val == key) break;
            pre = p;
            if (key < p.val) p = p.left;
            else p = p.right;
        }
        if (pre == null) return del(p);
        if (key < pre.val) pre.left = del(p);
        else if (key > pre.val) del(pre.right);
        return root;
    }

    public TreeNode del(TreeNode node) {
        if (node == null) return null;
        if (node.right == null) return node.left;
        TreeNode p = node.left;
        while (p.left != null) p = p.left;
        p.left = node.left;
        return node.right;
    }

    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) return null;
        if (key < root.val) {
            root.left = deleteNode(root.left, key);
        }
        else if (key > root.val) {
            root.right = deleteNode(root.right, key);
        }
        else {
            if (root.left == null || root.right == null) {
                return root.left == null ? root.left : root.right;
            } 
            TreeNode p = root.right, pre = root;
            while (p.left != null) {
                pre = p;
                p = p.left;
            }
            root.val = p.val;
            pre.left = null;
        }
        return root;
    }
}

class CQueue {
    LinkedList<Integer> stack1;
	LinkedList<Integer> stack2;

	public CQueue() {
		stack1 = new LinkedList<>();
		stack2 = new LinkedList<>();
	}

	public void appendTail(int value) {
		stack1.add(value);
	}

	public int deleteHead() {
		if (stack2.isEmpty()) {
			if (stack1.isEmpty()) return -1;
			while (!stack1.isEmpty()) {
				stack2.add(stack1.pop());
			}
			return stack2.pop();
		} else return stack2.pop();
	}
}

class SolutionT38{
    public ArrayList<String> Permutation(String str){
 
        ArrayList<String> list = new ArrayList<String>();
        if(str!=null && str.length()>0){
            PermutationHelper(str.toCharArray(),0,list);
            Collections.sort(list);
        }
        return list;
    }
    private void PermutationHelper(char[] chars,int i,ArrayList<String> list){
        if(i == chars.length-1){
            list.add(String.valueOf(chars));
        }else{
            Set<Character> charSet = new HashSet<Character>();
            for(int j=i;j<chars.length;++j){
                if(j==i || !charSet.contains(chars[j])){
                    charSet.add(chars[j]);
                    swap(chars,i,j);
                    PermutationHelper(chars,i+1,list);
                    swap(chars,j,i);
                }
            }
        }
    }
    
    private void swap(char[] cs,int i,int j){
        char temp = cs[i];
        cs[i] = cs[j];
        cs[j] = temp;
    }
}
