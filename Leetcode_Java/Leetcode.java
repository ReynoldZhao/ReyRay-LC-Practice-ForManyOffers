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


class Solution {

    public int divide(int dividend, int divisor) {
        if(dividend==Integer.MIN_VALUE&&divisor==-1) return Integer.MAX_VALUE;
        long int m = Math.abs(dividend), n = Math.abs(divisor);
        int mark = ((dividend<0)^(divisor<0)) ? -1:1;
        if (n==1) return (mark==1) ? m:-m;
        long int res = 0;
        while (m>=n){
            long int t = n, p = 1;
            while (m >= (t<<=1)) {
                t = t<<1;
                p = p<<1;
            }
            m = m - t;
            res = res + p;
        }
        return mark*res;
    }
    public int[] searchRange(int[] nums, int target) {
        int pos1 = findpos(nums,target,true);
        int pos2 = findpos(nums,target,false);
        int[] res = new int[2];
        if(nums.length==0||pos1==-1||pos2==-1){
            res[0] = -1;
            res[1] = -1;
        }
        else if(nums[pos1]==target&&nums[pos2]==target){
            res[0] = pos1;
            res[1] = pos2;
        }
        return res;
    }
    public int findpos(int[] nums, int target, boolean flag){
        int left = 0, right = nums.length - 1;
        int mid = (right-left)/2 + left;
        while(left<=right){
            mid = (right-left)/2 + left;
            if(nums[mid]<target){
                left = mid+1;
            }
            else if(nums[mid]>target){
                right = mid-1;
            }
            else return mid;
        }
        if(nums[mid]==target){
            if(flag){
                while(mid-1>=0&&nums[mid-1]==target){
                    mid = mid-1;
                }
            }
            else{
                while(mid+1<=nums.length-1&&nums[mid+1]==target){
                    mid = mid+1;
                }
            }
            return mid;
        }
        else return -1;
    }

    public boolean isValidSudoku(char[][] board) {
        boolean[][] rowflag = new boolean[9][9];
        boolean[][] colflag = new boolean[9][9];
        boolean[][] blockflag = new boolean[9][9];
        for(int i=0;i<9;i++){
            for(int j=0;j<9;j++){
                if(board[i][j]==".") continue;
                int c = board[i][j] - "1";
                if(rowflag[i][c]==true||colflag[c][j]==true||blockflag[3*(i/3)+j/3][c] = true) return false;
                rowflag[i][c] = true;
                colflag[c][j] = true;
                blockflag[3*(i/3)+j/3][c] = true;
            }
        }
        return true;
    }
    public String countAndSay(int n) {
        if(n==1) return "1";
        String s = countAndSay(n-1);
        StringBuilder res = new StringBuilder();
        for(int j = 0;j<s.length();j++){
            Character temp = s.charAt(j);
            int count = 1;
            while(j+1<s.length()&&s.charAt(j+1)==temp) {
                j++;
                count++;
            }
            res.append(count+""+temp);
        }
        return res.toString();
    }

    int firstMissingPositive(vector<int>& nums) {
        for (int i=0; i<nums.size(); i++){
            if(nums[i]>0&&nums[i]<nums.size()&&nums[i]!=nums[nums[i]-1]){
                swap(nums[i], nums[nums[i]-1]);
            }
        }
        for(int i=0;i<nums.size();i++){
            if(nums[i]!=i+1)
                return i+1;
        }
    }
    class Solution {
        public List<List<Integer>> permute(int[] nums) {
            List<List<Integer>> res = new List<List<Integer>();
            permutation(res, nums, 0);
        }
        public void permutation(List<List<Integer>> res, int[] nums, int k){
            if(k>=nums.length-1){
                res.add(new List<Integer>(nums));
            }
            for(int i = k;i<nums.length;++i){
                swap(nums, k, i);
                permutation(res, nums, k+1);
                swap(nums, i, k);
            }
        }
        public List<List<Integer>> permute(int[] nums) {
            List<List<Integer>> res = new ArrayList<>();
            Arrays.sort(nums);
            backtrack(es, new ArrayList<>(), 0);
            return res;
        }
        public void backtrack(List<List<Integer>> res, ArrayList<Integer> templist, int[] nums, int start){
            if(templist.size()==nums.length){
                res.add(new ArrayList<>(templist));
            }
            for(int i = start;i<nums.length;i++){
                templist.add(nums[i]);
                backtrack(res, templist, nums, start++);
                templist.remove(templist.size()-1);
            }
        }
        public void rotate(int[][] matrix) {
            int n = matrix.length;
            if(n==1) return ;
            int loop = n+1/2;
            for(int i=0;i<loop;i++){
                int startx = i, starty = i, length = n - 2*i;
                if(length<=1) break;
                int endx = i+length-1, endy = i+length-1;
                for(int j=0;j<length;j++){
                    int temp = matrix[startx][starty+j];
                    matrix[startx][starty+j] = matrix[endx-j][starty];
                    matrix[endx-j][starty] = matrix[endx][endy-j];
                    matrix[endx][endy-j] = matrix[startx+j][endy];
                    matrix[startx+j][endy] = temp;
                }
            }
        }
        public List<List<String>> groupAnagrams(String[] strs) {
            List<List<String>> res = new ArrayList<>();
            HashSet<Integer> visited = new HashSet<Integer>();
            while(visited.size()<str.length){
                String t = null;
                ArrayList<String> temp = new ArrayList<>();
                for(int i = 0;i<str.length;i++){
                    if(temp.isEmpty()&&!visited.contains(i)){
                        temp.add(strs[i]);
                        visited.add(i);
                        t = strs[i].sort();
                        continue;
                    }
                    if(visited.contains(i)) continue;
                    String s = strs[i];
                    if(s.sort()==t){
                        temp.add(strs[i]);
                        visited.add(i);
                        continue;
                    }
                }
                res.add(temp);
            }
            return res;
        }
        public int[][] merge(int[][] intervals) {
            Arrays.sort(intervals,(i1,i2)->Integer.compare(i1[0], i2[0]));
            List<int[]> res = new ArrayList<>();
            int[] newInterval = intervals[0];
            res.add(newInterval);
            for(int[] interval:intervals){
                if(interval[0]<=newInterval[1]){
                    newInterval[1] = Math.max(interval[1], newInterval[1]);
                }
                else{
                    newInterval = interval;
                    res.add(inter)
                }
            }
        }

        public int mySqrt(int x) {
            int left = 0, right = x/2;
            while(left<right){
                int mid = (right - left)/2 +left;
                if(mid*mid<x) left = mid+1;
                else if(mid*mid>x) right = mid;
                else {
                    right = mid;
                    break;
                }
            }
            return right;
        }

        public int calculate(String s) {
            if(s==null) return 0;
            Stack<Integer> number = new Stack<Integer>();
            char sign = '+';
            int num = 0;
            for(int i=0;i<s.length();i++){
                if(Character.isDigit(s.charAt(i))){
                    num = num*10 + s.charAt(i) - '0';
                }
                if((!Character.isDigit(s.charAt(i))&&' '!=s.charAt(i))||i=s.length()-1){
                    if(sign == '+'){
                        number.push(num);
                    }
                    if(sign == '-'){
                        number.push(-num);
                    }
                    if(sign == '*'){
                        number.push(number.pop()*num);
                    }
                    if(sign == '/'){
                        number.push(number.pop()/num);
                    }
                    sign = s.charAt(i);
                    num = 0;
                }
            }
            int res = 0;
            for(int n:number){
                res+=n;
            }
            return res;
        }       
    }
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }
    public void Inorder(TreeNode root, ArrayList<Integer> &arr){
        if(root==null) return;
        Inorder(root.left);
        arr.add(root.val);
        Inorder(root.right);
        return;
    }
    public int kthSmallest(TreeNode root, int k) {
        // List<Integer> arr = new ArrayList<>();
        // Inorder(root, arr);
        // return arr.get(k-1);

        // int count = countnode(root.left);
        // if(k<=count) return kthSmallest(root.left, k);
        // else(k>count+1) return kthSmallest(root.right, k - count - 1);
        // return root.val;
        int count = 0;
        TreeNode cur = root;
        Stack<TreeNode> s = new Stack<>();
        while(cur||!s.isEmpty()){
            while(cur){
                s.push(cur);
                cur = cur.left;
            }
            cur = s.top();
            s.pop();
            count++;
            if(count==k) return cur.val;
            cur = cur.right;
        }
    }
    public class MyTreeNode {
        int val;
        MyTreeNode left;
        MyTreeNode right;
        int count;
        MyTreeNode(int x) { val = x; count = 1; }
    }
    public MyTreeNode build(TreeNode root){
        if (!root) return null;
        MyTreeNode node = new MyTreeNode(root.val);
        node.left = build(root.left);
        node.right = build(root.right);
        if (node->left) node->count += node->left->count;
        if (node->right) node->count += node->right->count;
        return node;
    }
    public int helper(MyTreeNode root, k){
        if(root.left){
            if(k<=root.left.count){
                return helper(root, k);
            }
            else if(k>root.left.count+1){
                return helper(root.right, k - root.left.count - 1);
            }
            else return root.count;
        }
        else{
            if(k == 1) return root.val;
            return helper(root.right, k-1);
        }
    }
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        int[] fwd = new int[n], bwd = new int[n];
        fwd[0] = 1; bwd[n - 1] = 1;
        for (int i = 1; i < n; ++i) {
            fwd[i] = fwd[i - 1] * nums[i - 1];
        }
        for (int i = n - 2; i >= 0; --i) {
            bwd[i] = bwd[i + 1] * nums[i + 1];
        }
        for (int i = 0; i < n; ++i) {
            res[i] = fwd[i] * bwd[i];
        }
        return res;
    }
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null||root==p||root==q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(left&&right) return root;
        if(left||right){
            return left?lowestCommonAncestor(left, p, q):lowestCommonAncestor(right, p, q);
        }
    }
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null||root==p||root==q) return root;
        if(p.val<root.val&&q.val<root.val) return lowestCommonAncestor(root.left, p, q);
        if(p.val>root.val&&p.val>root.val) return lowestCommonAncestor(root.right, p, q);
        return root;
    }

    public boolean isAnagram(String s, String t) {
        if(s.isEmpty()&&t.isEmpty()) return true;
        if(s.isEmpty()||t.isEmpty()) return false;
        if(s.length()!=t.length()) return false;       
        int[] count = new int[s.size()];
        for(int i=0;i<s.length();i++){
            count[s.charAt(i) - 'a']++;
        }
        for(int j=0;j<s.length();j++){
            count[t.charAt(j)]--;
        }
        for(int i=0;i<s.length();i++){
            if(count[i]!=0) return false;
        }
        return true;
    }

    bool canAttendMeetings(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(),[](vector<int> v1, vector<int> v2){
            return v1[0]<v2[0];
        })
        Arrays.sort(intervals,(i1, i2)->Integer.compare(i1.get(0), i2.get(0));
    }
    
    public int numSquares(int n) {
        if(n%4==0) n = n/4;
        if(n%8==7) return 4;
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for(int i=0;i<=n;i++){
            for(int j=1;i+j*j<=n;j++){
                dp[i+j*j] = min(dp[i+j*j], dp[i]+1);
            }
        }
        return dp[n];

        ArrayList<Integer> dp = new ArrayList<Integer>();
        Arrays.fill(dp, 0);
        while(dp.size()<n){
            int m = dp.size(), val = Integer.MAX_VALUE;
            for(int i=1;i*i<=m;i++){
                val = Math.min(val, dp[m-i*i]+1);
            }
        }

    }

    bool knows(int a, int b);
    int findCelebrity(int n){
        int res=0;
        for(int i=0;i<n;i++){
            if(knows(res, i)) res = i;
        }
        for(int i=0;i<res;i++){
            if(!knows(i, res)||knows(res, i)) return -1;
        }
        for(int i=res+1;i<n;i++){
            if(!knows(i, res)) return -1;
        }
        return res;
    }

    public void moveZeroes(int[] nums) {
        for(int i=0;i<nums.length;i++){
            if(nums[i]==0)
        }
    }

    public int findDuplicate(int[] nums) {
        int left = 1, right = nums.length-1;
        while(left<right){
            int mid = (right - left) / 2 + left;
            int cnt = 0;
            for(int num:nums){
                if(num<=mid) cnt++;
            }
            if(cnt<=mid) left = mid+1;
            else right = mid;
        }
        return right;
        //数组快慢指针 牛皮！！！
        int slow =0, fast = 0, temp = 0;
        whlie(true){
            slow = nums[slow];
            fast = nums[nums[fast]];
            if(slow == fast) break;
        }
        while(true){
            slow = nums[slow];
            t = nums[t];
            if(slow == t) break;
        }
        return t;
    }

    //位操作 牛皮！！
    int findDuplicate(vector<int>& nums) {
        int res = 0, n = nums.size();
        for (int i = 0; i < 32; ++i) {
            int bit = (1 << i), cnt1 = 0, cnt2 = 0;
            for (int k = 0; k < n; ++k) {
                if ((k & bit) > 0) ++cnt1;
                if ((nums[k] & bit) > 0) ++cnt2;
            }
            if (cnt2 > cnt1) res += bit;
        }
        return res;
    }

    public void gameOfLife(int[][] board) {
        if(board==null||board.length==0||board[0].length==0)
        int h = board.length, w = h!=0?board[0].length:0;
        int dx = new int[]{-1, -1, -1, 0, 1, 1, 1, 0};
        int dy = new int[]{-1, 0, 1, 1, 1, 0, -1, -1};
        ArrayList<ArrayList<Integer>> died = new ArrayList<>();
        ArrayList<ArrayList<Integer>> revive = new ArrayList<>();
        for(int i=0;i<h;i++){
            for(int j=0;j<w;j++){
                int cnt = 0;
                for(int k=0;k<8;k++){
                    int x = i + dx[k], y = j + dy[k];
                    if(x>=0||x<h||y>=0||y<w&&(board[x][y]==1||board[x][y]==2)){
                        cnt++;
                    }
                }
                if(board[i][j]==1&&(cnt<2||cnt>3)) board[i][j] = 2; died.add({i, j})
                else if(board[i][j]==0&&cnt==3) board[i][j] = 3; revive.add({i, j});
            }
        }
        for(die:died){

        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                board[i][j] %= 2;
            }
        }
    }

    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        int res = 0;
        for (int i = 0; i < nums.size(); i++){
            for(int j = 0; j < i; j++){
                if(nums[j]<nums[i]){
                    dp[i] = max(dp[j], dp[j]+1);
                }
                res = Math.max(res, dp[i]);
            }
        }
        return res;
    }

    int lengthOfLIS(vector<int>& nums) {
        vector<int> dp;
        for (int i = 0; i < nums.size(); ++i) {
            int left = 0, right = dp.size();
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (dp[mid] < nums[i]) left = mid + 1;
                else right = mid;
            }
            if (right >= dp.size()) dp.push_back(nums[i]);
            else dp[right] = nums[i];
        }
        return dp.size();
    }
    //lowerbound

    class MedianFinder {

        /** initialize your data structure here. */
        public MedianFinder() {
            
        }
        
        public void addNum(int num) {
            if( maxheap.isEmpty() || num <= maxheap.peek()) maxheap.add(num);
            else if (num>=minheap.peek()) minheap.add(num)
            if (maxheap.size() > minheap.size()+1){
                int temp = maxheap.peek();
                maxheap.poll();
                minheap.add(temp);
            }
            else if (minheap.size() > maxheap.size()+1){
                int temp = minheap.peek();
                minheap.poll();
                maxheap.add(temp);
            }
        }
        
        public double findMedian() {
            return minheap.size()>maxheap.size()?minheap.peek():minheap.size()==maxheap.size()?:(double)(minheap.peek()+maxheap.peek())/2:maxheap.peek();
        }

        private:
            PriorityQueue<Integer> maxheap = new PriorityQueue<Integer>(1, new Comparator<Integer>() {
                @Override
                public int compare(Integer i1, Integer i2){
                    return i2.compareTo(i1);
                }
            });
            PriorityQueue<Integer> minheap = new PriorityQueue<Integer>();
    }

    public int coinChange(int[] coins, int amount) {
        if(amount <= 0) return 0;
        if (coins.length < 1) return -1;
        Arrays.sort(coins);
        int res = 0;
        for (int i = coins.length - 1; i >= 0; i--){
            if(amount / coins[i] == 0) continue;
            res += amount/coins[i];
            amount = amount % coins[i];
        }
        return amount==0?res:-1;
    }

    public int coinChangev2(int[] coins, int amount) {
        if(amount <= 0) return 0;
        if (coins.length < 1) return -1;
        Arrays.sort(coins);
        int res = 0;
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, 0);
        for (int i = coins[0]; i <= amount; i++){
            for (int j = 0; j < coins.length; j++) {
                if (i >= coins[j]) dp[i] = Math.min(dp[i], dp[i - coins[j]]) + 1;
            }
        }
        return dp[amount]>0?dp[amount]:-1;
    }
    public boolean isPowerOfThree(int n) {
        if (n%2==0)return false;
        if (n%10 != 1 || n%10 != 3 || n%10 != 7 || n%10 != 9) return false;
        if (n>=3 && (n%3==0 || n%9==0 || n%27==0 || n%81==0)) return true;
    }
}


 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 
class SolutionT328Odd Even Linked List {
    public ListNode oddEvenList(ListNode head) {
        
    }
}

class SolutionT72Edit distance {
    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m + 1][n + 1];

        for(int i = 0; i <= m; i++)
            dp[i][0] = i;
        for(int i = 1; i <= n; i++)
            dp[0][i] = i;

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j<= n; j++) {
                if(word1.charAt(i-1) == word2.charAt(j-1)) {
                    dp[i][j] = dp[i-1][j-1];
                }
                else {
                    dp[i][j] = Math.min(dp[i-1][j-1], Math.min(dp[i][j - 1], dp[i - 1][j])) + 1;
                }
            }
        }
        return dp[m][n];
    }
    public int helper(String word1, String word2, int i, int j, ArrayList<ArrayList<Integer>> memo) {
        if (i == word1.length()) return word2.length() - j;
        if (j == word2.length()) return word1.length() - i;
        if (memo.get(i).get(j) > 0) return memo.get(i).get(j);
        int res = 0;
        if (word1.charAt(i) == word2.charAt(j)) {
            return helper(word1, word2, i + 1, j + 1, memo);
        }
        else {
            int insertcnt = helper(word1, word2, i, j + 1, memo);
            int removecnt = helper(word1, word2, i + 1, j, memo);
            int replacecnt = helper(word1, word2, i + 1, j + 1, memo);
            res = Math.min(insertcnt, Math.min(removecnt, replacecnt)) + 1;
        }
        return memo[i][j] = res;;
    }
}

class SolutionT329 {
    public static int[] dx = {1, 0 , -1, 0};
    public static int[] dy = {0, 1, 0, -1};
    public int longestIncreasingPath(int[][] matrix) {
        if(matrix.length < 1 || matrix[0].length < 1) return 0;
        int m = matrix.length, n = matrix[0].length;
        int res = 1;
        int[][] cache = new int[m][n];
        Arrays.fill(cache, 0);
        for (int i = 0; i < m; i++){
            for (int j = 0; j <n; j++) {
                int len = DFS(matrix, i, j, cache, matrix[i][j]);
                res = Math.max(res, len);
            }
        }
        reeturn res;
    }
    public int DFS(int[][] matrix, int i, int j, int[][] cache, int pre){
        if (i<0 || i>=matrix.length || j<0 || j>=matrix[0].length || matrix[i][j] <= pre) {
            return 0;
        }
        if (cache[i][j] > 0) return cache[i][j];
        int res = 1;
        for (int k = 0; k <= 3; k++){
            int x = i + dx[k], y = j + dy[k];
            int len = 1 + DFS(matrix, x, y, cache, matrix[i][j]);
            res = Math.max(res, len);
        }
        cache[i][j] = res;
        return res;
    }
}
class SolutionT334 {
    public boolean increasingTriplet(int[] nums) {
        for (int i = 0; i < nums.length; i++) {

        }
        int slow = 0, fast = 1, last = 2;
        while(slow < fast && fast < last && last < nums.length){
            if (nums[slow] >= nums[fast]) {
                slow++;
                fast++;
                last++;
            }
            else 

        }
    }
}

class SolutionT109 {
    public TreeNode sortedListToBST(ListNode head) {
        if(head==null) return null;
        if(head.next==null) return TreeNode(head.val);
        ListNode slow, fast = head;
        while(fast.next!=null&&fast.next.next!=null){
            slow = slow.next;
            fast = fast.next;
        }
        TreeNode root = new TreeNode(slow.val);
        fast = slow.next;
        slow = null;
        root.left = sortedListToBST(head);
        root.right = sortedListToBST(fast);
        return root;
    }
}

class SolutionT347 {
    public List<Integer> topKFrequent(int[] nums, int k) {
        List<Integer> res = new ArrayList<Integer>();
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i])) {
                int cnt = map.get(nums[i]);
                map.put(nums[i], cnt);
            }
            else {
                map.put(nums[i], 1);
            }
        }
        List<Map.Entry<Integer, Integer>> list = new ArrayList<Map.Entry<Integer, Integer>>(map.entrySet()); //转换为list
        list.sort(new Comparator<Map.Entry<Integer, Integer>>() {
            @Override
            public int compare(Map.Entry<Integer, Integer> o1, Map.Entry<Integer, Integer> o2) {
                return o2.getValue().compareTo(o1.getValue());
            }
        });
        Collections.sort(list, new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                return o2.getValue().compareTo(o1.getValue());
            }
        });
        Set<E> set = map.keySet();
        Object[] arr = set.toArray();
        for(Object key:arr){
            res.add(set.get(key));
        }
        return res;
    }
}


class SolutionT378 {
    public int kthSmallest(int[][] matrix, int k) {
        PriorityQueue<Integer> maxheap = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2.compareTo(o1);
            }
        })
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                maxheap.add(matrix[i][j]);
            }
        }
        while (maxheap.size() > k) {
            maxheap.poll();
        }
        return maxheap.peek();
    }
}
//二分
public class SolutionT378_binary {
    public int kthSmallest(int[][] matrix, int k) {
        int lo = matrix[0][0], hi = matrix[matrix.length - 1][matrix[0].length - 1] + 1;//[lo, hi)
        while(lo < hi) {
            int mid = ( hi - lo ) / 2 + lo;
            int cnt = 0, j = matrix.length - 1;
            for (int i = 0; i < matirx.length; i++) {
                while (matrix[i][j] > mid && j >= 0) j--;
                cnt += (j + 1);
                //找到第一个大于当前target的位置 C++
                //cnt += upper_bound(matrix[i].begin, matrix[i].end, mid) - matrix[i].begin();
            }
            if (cnt < k) left = mid + 1;
        //cnt = search_less_equal(matrix, target);
            else hi = mid;
        }
        //最后right = left；
        return lo;
    }
//二分中加入这个ascend矩阵二分的性质
    public int search_less_equal(int[][] matrix, int target) {
        int i = matrix.length - 1, j = 0;
        int res = 0;
        while (i >= 0 && j < matrix.length) {
            if (matrix[i][j] <= target) {
                res += i + 1;//这一列以上的元素
                j++;
            }
            else {
                i--;
            }
        }
        return res;
    }
}

class SolutionT384 {

    private int[] nums;
    private Random random;

    public Solution(int[] nums) {
        this.nums = nums;
        Random random = new Random();
    }
    
    /** Resets the array to its original configuration and return it. */
    public int[] reset() {
        return nums;
    }
    
    /** Returns a random shuffling of the array. */
    public int[] shuffle() {
        int[] a = nums.clone();
        random.nextInt(j+1);
    }
    
    public void helper(int[] arr, int start) {
        if(l.size() == arr.length) return;
        for (int i = start; i < arr.length; i++) {
            swap(arr, start, i);
            helper(arr, start + 1);
            swap(arr, start, i);
        }
        return;
    }
}

class RandomizedSet {
    private HashMap<Integer, Integer> map;
    private List<Integer> list;
    public Random random;
    /** Initialize your data structure here. */
    public RandomizedSet() {
        map = new HashMap<Integer, Integer>();
        list = new ArrayList<Integer>();
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if(map.containsKey(val)) return false;
        else{
            map.put(val, list.size());
            list.add(val);
        }
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        if(map.containsKey(val)) {
            int pos = map.get(val);
            if (pos < nums.size - 1) {
                int lastval = nums.get(nums.size() - 1);
                nums.set(pos, lastval);
                map.set(pos, lastval);
            }
            nums.remove(nums.size() - 1);
            map.remove(val);
            return true;
        }
        else return false;
    }
    
    /** Get a random element from the set. */
    public int getRandom() {
        return nums.get(random.nextInt(nums.size()));
    }
}

class SolutionT395 {
    public int longestSubstring(String s, int k) {
        int res = 0;
        int n = s.length();
        for (int i = 0; i + k <= s.length(); i++) {
            int max_idx = i,  mask = 0;
            int[] digit = new int[26];
            for (int j = i; j < s.length(); j++) {
                int t = s.charAt(j) - 'a';
                digit[t]++;
                if(digit[t] >= k){
                    mask = mask & (~(1<<t));
                }else if(digit[t] < k) {
                    mask |= (1<<t);
                }
                if (mask == 0) {
                    res = Math.max(j - i + 1, res);
                    max_idx = j + 1;
                }
            }
            if(max_idx != n - 1) i = max_idx;
            else return res;
        }
        return res;
    }
}

class SolutionT395_v2 {
    public:
        int longestSubstring(String s, int k) {
            int res = 0, n = s.length();
            for (int cnt = 1; cnt <= 26; cnt++) {
                int uniquecnt = 0, start = 0, i = 0;
                int[] charcnt = new int[26];
                while (i < n - 1) {
                    bool valid = true;
                    int t = s.charAt(i) - 'a';
                    if (charcnt[t] == 0) {
                        charcnt[t]++;
                        uniquecnt++;
                    }
                    while (uniquecnt > cnt) {
                        if(--charcnt[s.charAt(start++) - 'a'] == 0) uniquecnt--;
                    }
                    for (int j = 0; j < 26; ++j) {
                        if (charCnt[j] > 0 && charCnt[j] < k) valid = false;
                    }
                    if (valid) res = Math.max(res, i - start);     
                }
            }
            return res;
        }
};

class SolutionT315 {
    public List<Integer> countSmaller(int[] nums) {
        List<Integer> t = new ArrayList<>();
        Integer[] res = new int[nums.length];
        int n = nums.length - 1;
        for (int i = n; i >= 0; i--) {
            int left = 0, right = t.size() - 1;
            while (left < right) {
                int mid = (right - left) / 2;
                if (nums[mid] < nums[i]) left = mid + 1;
                else right = mid; 
            }
            res[i] = right;
            t.add(right, nums[i]);
        }
        return Arrays.asList(res);
    }
}


public class Solution {
    class Node{
        Node left, right;
        int val, smaller, dup = 1;
        public Node(int v, int s) {
            val = v;
            smaller = s;
            left = null;
            right = null;
        }
    }
    public List<Integer> countSmaller(int[] nums) {
        Integer[] res = new Integer[nums.length];
        Node root = null;
        for (int i = nums.length - 1; i >= 0; i--) {
            res[i] = insert(root, nums[i]);
            //root = insert(root, nums[i], res, i, 0);
        }
    }
    private int insert(Node root, Integer num) {
        if (root == null) {
            root = new Node(num, 0);
            return 0;
        }
        if (num < root.val) {
            root.smaller++;
            return insert(root.left, num);
        }
        else if (num == root.val) {
            root.dup++;
            return root.smaller;
        }
        else {
            return insert(root.right, num) + root.smaller + root.dup;
        }
    }
    private Node insert(Node root, Integer num, Integer[] res, int index, int presum) {
        if (root == null) {
            node = new Node(num, 0);
            res[index] = presum;
        }
        if (num < root.val) {
            root.smaller++;
            node.left = insert(root.left, num, res, index, presum);
        }
        else if (num == root.val) {
            root.dup++;
            res[index] = presum + node.smaller;
        }
        else {
            root.right = insert(root.right, num, res, index, node.smaller + node.dup + presum);
        }
        return root;
    }
}



class SolutionT876 {
    public ListNode middleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast.next!=null && fast!=null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
}

class SolutionT1432 {
    public int maxDiff(int num) {
        char[] a = Integer.toString(num).toCharArray(), b = a.clone();
        int big_index = 0, small_index = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] != '9') {
                big_index = i;
                break;
            }
        }
        for (int i = 0; i < a.length; i++) {
            if (a[i] != '1' && a[i] != '0') {
                small_index = i;
                break;
            }
        }
        char big = a[big_index], small = a[small_index];
        for (int i = 0; i < a.length; i++) {
            a[i] = a[i] == big ? '9':a[i];
            b[i] = (b[i] == small) ? small_index==0?'1':'0':b[i];
        }
        return Integer.parseInt(String.valueOf(a)) - Integer.parseInt(String.valueOf(b));
    }
}

class SolutionT1436 {
    public String destCity(List<List<String>> paths) {
        Set<String> set= new HashSet<>();
        for (List<String> l: paths) set.add(l.get(1));
        for (List<String> l: paths) set.remove(l.get(0));
        return set.iterator().next();
    }
}

class SolutionT1443 {
    public int minTime(int n, int[][] edges, List<Boolean> hasApple) {
        Map<Integer, List<Integer>> graph = create_graph(edges);
        Map<Integer, bool> visited = new HashMap<>();
        return dfs(graph, 0, hasApple, 0, visited);
    }

    private Map<Integer, List<Integer>> create_graph(int[][] edges) {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for(int i = 0; i < edges.length; i++) {
            List<Integer> list = graph.getOrDefault(edges[i][0], new ArrayList<>());
            list.add(edges[i][1]);
            graph.put(edges[i][0], list);			
			list = graph.getOrDefault(edges[i][1], new ArrayList<>());
            list.add(edges[i][0]);
            graph.put(edges[i][1], list);
        }
        return graph;
    }

    private int dfs(Map<Integer, List<Integer>> graph, int node, List<Boolean> hasApple, int myCost, Map<Integer, Boolean> visited) {
        Boolean v = visited.getOrDefault(node, false);
        if (v) return 0;
        visited.put(v, true);

        int childrenCost = 0;
        for (int n:graph.getOrDefault(node, new ArrayList<>())) {
            childrenCost += dfs(graph, n, hasApple, 2, visited);
        }
        if (childrenCost == 0 && hasApple.get(node) == false) {
            return 0;
        }
        return childrenCost + myCost;
    }


    public int minTime(int n, int[][] edges, List<Boolean> hasApple) {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        Set<Integer> visited = new HashSet<>();     
        for (int[] edge : edges) {
            graph.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            graph.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
        }       
        return dfs(graph, 0, hasApple, 0, visited);
    }
    
    private int dfs(Map<Integer, List<Integer>> graph, int cur, List<Boolean> hasApple, int cost, Set<Integer> visited) {
        if (visited.contains(cur)) {
            return 0;
        }        
        visited.add(cur);      
        int childrenCost = 0;       
        for (int children : graph.get(cur)) {
            childrenCost += dfs(graph, children, hasApple, 2, visited);
        }       
        if (childrenCost == 0 && hasApple.get(cur) == false) {
            return 0; 
        }
        return cost + childrenCost;
    }
}

public class SolutionT373 {
    public List<int[]> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        PriorityQueue<int[]> heap = new PriorityQueue<>((a,b)->a[0]+a[1] - b[0] - b[1]);
        List<int[]> res = new ArrayList<>();
        if(nums1.length==0 || nums2.length==0 || k==0) return res;
        for(int i=0; i<nums1.length && i<k; i++) que.offer(new int[]{nums1[i], nums2[0], 0});
        while(k-- && !heap.isEmpty()) {
            int[] cur = heap.poll();
            res.add(new int[]{cur[0], cur[1]});
            if(cur[2] == nums2.length -1) continue;
            que.offer(new int[]{cur[0],nums2[cur[2]+1], cur[2]+1});

        }
    }
}

class SolutionT719 {
    private int upper_bound(int[] a, int low, int high, int key){
        if (a[high] <= key) return high + 1;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if(a[mid] <= key) low = mid + 1;
            else high = mid;
        }
        return high;
    }

    private int countPairs(int[] a, int mid) {
        int n = a.length, res = 0;
        for (int i = 0; i < n; i++) {
            res += upper_bound(a, i, n - 1, a[i] + mid) - i - 1;
        }
        return res;
    }

    public int smallestDistancePair(int a[], int k) {
        int n = a.length;
        Arrays.sort(a);
        int low = a[1] - a[0];
        for (int i = 1; i < n-1; i++) {
            low = Math.min(low, a[i+1] - a[i]);
        }
        int high = a[n-1] - a[0];
        while (low < high) {
            int mid = low + (high - low) / 2;
            int cnt = countPairs(a, mid);
            if (cnt < k) low = mid + 1;
            else high = mid;
        }
        return high;
    }
}

class SolutionT1439 {
    public int kthSmallest(int[][] mat, int k) {
        final int R = mat.length;
        final int C = mat[0].length;
    
        // max priority queue for the first row
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> b - a);
        for (int c = 0; c < C; c++) {
            pq.add(mat[0][c]);
            // keep pq size less than or equal to k
            if (pq.size() > k) {
                pq.poll();
            }
        }

        for (int r = 1; r < R; r++) {
            PriorityQueue<Integer> nextPq = new PriorityQueue<>((a, b) -> b - a);
            for (int i : pq) {
                for (int c = 0; c < C; c++) {
                    nextPq.add(i + mat[r][c]);
                    if (nextPq.size() > k) {
                    nextPq.poll();
                    }
                }
            }
            pq = nextPq;
        }
        return pq.poll();
    }
}
