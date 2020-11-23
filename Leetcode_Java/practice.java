import java.util.Arrays;
import java.util.HashMap;
import java.lang.Integer;
import java.lang.Character;
import java.lang.String;
import java.lang.Math;
import java.util.PriorityQueue;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Comparator;
import java.util.LinkedList;


/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode c1 = l1;
        ListNode c2 = l2;
        ListNode temp = new ListNode(0);
        ListNode result = temp;
        int sum = 0;
        while(c1!=null && c2!=null){
            sum = sum/10;
            if(c1!=null){
                sum += c1.
            }
        }
    }
}


public class Solution {
    public double findMedsianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length, left = (m + n + 1) / 2, right = (m + n + 2) / 2;
        return (findKth(nums1, 0, nums2, 0, left) + findKth(nums1, 0, nums2, 0, right)) / 2.0;
    }
    int findKth(int[] nums1, int i, int[] nums2, int j, int k) {
        if(nums1.size()<=i) return nums2[j+k];
        if(nums2.size()<=j) return nums1[i+k];
        if(k==1) return Math.min(nums1[i], nums2[j]);
        int v1 = ((i + k/2 - 1) < nums1.length) ? nums1[i+k/2-1] : Integer.MAX_VALUE;
        if(nums1[i] < nums2[j]){

        }
    }
}

class Solution {
    public String longestPalindrome(String s) {
        if(s.length() < 2) return s;
        int[][] dp = new int[s.length][s.length];
        for(int i = 0; i < s.length(); i++){
            dp[i][i] = 1;
            for(int j=0; j < s.length(); j++){
                dp[i][j] = (s[i] == s[j]) && (j-i < 2 || dp[i+1][j-1])
            }
        }
    }
}

class Solution {
    public int maxArea(int[] height) {
        int left = 0, right = height.length();
        int h = 0, width = 0, result = 0;
        while(i<j){
            result = Math.max(result, Math.min(height[i], height[j])*(j-i));
            height[i] < height[j] ? ++i:--j;
        }
        return result;
    }
}

class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new linkedList<>();
        for (int i = 0; i < nums.length - 2; i++){
            if (i==0 || (i > 0 && nums[i] != nums[i-1])){
                int left = i+1, right = nums.length - 1, result = 0 - nums[i];
                while (left < right) {
                    if (nums[left] + nums[right] == sum) {
                        res.add(Arrays.asList(num[]))
                    }
                } 
            }
        }
    }
    public List<String> letterCombinations(String digits) {
        LinkedList<String> ans = new LinkedList<String>();
        if(digits.empty()) return ans;
        String[] mapping = new String[] {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        ans.add("");
        for (int i = 0;i < digits.length() - 1;i++) {
            int x = Character.getNumericValue(digits.charAt(i));
            while (ans.peek().length() == i){
                String t = ans.remove();
                for (char s : mapping[x].toCharArray()) {
                    ans.add(t + s);
                }
            }
        }
    }
}

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode t = new ListNode(0);
        ListNode fast = head;
        int count = 0;
        while (count <= n) {
            fast = fast -> next;
            count ++;
        }
        t = head;
        while (fast != null) {
            t = t -> next;
            fast = fast -> next;
        }
        t ->next = t -> next -> next;
        return head;


        }
        
    }
}

class Solution {
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<Character>();
        for (char x:s.toCharArray){
            if (x == "(") stack.push(")");
            else if (x == "{") stack.push("}");
            else if (x == "[") stack.push("]");
            else if(s.empty() || stack.pop() != s) return false;
            
        }
        

    
    }
}

public class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;

        ListNode temp = new ListNode(0);
        ListNode cur  = new ListNode(0);
        temp = cur;
        while (l1 && l2){
            if (l1->val < l2->val) {
                cur -> next = l1;
                l1 = l1->next;
            }
            else {
                cur -> next = l2;
                l2 = l2->next;
            }
            cur = cur->next;
        }
        cur -> next = l1 ? l1:l2;
        return temp->next;
        
    }
    
    public ListNode mergeKLists(List<ListNode> lists) {
        if (lists.size()==0) return null;
        if (lists.size()==1) return lists.get(0);
        if (lists.size()==2) return mergeTwoLists(lists.get(0), lists.get(1));
        return mergeTwoLists(mergeKLists(lists.subList(0, lists.size()/2)), 
            mergeKLists(lists.subList(lists.size()/2, lists.size())));
        
        if (lists.size == 0) return null;
        int n = lists.size();
        while (n > 1){
            k = (n + 1) / 2;
            for (int i = 0;i < n / 2;i++){
                list[i] = mergeTwoLists(lists[i], lists[i + k]);
            }
            n = k;
        }
        return lists[0];
    }
}

public class Solution {
    public ListNode mergeKLists(List<ListNode> lists) {
        if (lists==null||lists.size()==0) return null;
        PriorityQueue<ListNode> queue = new PriorityQueue<ListNode>(lists.size(), new Comparator<ListNode>{
            @Override
            public int compare<ListNode l1, ListNode l2){
                if (l1.val < l2.val)
                    return -1;
                else if (l1.val == l2.val)
                    return 0;
                else 
                    return 1;
            }
        });

        ListNode dummy = new ListNode(0);
        ListNOde tail = dummy;

        for (ListNode node : lists){
            if(node != null)
                queue.add(node);
        }

        while(!queue.isEmpty()){
            tail.next = queue.poll();
            tail = tail.next;
            if (tail.next != null) queue.add(tail.next);
        }
        return dummy.next;

    }
}

class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> lists = new ArrayList<>();
        DFS();
        return lists;
    }
    void DFS(List<List<Integer>> lists, List<Integer> templist, int[] nums){
        if (templist.size() == nums.length) {
            lists.add(templist);
        }
        for (int i = 0; i < nums.length; i++) {
            if (templist.containsKey(nums[i])) continue;
            templist.add(nums[i]);
            DFS(lists, templist, nums);
            templist.remove(templist.size() - 1);
        }
    }
}

class Solution {
    public int longestValidParentheses(String s) {
        int res = 0;
        List<Integer> dp[] = new List<Integer>;
        for (int i = 1; i <= s.size(); i++){
            int j = i - 3 + 1 - dp[i - 1];
            if  (s[i-1] == '(' || j < 0 || s[j] == ')') {
                dp[i] = 0;
            }
            else {
                dp[i] = dp[i-1] + 2 + dp[j];
                res = max(res, dp[i]);
            }
        }
        return res;
    }
}

class Solution {
    public int search(int[] nums, int target) {
        int lo = 0, hi = nums.size() - 1;
        while (lo < hi){
            int mid = (lo + hi) / 2;
            if (nums[mid] < nums[hi]) lo = mid + 1;
            else hi = mid - 1;
        }
        int rot = lo;
        int start = (target < nums[0] ) ? rot : 0;
        int end = (target > nums[nums.size() - 1]) ? rot - 1 : nums.size();
        while (start < end){
            int mid = start + (end - start) / 2;
            if (target == nums[mid]) return mid;
            else if (nums[mid])
        }

    public static class Node {
        public int value;
        public Node next;
        public Node rand;

        public Node(int data) {
            this.value = data;
        }
    }

    public static Node copyListWithRand1(Node head) {
        HashMap<Node,Node> map = new HashMap<Node,Node>();
        Node cur = head;
        while (cur) {
            map.put(cur, new Node(cur.value));
            cur = cur.next;
        }
        cur = head;
        while (cur) {
            map.get(cur).next = map.get(cur.next);
            map.get(cur).rand = map.get(cur.rand);
            cur = cur.next;
        }
    }

    public TreeNode Convert(TreeNode pRootOfTree) {
        stack<int> stack = new stack<int>();
        TreeNode p = pRootOfTree;
        TreeNode dummy = new TreeNode(0);
        TreeNode pre = null;
        boolean isFirst = true;
        while (p) {
            while (p!=null || !stack.isEmpty()) {
                while (p) {
                    stack.push(p);
                    p = p.left;
                }
                p = stack.pop();
                if (isFirst) {
                    dummy.next = p;
                    pre = p;
                    isFirst = false;
                }
                else {
                    pre.right = p;
                    p.left = pre;
                    pre = p;
                }
                p = p.right;
            }

        }
    }

    public TreeNode Convert(TreeNode pRootOfTree) {
        if (!pRootOfTree) return null;
        if (!pRootOfTree.left && !pRootOfTree.right) return pRootOfTree;
        TreeNode left = Convert(pRootOfTree.left);
        TreeNode p = left;
        while (p && p.right) {
            p = p.right;
        }
        if (left!=null) {
            p.right = pRootOfTree;
            pRootOfTree.left = p;
        }
        TreeNode right = Convert(pRootOfTree.right)
    } 
    


    public ArrayList<String> Permutation(String str) {
        ArrayList<String> res = new ArrayList<String>();
        if (str!=null && str.length() > 0) {
            helper(str.toCharArray(), 0, res);
            Arrays.sort(res);
        }
        return res;
    }

    public void help(char[] chars, int i, ArrayList<String> res) {
        if (i==chars.length-1) res.add(String.valueof(chars));
        else{
            Set<Character> set = new HashSet<Character>();
            for (int j = i; j<chars.length; ++j) {
                if (i == j || !map.containsKey(chars[j])) {
                    set.add(chars[j]);
                    swap(chars, i, j);
                    help(chars, i + 1, res);
                    swap(chars, i, j);
                }
            }  
        }
    }

    public int MoreThanHalfNum_Solution(int [] array) {
        if (array.size == 0) return 0;
        if (array.size == 1) return array[0];
        int result = array[0], times = 0;
        for (int i = 0; i<array.size; ++i) {
            if (array[i] == result) times++;
            else {
                times--;
                if (times < 0) {
                    times = 0;
                    result = array[i];
                }
            }
        }
    }

    public int MoreThanHalfNum_Solution(int [] array) {
        if (arrary.length<=0) return 0;
        int start = 0, end = array.size - 1;
        int length = array.size;
        int middle = length/2;
    
        int index = pertition(arrary, start, end);
        
        while (index!=middle) {
            if (index<middle) index = pertition(arrary, index+1, end);
            else index = pertition(arrary, start, index-1);
        }
        int result = arrary[index];
    }
    public int pertition(int [] arrary, int start, int end) {
        int flag = (arrary[start] + arrary[end])/2;
        while (start<end) {
            while (arrary[end] > flag) end--;
            swap(arrary, start, end);
            while (arrary[start] <= flag) start++;
            swap(arrary, start, end);
        }
        return start;
    }
    public void swap(int [] arrary, int i, int j) {
        int temp = arrary[i];
        arrary[i] = arrary[j];
        arrary[j] = arrary[i];
    }
    public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
        ArrayList<Integer> res = new ArrayList<Integer>();
        int length = input.length;
        if (k>length || k==0) return res;
        PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(k, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2.compareTo(o1);
            }
        });
        for (int i=0;i<input.length;i++) {
            if (maxHeap.size()<k) maxHeap.offer(input[i]);
            else if (maxHeap.peek() > input[i]) {
                Integer temp = maxHeap.poll();
                temp = null;
                maxHeap.offer(input[i]);
            }
        }
        for (Integer i : maxHeap) {
            res.add(i);
        }
        return res;
    }

    public int FindGreatestSumOfSubArray(int[] array) {
        if (arrary.length == 0) return 0;
        int maxsum = 0;
        int sum = 0
        for (int i=0;i<array.length;i++) {
            if (sum<0) sum = 0;
            sum+=arrary[i];
            maxsum = max(sum, maxsum);
        }
        return maxsum;
    }

    public String PrintMinNumber(int [] numbers) {
        Arrays.sort(numbers);
        ArrayList<Integer> res = new ArrayList<Integer>();
        helper(numbers, res, 0);
    }
    public void helper(int [] numbers, ArrayList<String> res, int i) {
        if (i==numbers.length-1) {
            res.add(StringofArray(numbers));
            return;
        }
        for (int i=0;i<numbers.length;i++) {
            for (int j = i + 1; j<numbers.length;j++) {
                swap(numbers, i, j);
                helper(numbers, res, j);
                swap(numbers, j, i);
            }
        }
    }
    public String PrintMinNumber(int [] numbers) {
        String res = "";
        ArrayList<Integer> list = new ArrayList<Integer>();
        for (int i =0;i<numbers.length;i++) {
            list.add(numbers[i]);
        }
        Collections.sort(list, new Comparator<Integer>(){
            public int compare(Integer str1, Integer str2) {
                String s1 = str1+""+str2;
                String s2 = str2+""+str1;
                return s1.compareTo(s2);
            }
        }

    }

    public int FirstNotRepeatingChar(String str) {
        LinkedHashMap<Character,Integer> map = new LinkedHashMap<Character,Integer>();
        for (int i=0;i<str.length();i++){
            if(map.containsKey(str.charAt(i))) {
                int times = map.get(str.charAt(i));
                map.put(str.charAt[i], ++times);
            }
            else
        }
        return 0;
    }

    public int TreeDepth(TreeNode root) {
        if (root==NULL) return 0;
        int left = TreeDepth(root->left);
        if (left == -1) return -1;
        int right = TreeDepth(root->right);
        if (right == -1) return -1;
        if (abs(right-left)>=2) return -1;
        else return max(right, left)+1;
    }

    public void FindNumsAppearOnce(int [] array,int num1[] , int num2[]) {
        if (arrar.length==2) {
            num1[0] = arrary[0];
            num1[1] =arrary[1]; 
        }
        int bitResult = 0;
        for (int i=0;i<arrary.length;i++) {
            bitResult = bitResult^arrary[i];
        }
        int index = findFirst(bitResult);
        for (int i=0;i<arrary.length;i++) {
            if (isBit(arrary[i], index)) {
                num1[0] = num1^arrary[i];
            }
            else num2[0] = num2^arrary[i];
        }
    }
    public int findFirst(int b) {
        int index = 0;
        while(b&1==0){
            index++;
            b>>1;
        }
        return index;
    }
    public boolean isBit(int a, int index) {
        return ((a>>index)&1==1);
    }

    public ArrayList<ArrayList<Integer> > FindContinuousSequence(int sum) {
        ArrayList<ArrayList<int>> res = new ArrayList<ArrayList<int>>();
        for (int i=Math.sqrt(2*sum);i>=2;i--) {
           if (i&1==1 && sum%i==0 || (sum/i)*2==i) {
               ArrayList<int> list =new ArrayList<int>();
               for (int j=0, k = sum/i - (n-1)/2;j<i;j++) {
                    k = k+j;
                    list.add(k);
               }
           }
       }
    }

    public String ReverseSentence(String str) {
        if(str.trim().equals("")||str==null){
            return str;
        }
        String[] a = str.spilt(" ");
        StringBuffer o = new StringBuffer();
        int i;
        for (int i=a.length;i>0;i--){
            o.append(str[i]);
            if(i>0)
                o.append(" ");
        }
        return o.toString();
    }


    public boolean isNumeric(char[] str) {
        String string = String.valueOf(str);
        return string.matches("[\\+\\-]?\\d*(\\.\\d+)?([eE][\\+\\-]?\\d+)?")
    }

    public ListNode deleteDuplication(ListNode pHead)
    {   
        if(pHead==null||pHead.next==null) return null;
        ListNode temp = new ListNode(0);
        temp.next = pHead;
        ListNode pre = temp;
        ListNode cur = pHead;
        while (cur!=null) {
            if(cur.next&&cur.val == cur.next.val){
                while (cur.next!=null&&cur.val == cur.next.val) {
                    cur = cur.next;
                }
                pre.next = cur.next;
                cur = cur.next;
            }
            else {
                pre = pre.next;
                cur = cur.next;
            }
        }
        return temp.next;
    }
    public ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
        boolean flag = false;
        queue<TreeNode> q;
        stack<TreeNode) s;
        q.push(pRoot);
        ArrayList<ArrayList<Integer>> res = new ArrayList<Integer>();
        while(q.isEmpty()!=null&&s.isEmpty()!=null){
            ArrayList<Integer> temp = new ArrayList<Integer>();
            if(flag == false){
                while(q.isEmpty()!=true){
                    TreeNode t = q.peek();
                    temp.put(t.value);
                    if(t.left) s.push(t.left);
                    if(t.right) s.push(t.right);
                    q.pop();
                }
            }
            else {
                while(s.isEmpty()!=true){
                    TreeNode t = s.top();
                    temp.put(t.value);
                    if(t.left) s.push(t.left);
                    if(t.right) s.push(t.right);
                    q.pop();
                }               
            }
        }
    }

    public ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<Integer>();
        if(pRoot==null) return res;
        ArrayList<Integer> list = new ArrayList<>();
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.addLast<null>;
        queue.addLast<pRoot>;
        boolean left2right = true;
        while (queue.size()!=1) {
            TreeNode node = queue.removeFirst();
            if(node==null){
                Iterator<TreeNode> iter = null;
                if(left2right){
                    iter = queue.Iterator();
                }
                else{
                    iter = queue.descendingIterator();
                }
                left2right = !left2right;
                while(iter.hasNext()){
                    TreeNode temp = (TreeNode)iter.next();
                    list.add(temp.val);
                }
                res.add(new ArrayList<Integer>(list));
                list.clear();
                queue.addLast(null);
                continue;
            }
        }
        if (node.left != null) {
            queue.addLast(node.left);
        }
        if (node.right != null) {
            queue.addLast(node.right);
        }
    }
    ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if(pRoot==null) return res;
        ArrayList<Integer> temp = new ArrayList<>();
        Queue<Integer> queue = new LinkedList<>();
        queue.add(pRoot);
        int cnt = 0;
        while(!queue.isEmpty()){
            while(cnt<queue.size()){
                TreeNode head = queue.removeFirst();
                temp.add(head.val);
                if(temp.left!=null) queue.add(temp.left);
                if(temp.right!=null) queue.add(temp.right);
                cnt++;
            }
            res.add(new ArrayList(temp));
            cnt = 0;
            temp.clear();
        }
    }
}

public class SolutionMedian {

    PriorityQueue<Integer> Maxq = new PriorityQueue<Integer>();
    PriorityQueue<Integer> Minq = new PriorityQueue<Integer>(new Comparator<Integer>(){
        public int compare(Integer i1, Integer i2){
            return i2<i1;
        }
    });
    public void Insert(Integer num) {
        if(Maxq.isEmpty()||num<Maxq.peek()) Maxq.add(num);
        else {
            Minq.add(num);
        }
        if(Maxq.size()+2==Minq.size()) {
            Maxq.add(Minq.peek());
            Minq.poll();
        }
        if(Maxq.size()== Minq.size()+1){

        }
    }

    public Double GetMedian() {
        
    }


}