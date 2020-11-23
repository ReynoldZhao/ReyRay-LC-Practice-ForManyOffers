import java.util.Arrays;
import java.util.HashMap;
import java.lang.Integer;
import java.lang.Character;
import java.lang.String;
import java.lang.reflect.Array;
import java.lang.Math;
import java.util.PriorityQueue;
import java.util.Stack;
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

class Solution_T55Jump_Game {
    public boolean canJump(int[] nums) {
        if(nums.length<1) return false;
        int[] reach = new int[nums.length];
        int max_reach = nums[0];
        int i = 0;
        while (i <= max_reach){
            reach [i] = i + nums[i];
            max_reach = Math.max(max_reach, reach[i]);
            if(max_reach >= nums.length - 1) return true;
            i++;
        }
        return false;
    }
}

class Solution_T45Jump_GameII {
    public int jump(int[] nums) {
        if(nums.length<1) return 0;
        int[] reach = new int[nums.length];
        int max_reach = 0;
        int step = 0;
        int i = 0;
        while (max_reach < nums.length - 1){
            for (; i <= max_reach; i++) {
                reach[i] = i + nums[i];
                if (reach[j] >= max_reach) {
                    max_reach = reach[j];
                    next_i = j;
                }
            }
            step++;
        }
        return step;
    }
}

class Solution63_Unique_Path {
    public int uniquePaths(int m, int n) {
        int[][] matrix = new int[m][n];
        for (int i = 0; i < m; i++) {
            matrix[i] = 1;
        }
        for (int j = 0; j < n; j++){
            matrix[j] = 1;
        }
        for(int i = 1; i < m; i++){
            for (int j = 1; j < n; j++){
                matrix[i][j] = matrix[i-1][j] + matrix[i][j-1];
            }
        }
        return matrix[m - 1][n - 1];
    }

    class Solution {
        public int uniquePathsWithObstacles(int[][] obstacleGrid) {
            int m = obstacleGrid.length;
            int n = obstacleGrid[0].length;
            if(m < 1 || n < 1) return 0;
            int[][] matrix = new int[m][n];
            for (int i = 0; i < m; i++) {
                if(obstacleGrid[i][0]==1){
                    for (int j = i; j < m; j++) matrix[j][0] = 0;
                    break;
                }
                else matrix[i][0] = 1;
            }
            for (int j = 0; j < n; j++){
                if(obstacleGrid[0][j]==1){
                    for (int i = j; i < n; i++) matrix[0][i] = 0;
                    break;
                }
                else matrix[0j][] = 1;
            }
            for(int i = 1; i < m; i++){
                for (int j = 1; j < n; j++){
                    if (obstacleGrid[i][j]==1) {
                        matrix[i][j] = 0;
                        continue;
                    }
                    matrix[i][j] = matrix[i-1][j] + matrix[i][j-1];
                }
            }
            return matrix[m-1][n-1];

            int[] matrix = new int[n];
            matrix[0] = 1;
            for(int i = 1; i < n ; i++){
                if (obstacleGrid[0][i] == 1) matrix[i] = 0;
                else matrix[i] += matrix[i-1];
            }
            for (int j = 1; j < m; j++)
                for(int i = 0; i < n ; i++){
                    if (obstacleGrid[j][i] == 1) matrix[i] = 0;
                    if (i == 0) matrix[i] = matrix[i];
                    else matrix[i] += matrix[i-1];
                }
            }
        }
    }
}

class Solution T3. Longest Substring Without Repeating Characters{
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int start = 0;
        int max = 0;
        for (int i = 0; i < s.length(); i++){
            Character temp = s.charAt(i);
            if (map.containsKey(temp)) {
                if (map.get(temp) >= start) start = map.get(temp);
            }
            map.put(temp, i)
            max = Math.max(max, i - start + 1);
        }
        return max;
    }
}

class Solution T22 Generate Parentheses{
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<String>();
        String temp = "";
        helper(res, temp, n);
        return res;
    }
    public void helper(List<String> res, String temp, int n){
        if(n==0) res.add(temp);
        n = n - 2;
        String t1 = "()" + temp;
        String t2 = "(" + temp + ")";
        helper(res, t1, n - 2);
        helper(res, t2, n - 2);
        return;
    }
    public int longestValidParentheses(String s) {
        Stack<Character> stack = new Stack<>();
        int res = 0, start = 0;
        for (int i = 0; i < s.length(); i++){
            if (s.charAt(i) == "(") stack.push(i);
            else if (s.charAt(i) == ")") {
                if (stack.isEmpty()) start = i + 1;
                else {
                    stack.pop();
                    int temp = stack.top();
                    res = stack.isEmpty()? Math.max(res, i - start + 1):Math.max(res, i - temp);
                }
            }
        }
    }

    public int longestValidParentheses_DP(String s) {
        int res = 0;
        int[] dp = new int[s.length() + 1];
        for (int i = 1; i <= s.length; ++i) {
            int j = i - 1 - dp[i - 1];
            if (s[i - 1] == '(' || j < 0 || s[j] == ')') {
                dp[i] = 0;
            } else {
                dp[i] = dp[i - 1] + 2 + dp[j - 1];
                res = max(res, dp[i]);
            }
    }

    public int longestValidParentheses_v3(String s) {
        int left = 0, right = 0, res = 0, start = 0;
        for (int i = 0; i < s.size(); i++) {
            if(s.charAt(i) == ")") {
                right++;
                if (right > left) {
                    start = i + 1;
                    res = Math.max(res, left);
                }
                else if (right == left) {
                    res = Math.max(res, left);
                }
            }
            else if(s.charAt(i) == "("){
                left++;
            }
        }
    }
    public double mypow(double x, int n){
        double res = 1.0 ;
        for (int i = n; i > 0 ; i/=2) {
            res = res * res;
            if (i%2!=0) {
                res *= x;
            }
        }
        return res;
    }
} 

class Solution T324 Wiggle Sort II{
    public void wiggleSort(int[] nums) {
        
    }

    public int findKthLargest(int[] data, int k) {
        int left = 0, right = data.length - 1;
        int pos = partition(data, left, right);
        while (true) {
            int pos = partition(nums, left, right);
            if (pos == k - 1) return nums[pos];
            if (pos > k - 1) right = pos - 1;
            else left = pos + 1;
        }
    }

    int partition(int[] nums, int left, int right) {
        int pivot = nums[left], l = left + 1, r = right;
        while (l <= r) {
            if (nums[l] < pivot && nums[r] > pivot) {
                swap(nums[l++], nums[r--]);
            }
            if (nums[l] >= pivot) ++l;
            if (nums[r] <= pivot) --r;
        }
        swap(nums[left], nums[r]);
        return r;
    }

    //三路快排另一种写法
    public static void quickSort(int[] arr, int l, int r) {
		if (l < r) {
			swap(arr, l + (int) (Math.random() * (r - l + 1)), r);
			int[] p = partition(arr, l, r);
			quickSort(arr, l, p[0] - 1);
			quickSort(arr, p[1] + 1, r);
		}
	}

    //我最爱的三路快排
	public static int[] partition(int[] arr, int l, int r) {
		int less = l - 1;
		int more = r;
		while (l < more) {
			if (arr[l] < arr[r]) {
				swap(arr, ++less, l++);
			} else if (arr[l] > arr[r]) {
				swap(arr, --more, l);
			} else {
				l++;
			}
		}
		swap(arr, more, r);
		return new int[] { less + 1, more };
	}
}

class SolutionT78Subsets {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new List<List<Integer>>();
        List<Integer> temp = new List<Integer>();
        backstrack(res, temp, 0);
        return res;
    }

    public void backstrack(List<List<Integer>> res, List<Integer> temp, int[] nums, int index) {
        if(index >= nums.length) {
            res.add(temp);
            return;
        }
        temp.add(nums[index]);
        backstrack(res, temp, nums, index);
        temp.remove(nums.length() - 1);
        backstrack(res, temp, nums, index);
        return 
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


 