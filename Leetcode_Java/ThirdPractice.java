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
    
    public static List<Integer> junieBirthday(List<Integer> nums, List<List<Integer>> questions) {
        List<Integer> result = new ArrayList<>();
        for(int i = 0; i < questions.size(); i++) {
            List<Integer> q = questions.get(i);
            List<Integer> temp_sublist = nums.subList(q.get(0), q.get(1) + 1);
            result.add(subarraysDivByK(temp_sublist.toArray(), q.get(2)));
        }
    }
    
    public int subarraysDivByK(int[] nums, int k) {
        int n = nums.length;
        int prefixMod = 0, result = 0;

        // There are k mod groups 0...k-1.
        int[] modGroups = new int[k];
        modGroups[0] = 1;

        for (int num: nums) {
            // Take modulo twice to avoid negative remainders.
            prefixMod = (prefixMod + num % k + k) % k;
            // Add the count of subarrays that have the same remainder as the current
            // one to cancel out the remainders.
            result += modGroups[prefixMod];
            modGroups[prefixMod]++;
        }

        return result;
    }
}
