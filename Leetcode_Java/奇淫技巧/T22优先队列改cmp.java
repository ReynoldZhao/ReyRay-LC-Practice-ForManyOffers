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

PriorityQueue<int[]> que = new PriorityQueue<>((a,b)->a[0]+a[1]-b[0]-b[1]);

class SolutionT378 {
    public int kthSmallest(int[][] matrix, int k) {
        PriorityQueue<Integer> maxheap = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2.compareTo(o1);
            }
        });
    }