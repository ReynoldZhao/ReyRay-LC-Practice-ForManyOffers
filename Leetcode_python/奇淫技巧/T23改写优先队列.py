from Queue import PriorityQueue
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        dummy = listNode(None)
        cur = dummy
        q = PriorityQueue()
        for node : lists:
            if node:
                q.put((node.val, node))
        while q.size() > 0:
            cur.next = q.get()[1]
            cur = cur.next
            if cur.next:
                q.put((cur.next.val, cur.next))
        return dummy.next