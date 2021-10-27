from Queue import PriorityQueue
import collections
import heapq
import functools

# class T23Solution(object):
#     def mergeKLists(self, lists):
#         """
#         :type lists: List[ListNode]
#         :rtype: ListNode
#         """
#         dummy = listNode(None)
#         cur = dummy
#         q = PriorityQueue()
#         for node : lists:
#             if node:
#                 q.put((node.val, node))
#         while q.size() > 0:
#             cur.next = q.get()[1]
#             cur = cur.next
#             if cur.next:
#                 q.put((cur.next.val, cur.next))
#         return dummy.next

@functools.total_ordering
class Element:
    def __init__(self, count, word):
        self.count = count
        self.word = word
        
    def __lt__(self, other):
        if self.count == other.count:
            return self.word > other.word
        return self.count < other.count
    
    def __eq__(self, other):
        return self.count == other.count and self.word == other.word

#python的 heapq改写的大小比较顺序是自然顺序
#由于pythonheapq只能是小顶堆，self.count < other.count，频率小的靠近堆顶
#self.word > other.word，字符顺序大的靠近堆顶
#总之就是返回true就是靠近堆顶
class Solution(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        counts = collections.Counter(words)   
        
        freqs = []
        heapq.heapify(freqs)
        for word, count in counts.items():
            heapq.heappush(freqs, (Element(count, word), word))
            if len(freqs) > k:
                heapq.heappop(freqs)
        
        res = []
        for _ in range(k):
            res.append(heapq.heappop(freqs)[1])
        return res[::-1]