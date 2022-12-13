from abc import abstractproperty
from ast import Index
import collections
from typing import Collection, List, Optional
from collections import *
import sys
import bisect
import heapq
import math

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        startTime = sorted([t[0] for t in intervals])
        endTime = sorted([t[1] for t in intervals])
        endPos = 0
        res = 0
        for i in range(len(intervals)):
            if startTime[i] < endTime[endPos]:
                res += 1
            else:
                endPos += 1
        return res

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        map = collections.defaultdict(int)
        for i in intervals:
            map[i[0]]+=1
            map[i[1]]-=1
        sort_map = collections.OrderedDict(map)
        res = 0
        rooms = 0
        for item in sort_map.items():
            rooms += item[1]
            res = max(res, rooms)
        return res

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        intervals.sort(key = lambda x : x[0])
        heap = []
        for i in intervals:
            if len(heap) > 0 and i[0] >= heap[0]:
                heapq.heappop(heap)
            heapq.heappush(heap, i[1])
        return len(heap)

class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        st = []
        cnt = k
        for i in range(len(s)):
            if len(st) == 0:
                st.append(s[i])
                cnt = k - 1
                continue
            if s[i] == st[-1]:
                cnt -= 1
            else:
                cnt = k  - 1
            st.append(s[i])
            if cnt == 0:
                while len(st) > 0 and cnt < k:
                    st.pop()
                    cnt += 1
        return "".join(st)

class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        n = len(s)
        count = [0 for i in range(n)]
        st = []
        for i in range(len(s)):
            if len(st) <= 0:
                st.append(i)
                count[i] = 1
                continue
            if s[i] == s[st[-1]]:
                idx = st[-1]
                st.append(i)
                count[i] = count[idx] + 1
                if count[i] == k:
                    for j in range(k): st.pop()
            else:
                st.append(i)
                count[i] = 1
        str_list = [s[i] for i in st]
        return "".join(str_list)

class UndergroundSystem:

    def __init__(self):
        self.timeData = collections.defaultdict(int)
        self.travelTimes = collections.defaultdict(int)
        self.travelRecord = collections.defaultdict(list)

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        self.travelRecord[id] = [stationName, t]

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        startCity = self.travelRecord[id][0]
        startTime = self.travelRecord[id][1]
        encodeKey = startCity + "2" + stationName
        self.timeData[encodeKey] += t - startTime
        self.travelTimes[encodeKey] += 1

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        encodeKey = startStation + "2" + endStation
        return self.timeData[encodeKey] / self.travelTimes[encodeKey]


# Your UndergroundSystem object will be instantiated and called as such:
# obj = UndergroundSystem()
# obj.checkIn(id,stationName,t)
# obj.checkOut(id,stationName,t)
# param_3 = obj.getAverageTime(startStation,endStation)

class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        heapA = []
        heapB = []
        counter = 0
        for cost in costs:
            heapq.heappush(heapA, (cost[0], counter))
            heapq.heappush(heapB, (cost[1], counter))
            counter+=1
        added = set()
        res = 0
        for i in range(len(costs)):
            if i % 2 == 0:
                while heapA[0][1] in added:
                    t = heapq.heappop(heapA)
                res += heapA[0][0]
                added.add(heapA[0][1])
                heapq.heappop(heapA)
            else:
                while heapB[0][1] in added:
                    t = heapq.heappop(heapB)
                res += heapB[0][0]
                added.add(heapB[0][1])
                heapq.heappop(heapB)
        return res

class Solution:
    def maxSumTwoNoOverlap(self, nums: List[int], firstLen: int, secondLen: int) -> int:
        n = len(nums)
        sum = [0 for i in range(n + 1)]

        for i in range(1, n + 1):
            sum[i] = sum[i - 1] + nums[i - 1] 

        # first 在前 second 在后，遍历一个，dp[i] 记录i之前 另一个的sumMax
        res = 0
        dp = [0 for i in range(n + 1)]
        for i in range(firstLen, n + 1):
            tempSum = sum[i] - sum[i - firstLen]
            if i - firstLen - secondLen >= 0:
                tempSumSec = sum[i - firstLen] - sum[i - firstLen - secondLen]
                dp[i - firstLen] = max(dp[i - firstLen - 1], tempSumSec)
                res = max(res, tempSum + dp[i - firstLen])
        
        dp1 = [0 for i in range(n + 1)]
        for i in range(secondLen, n + 1):
            tempSum = sum[i] - sum[i - secondLen]
            if i - firstLen - secondLen >= 0:
                tempSumSec = sum[i - secondLen] - sum[i - firstLen - secondLen]
                dp[i - secondLen] = max(dp[i - secondLen - 1], tempSumSec)
                res = max(res, tempSum + dp[i - secondLen])
        
        return 

class Solution:
    def fun(self, input: List[str]):
        map = collections.defaultdict(str)
        longest = 0
        longestStr = ""
        res = []
        for q in input:
            idx = ord(q[1]) - ord("0")
            map[idx] = q[0]
            if idx == longest + 1:
                while longest + 1 in map:
                    longestStr = longestStr + map[longest + 1]
                    longest += 1
            res.append(longestStr)
        return res

input = ["c3", "b2", "a1"]
obj = Solution()
print(obj.fun(input))
            
#merge intervals
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x : x[0])
        idx = 0
        res = []
        temp = []
        while idx < len(intervals):
            st = intervals[idx][0]
            end = intervals[idx][1]
            while idx + 1 < len(intervals) and intervals[idx + 1][0] <= end:
                end = max(end, intervals[idx + 1][1])
                idx += 1
            temp = [st, end]
            res.append(temp)
            idx += 1
        return res

class Solution:
    def KMin(self, arr: List[int], k: int):
        counter = [0] * 10001
        for a in arr:
            counter[a]+=1
        res = []
        idx = 0
        for i in range(len(counter)):
            while(counter[i] >= 1 and idx < k):
                res.append(arr[idx])
                idx += 1
                counter[i] -= 1
            if idx == k:
                break
        return res
    
    #maxHeap
    def KMin(self, arr: List[int], k: int):
        heap = []
        for a in arr:
            if len(heap) < k:
                heapq.heappush(heap, -a)
                continue
            if -a > heap[0]:
                heapq.heappop(heap)
                heapq.heappush(heap, -a)
        res = [-i for i in heap].sort()
        return res

class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child

class Solution:
    def flatten(self, head: 'Node') -> 'Node':
        dummy = head
        cur = head
        while cur:
            if not cur.child:
                cur = cur.next
                continue
            nextNode = cur.next
            childHead = cur.child
            childEnd = childHead
            while childEnd.next:
                childEnd = childEnd.next
            cur.next = childHead
            childHead.prev = cur
            cur.child = None
            childEnd.next = nextNode
            if nextNode: nextNode.prev = childEnd
        return dummy

# BrowserHistory with list
class BrowserHistory:

    def __init__(self, homepage: str):
        self.history = [homepage]
        self.current = 0

    def visit(self, url: str) -> None:
        self.history = self.history[:self.current+1]
        self.history.append(url)
        self.current += 1

    def back(self, steps: int) -> str:
        self.current = max(0, self.current-steps)
        return self.history[self.current]

    def forward(self, steps: int) -> str:
        self.current = min(len(self.history)-1, self.current+steps)
        return self.history[self.current]

#BrowserHistroy with st
class BrowserHistory:

    def __init__(self, homepage: str):
        self.cur_url = homepage
        self.back_st = []
        self.forward_st = []

    def visit(self, url: str) -> None:
        self.back_st.append(self.cur_url)
        self.forward_st = []
        self.cur_url = url

    def back(self, steps: int) -> str:
        steps = min(len(self.back_st), steps)
        for i in range(steps):
            self.forward_st.append(self.cur_url)
            self.cur_url = self.back_st.pop()
        return self.cur_url

    # def forward(self, steps: int) -> str:
        


# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)


# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)
        

# Definition for a binary tree node.

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    # def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
    #     map = collections.defaultdict(list)
    #     queue = collections.deque()
    #     queue.append((root, 0))
    #     level = 0
    #     while len(queue) > 0:
    #         size = len(queue)
    #         for i in range(size):
    #             node, col = queue.popleft()
    #             map[col].append(node.val)
    #             if node.left: queue.append((node.left, col - 1))
    #             if node.right: queue.append((node.right, col + 1))
    #     ordered_map = sorted(map.items(), key=lambda x : x[0])
    #     res = []
    #     for i in ordered_map: res.append(i[1])
    #     return res

    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        map = collections.defaultdict(list)
        queue = collections.deque()
        queue.append((root, 0))
        level = 0
        while len(queue) > 0:
            size = len(queue)
            for i in range(size):
                node, col = queue.popleft()
                map[col].append((level, node.val))
                if node.left: queue.append((node.left, col - 1))
                if node.right: queue.append((node.right, col + 1))
            level+=1
        ordered_map = sorted(map.items(), key=lambda x : x[0])
        res = []
        for li in ordered_map:
            t = li[1]
            t.sort()
            res.append([a[1] for a in t])
        return res

class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        res = []
        temp = [0]
        n = len(graph)

        def dfs(cur, temp: List[int]) :
            if cur == n - 1:
                res.append(temp.copy())
            t = temp.copy()
            for i in graph[cur]:
                t.append(i)
                dfs(i, t)
                t.pop()
        
        dfs(0, temp)

        return res
        