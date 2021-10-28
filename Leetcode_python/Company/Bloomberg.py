from _typeshed import Self
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
            