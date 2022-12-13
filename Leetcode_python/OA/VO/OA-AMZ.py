from _typeshed import Self
from asyncio import FastChildWatcher
from audioop import reverse
from bisect import bisect, bisect_left
from cmath import inf, pi
import collections
import heapq
from pydoc import Helper
from re import M
import turtle
from typing import Collection, List, Optional
from collections import *
import sys
import math
from numpy import diff
from regex import R

from sqlalchemy import false, true
from sympy import Q

class maxAggregateTemperatureChange:
    def minimumHealth(self, tempChange: List[int]) -> int:
        presumArray = []
        sum = 0
        for t in tempChange:
            sum += t
            presumArray.append(sum)
        res = float(-inf)
        for i in range(len(tempChange)):
            pre = presumArray[i]
            back = sum - presumArray[i] + tempChange[i]
            res = max(res, max(pre, back))
        return res

class makePowerNonDecreasing:
    def solution(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        add = 0
        preLargest = nums[0]
        for i in range(1, len(nums)):
            if nums[i] + add <= preLargest:
                add += preLargest - (nums[i] + add)
            preLargest = max(preLargest, nums[i] + add)
        return add


class consecutivelyDecreasing:
    def solution(self, nums: List[int]) -> int:
        if (len(nums) == 0):
            return 0
        dp = [1] * len(nums)
        res = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] - 1:
                dp[i] = dp[i - 1] + 1
            res += dp[i]
        return res

class Solution:
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        nums.sort()
        n, l, r = len(nums), 0, nums[-1] - nums[0]
        while l < r:
            mid = (r - l) // 2 + l
            cnt, start = 0, 0
            for i in range(len(nums)):
                while start < n and nums[i] - nums[start] > mid:
                    start += 1
                cnt += i - start
            if cnt < k:
                l = mid + 1
            else:
                r = mid
        return r 

    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        heap = [(nums[i+1]-nums[i], i, i + 1) for i in range(n-1)]
        heapq.heapify(heap)

        for _ in range(k):
            d, root, next_idx = heapq.heappop(heap)

            if next_idx + 1 < n:
                heapq.heappush((nums[next_idx + 1] - nums[root], root, next_idx + 1))
        
        return d

