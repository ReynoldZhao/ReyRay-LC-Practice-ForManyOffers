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

class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> List[int]:
        idxMap = collections.defaultdict(int)
        sum = 0
        min_long = float(inf)
        res_list = []
        pre_sum = [0 for i in range(len(nums) + 1)]
        idxMap[0] = 0
        # for i in range(len(nums)):
        #     sum += nums[i]
        #     if sum >= k:
        #         mod_val = sum % k
        #         if mod_val in idxMap:
        #             pre_idx = idxMap[mod_val]
        #             if i - pre_idx + 1 <= min_long:
        #                 res_list = nums[pre_idx:i+1]
        #                 min_long = i - pre_idx + 1

        #         idxMap[mod_val] = i
        for i in range(1, len(nums) + 1):
            sum += nums[i - 1]
            pre_sum[i] = sum

            if sum >= k:
                mod_val = sum % k
                if mod_val in idxMap:
                    pre_idx = idxMap[mod_val]
                    if sum - pre_sum[pre_idx] >= k and i - pre_idx <= min_long:
                        if pre_idx == 0:
                            res_list = nums[i]
                        else:
                            res_list = nums[pre_idx - 1:i]
                        min_long = i - pre_idx

                idxMap[mod_val] = i
        
        return res_list

class Solution:
    def shortestSubarrayDivisibleByK(self, nums: List[int], k: int) -> List[int]:
        idxMap = collections.defaultdict(int)
        sum = 0
        min_long = float(inf)
        res_list = []
        pre_sum = [0 for i in range(len(nums) + 1)]
        idxMap[0] = 0
        for i in range(1, len(nums) + 1):
            sum += nums[i - 1]
            pre_sum[i] = sum

            if sum >= k:
                mod_val = sum % k
                if mod_val in idxMap:
                    pre_idx = idxMap[mod_val]
                    if sum - pre_sum[pre_idx] >= k and i - pre_idx <= min_long:
                        if pre_idx == 0:
                            res_list = nums[:i]
                        else:
                            res_list = nums[pre_idx:i]
                        min_long = i - pre_idx

                idxMap[mod_val] = i
        
        return res_list

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        res = []
        intervals.sort(key=lambda k:k[0])
        idx = 0
        while idx < len(intervals):
            cur_interval = intervals[idx]
            next_idx = idx + 1
            while next_idx < len(intervals) and intervals[next_idx][0] <= cur_interval[1]:
                cur_interval[1] = max(intervals[next_idx][1], cur_interval[1])
                next_idx += 1
            idx = next_idx
            res.append(cur_interval)
        return res

# Divide an array into subsequences with sum no greater‍‍‍‌‌‌‍‌‌‌‍‌‌‍‍‍‌‍‌ than M, minimize the sum of maximum value from each division
class Solution:
    def solve(self, nums: List[int], M: int) -> int:
        n = len(nums)
        dp = [float(inf)] * n
        sum = [0 for i in range(n)]
        max_in_sub = [[0 for i in range(n)] for i in range(n)] 

        dp[0] = nums[0]
        for i in range(n):
            sum[i] = (0 if i == 0 else sum[i-1]) + nums[i]
        
        for i in range(n):
            cur_max = nums[i]
            for j in range(i, n):
                if nums[j] >= cur_max:
                    cur_max = nums[j]
                max_in_sub[i][j] = cur_max
        
        #nums[1 ... n]
        for i in range(1, n):
            #forward 0, 1, 2, ..., i
            for k in range(i+1):
                if k == i and sum[i] <= M:
                    dp[i] = min(dp[i], max_in_sub[0][i])
                elif i > k and (sum[i] - sum[i - k - 1]) <= M:
                    dp[i] = min(dp[i], max_in_sub[i-k][i] + dp[i - k - 1])
        
        return dp[-1]

class Solution:
    def solve(self, nums: List[int], M: int) -> int:
        n = len(nums)
        #dp[i] = the minimum sum of maximum value from each division nums[0,... i - 1]
        dp = [float(inf) for i in range(n + 1)]
        sum = [0 for i in range(n + 1)]
        max_in_sub = [[0 for i in range(n)] for i in range(n)] 

        dp[0] = 0
        for i in range(1, n + 1):
            sum[i] = sum[i - 1] + nums[i - 1]
        
        for i in range(n):
            cur_max = nums[i]
            for j in range(i, n):
                if nums[j] >= cur_max:
                    cur_max = nums[j]
                max_in_sub[i][j] = cur_max
        
        for i in range(1, n + 1):

            #k = 1, newsub = [nums[i]], k = 2, newsub = [nums[i-1], nums[i]]
            #k == i, newsub = [nums[0], nums[1], ... nums[i]]
            for k in range(1, i + 1):
                if sum[i] - sum[i-k] <= M:
                    dp[i] = min(dp[i], max_in_sub[i - k][i - 1] + dp[i - k])
        
        return dp[-1]


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        sum = 0
        res = float(-inf)
        for i in range(len(nums)):
            sum += nums[i]
            res = max(res, sum)
            if sum < 0:
                sum = 0
        return res

class Solution:
    def minimumResistancePath(self, matrix: List[List[int]]):
        m = len(matrix)
        n = len(matrix[0])

        dp = [matrix[i][0] for i in range(m)]

        for i in range(1, n):
            tmp_dp = [0 for i in range(m)]
            for j in range(m):
                if j == 0:
                    tmp_dp[j] = matrix[j][i] + min(dp[j], dp[j + 1])
                elif j == m - 1:
                    tmp_dp[j] = matrix[j][i] + min(dp[j], dp[j - 1])
                else:
                    tmp_dp[j] = matrix[j][i] + min(dp[j], dp[j-1], dp[j + 1])
            dp = tmp_dp
        
        return min(dp)

class Solution:
    def sovle(self, nums: List[int], questions: List[List[int]]):
        ques_amount = len(questions)
        