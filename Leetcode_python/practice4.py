from _typeshed import Self
from asyncio import FastChildWatcher
from audioop import reverse
from base64 import decode
from bisect import bisect, bisect_left
from cmath import inf, pi
import collections
from curses.ascii import isdigit
from functools import lru_cache
import heapq
from operator import le
from pydoc import Helper
from re import I, M
import turtle
from typing import Collection, List, Optional
from collections import *
import sys
import math
from matplotlib.cbook import flatten
from numpy import diff, sort
from regex import R
import random

from sqlalchemy import false, true
from sympy import N, Q

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:

    def isValidBST(self, root: Optional[TreeNode]) -> bool:

        def helper(node, low, high):
            if not node:
                return True
            if node.val >= high or node.val <= low:
                return False
            
            return helper(node.left, low, node.val) and helper(node.right, node.val, high)
        
        return helper(root, -math.inf, math.inf)

    def isValidBST(self, root: Optional[TreeNode]) -> bool:

        def helper(node, low, high):
            if not node:
                return True
            if node.val >= high or node.val <= low:
                return False
            
            return helper(node.left, low, node.val) and helper(node.right, node.val, high)
        
        return helper(root, -math.inf, math.inf)

class Solution:
    def largestBSTSubtree(self, root: Optional[TreeNode]) -> int:

        res = 1

        err = - 10**4 - 1

        def helper(node):
            if not node:
                return (err, err, 0)
            if not node.left and not node.right:
                return (node.val, node.val, 1)

            lmin, lmax, lcnt = helper(node.left)
            rmin, rmax, rcnt = helper(node.right)

            if lmax < node.val and node.val < rmin and lcnt > 0 and rcnt > 0:
                t_res = lcnt + rcnt + 1
                res = max(res, t_res)
                return (lmin, rmax, t_res)
            elif node.left == None and node.val < rmin and rcnt > 0:
                t_res = rcnt + 1
                res = max(res, t_res)
                return (node.val, rmax, t_res)
            elif node.right == None and lmax < node.val and lcnt > 0:
                t_res = lcnt + 1
                res = max(res, t_res)
                return (lmin, node.val, t_res)
            else:
                return (err, err, 0)
        
        # def helper(node):
        #     if not node:
        #         return (err, err, 0)
        #     if not node.left and not node.right:
        #         return (node.val, node.val, 1)

        #     lmin, lmax, lcnt = helper(node.left)
        #     rmin, rmax, rcnt = helper(node.right)

        #     if lmax < node.val and node.val < rmin and lcnt > 0 and rcnt > 0:
        #         t_res = lcnt + rcnt + 1
        #         res = max(res, t_res)
        #         return (lmin, rmax, t_res)
        #     else:
        #         return (-1, -1, 0)

        _1, _2, _3 = helper(root)

        return res

class Solution:
    def largestBSTSubtree(self, root: Optional[TreeNode]) -> int:
        
        err = - 10**4 - 1

        def helper(node):
            if not node:
                return (err, err, 0, 0)
            
            if not node.left and not node.right:
                return (node.val, node.val, 1, 1)

            lmin, lmax, lcnt, lres = helper(node.left)
            rmin, rmax, rcnt, rres = helper(node.right)

            res = max(lres, rres)
            
            if lmax < node.val and node.val < rmin and lcnt > 0 and rcnt > 0:
                t_res = lcnt + rcnt + 1
                res = max(res, t_res)
                return (lmin, rmax, t_res, res)
            else:
                return (err, err, 0, res)
        
        _1, _2, _3, res = helper(root)

        return res

class Solution:
    def largestBSTSubtree(self, root: Optional[TreeNode]) -> int:
        
        # res = 1

        err = - 10**4 - 1

        def helper(node):
            if not node:
                return (err, err, 0, 0)
            if not node.left and not node.right:
                return (node.val, node.val, 1, 1)

            lmin, lmax, lcnt, lres = helper(node.left)
            rmin, rmax, rcnt, rres = helper(node.right)
            
            res = max(lres, rres)

            if lmax < node.val and node.val < rmin and lcnt > 0 and rcnt > 0:
                t_res = lcnt + rcnt + 1
                res = max(res, t_res)
                return (lmin, rmax, t_res, res)
            elif node.left == None and node.val < rmin and rcnt > 0:
                t_res = rcnt + 1
                res = max(res, t_res)
                return (node.val, rmax, t_res, res)
            elif node.right == None and lmax < node.val and lcnt > 0:
                t_res = lcnt + 1
                res = max(res, t_res)
                return (lmin, node.val, t_res, res)
            else:
                return (err, err, 0, res)
        
        _1, _2, _3, res = helper(root)


class Solution:
    def trap(self, height: List[int]) -> int:
        res = 0
        l, r = 0, len(height) - 1
        while l < r:
            t = max(height[l], height[r])

            while l < r and height[l] < t:
                res += t - height[l]
                l += 1
            
            while l < r and height[r] < t:
                res += t - height[r]
                r -= 1
        return res

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        st = []
        t_res = -float(inf)
        heights.append(0)
        for i in range(len(heights)):
            if len(st) == 0 or heights[i] >= heights[st[-1]]:
                st.append(i)
            else:
                while len(st) != 0 and heights[i] < heights[st[-1]]:
                    area = heights[st[-1]] * (st[-1] - i)
                    st.pop()
                    t_res = max(t_res, area)
                st.append(i)
        return t_res

    def largestRectangleArea(self, heights: List[int]) -> int:
        heights.append(-1)
        st = []
        t_res = -float(inf)
        for i in range(len(heights)):
            while len(st) != 0 and heights[i] <= heights[st[-1]]:
                cur = st[-1]
                st.pop()
                width = i if len(st) == 0 else i - st[-1] - 1
                area = heights[cur] * width
                t_res = max(t_res, area)
            st.append(i)
        return t_res

class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:
        res = 0
        st = []
        INF = float(inf)
        A = [-INF] + nums + [-INF]
        for k, v in enumerate(A):
            while st and v < A[st[-1]]:
                j = st[-1]
                st.pop()
                i = st[-1]
                res -= A[j] * (j - i) * (k - j)
            st.append(k)
        
        A = [INF] + nums + [INF]
        st = []
        for k, v in enumerate(A):
            while st and v > A[st[-1]]:
                j = st[-1]
                st.pop()
                i = st[-1]
                res += A[j] * (j - i) * (k - j)
            st.append(k)
        
        return res

class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        if k >= len(num):
            return "0"
        st = []
        for i in range(len(num)):
            while st and num[i] < st[-1] and k > 0:
                st.pop()
                k-=1
            st.append(num[i])
        while len(st) > 1 and st[0] == "0":
            st = st[1:]
        if k >= len(st):
            return "0"
        if k > 0:
            st = st[:-k]
        return "".join(st)

class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        alpha_count = [0 for i in range(256)]
        visited = [0 for i in range(256)]
        res = ["0"]
        for c in s:
            alpha_count[ord(c)] += 1
        for c in s:
            alpha_count[ord(c)] -=1
            if visited[ord(c)] > 0:
                continue;
            while c < res[-1] and alpha_count[ord(res[-1])] > 0:
                t = res.pop()
                visited[ord(t)] = 0
            res.append(c)
            visited[ord(c)] = 1
        res = res[1:]
        return "".join(res)

# """
# This is ArrayReader's API interface.
# You should not implement it, or speculate about its implementation
# """
#class ArrayReader:
#    def get(self, index: int) -> int:

class Solution:
    def search(self, reader: 'ArrayReader', target: int) -> int:
        l, r = 0, 10**4
        while l < r:
            mid = (r - l) // 2 + l
            v = reader.get(mid)
            if v == 2**31 - 1:
                r = mid
            
            if v == target:
                return mid
            elif v < target:
                l = mid + 1
            else:
                r = mid

        return -1

class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (r - l) // 2 + l
            v = nums[mid]
            if v < nums[r]:
                r = mid
            elif v > nums[r]:
                l = mid + 1
        return nums[r]

class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        st = []
        r, l = -1, -2
        # increasing stack
        for i in range(len(nums)):
            while (len(st) > 0 and nums[i] < nums[st[-1]]):
                l = min(l, st[-1])
                st.pop()
            st.append(i)
        
        st.clear()
        for i in range(len(nums) - 1, -1, -1):
            while (len(st) > 0 and nums[i] > nums[st[-1]]):
                r = max(r, st[-1])
                st.pop()
            st.append(i)

        return r - l + 1 if r - l > 0 else 0

class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        l, r = 0, len(arr) - k
        while l < r:
            mid =(r - l) // 2 + l
            if x - arr[mid] > arr[mid + k] - x :
                l = mid + 1
            else :
                r = mid
        return arr[r:r+k]

class HitCounter:

    #word, but not a good circular array
    def __init__(self):
        self.pool = [(i, 0) for i in range(301)]

    def hit(self, timestamp: int) -> None:
        index = timestamp % 300
        pre = self.pool[index]
        pre_ts = pre[0]
        pre_cnt = pre[1]
        if timestamp > pre_ts:
            self.pool[index] = (timestamp, 1)
        else:
            self.pool[index] = (pre_ts, pre_cnt + 1)
            

    def getHits(self, timestamp: int) -> int:
        sum = 0
        for t in self.pool:
            sum += t[1]
        return sum

    #或者用deque
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        st = []
        third = -float(inf)
        for i in range(len(nums) - 1, -1, -1):
            if (nums[i] < third):
                return True
            while (len(st) > 0 and nums[i] > st[-1]):
                third = max(third, st[-1])
                st.pop()
            st.append(nums[i])
        return False

    def find132pattern(self, nums: List[int]) -> bool:
        st = []
        first = float(inf)
        for i in range(len(nums)):
            if nums[i] > first:
                return True
            while (len(st) > 0 and nums[i] > st[-1]):
                first = min(first, st[-1])
                st.pop()
            st.append(nums[i])
        return True

class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        all = sum(weights)
        l, r = 0, (all // days + 1) 

        def finishDay(cap):
            index = 0
            t_sum = 0
            days = 0
            while index < len(weights):
                if t_sum + weights[index] <= cap:
                    t_sum += weights[index]
                else:
                    t_sum = 0
                    days += 1
                index += 1

        while l < r :
            mid = (r - l) // 2 + l
            exp_days = finishDay(mid)
            
            if exp_days > days:
                l = mid + 1
            else:
                r = mid - 1
        
        return r

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        res = 0

        def helper(index, pos, sum):
            sym = 1 if pos else -1
            if (index >= len(nums)):
                if sum == target:
                    res += 1
                else:
                    return
            helper(index + 1, True, sum + sym * nums[index])
            helper(index + 1, False, sum - sym * nums[index])
        
        helper(0, True, 0)
        helper(0, False, 0)

        return res

    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        self.res = 0
        self.memo = collections.defaultdict(defaultdict)#[collections.defaultdict for i in range(len(nums))]

        def helper(index, sum):
            if index >= len(nums):
                return sum == 0
            if self.memo[index][sum] > 0:
                return self.memo[index][sum]

            cnt1 = helper(index+1, sum+nums[index])
            cnt2 = helper(index+1, sum-nums[index])

            self.memo[index][sum] = cnt1 + cnt2

            return self.memo[index][sum]

        return helper(0, 0)
    
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        count = defaultdict(int)

        for x in nums:
            tmp = defaultdict(int)
            for y in count:
                tmp[y + x] += count[y]
                tmp[y - x] += count[y]
            count = tmp
        
        return tmp[target]

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        def bst(index, sum, tmp):
            if index >= len(candidates):
                if sum == target:
                    res.append(tmp)
                return
            if sum == target:
                res.append(tmp)
            for i in range(index, len(candidates)):
                tmp.append(candidates[i])
                bst(index + 1, sum + candidates[index], tmp)
                tmp.pop()
        bst(0, 0, [])
        return res

class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        def helper(index, sum, tmp):
            
            for i in range(index, len(candidates)):
                if i > index and candidates[i] == candidates[i-1]:
                    continue

class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        if k > len(nums):
            return False
        all_nums = sum(nums)
        if all_nums % k != 0:
            return False
        self.visited = [0 for i in range(len(nums))]
        target = all_nums / k
        def helper(index, k, target, curSum):
            if k == 1:
                #前两个if已经排除了不存在k种的情况，就是剩下的数的和不等于一个target的情况
                return True
            if curSum == target:
                return helper(0, k-1, target, 0)
            for i in range(index, len(nums)):
                if self.visited[i] == 1:
                    continue
                self.visited[i] = 1
                if helper(index+1, k, curSum + nums[i]):
                    return True
                self.visited[i] = 0
        return helper(0, k, target, 0)

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordSet = set(wordList)
        if endWord not in wordSet:
            return 0
        pathCnt = defaultdict(int)
        pathCnt[beginWord] = 1
        q = deque()
        q.append(beginWord)
        while (len(q) > 0):
            word = q.popleft()
            for i in range(len(word)):
                t = list(word)
                for j in range(26):
                    t[i] = chr(ord('a') + j)
                    temp_word = "".join(t)
                    if temp_word in wordSet and temp_word == endWord:
                        return pathCnt[word] + 1
                    if temp_word in wordSet and temp_word not in pathCnt:
                        q.append(temp_word)
                        pathCnt[temp_word] = pathCnt[word] + 1
        return 0

class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        if k > len(nums):
            return False
        all_nums = sum(nums)
        if all_nums % k != 0:
            return False
        visited = ['0' for i in range(len(nums))]
        target = all_nums / k
        nums.sort(reverse=True)
        memo = {}
        def helper(index, k, curSum):
            mask = "".join(visited)
            if k == 1:
                #前两个if已经排除了不存在k种的情况，就是剩下的数的和不等于一个target的情况
                return True
            if curSum > target:
                return False
            if mask in memo:
                return memo[mask]
            if curSum == target:
                memo[mask] = helper(0, k-1, 0)
                return memo[mask]
            for i in range(index, len(nums)):
                if visited[i] == '1':
                    continue
                visited[i] = '1'
                if helper(index+1, k, curSum + nums[i]):
                    return True
                visited[i] = '0'
            memo[mask] = False
            return memo[mask]
        return helper(0, k, 0)    

class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        air_map = defaultdict(defaultdict)
        for f in flights:
            air_map[f[0]][f[1]] = f[2]
        dq = deque()
        dq.append((src, 0))
        stop = -1
        minRes = float(inf)
        visited = set()
        visited.add(src)
        while (len(dq) > 0 and stop < k) :
            cur_q_len = len(dq)
            stop+=1
            for i in range(cur_q_len):
                cur= dq.popleft()
                cur_city = cur[0]
                cur_cost = cur[1]
                aval_des = air_map[cur_city].keys()
                for d in aval_des:
                    if d == dst:
                        minRes = min(minRes, cur_cost + air_map[cur_city][d])
                        continue
                    if d in visited:
                        continue
                    visited.add(d)
                    t_cost = cur_cost + air_map[cur_city][d]
                    dq.append((d, t_cost))
        return minRes

class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        dp = [[1e9 for i in range(n)] for j in range(k + 1)]
        dp[0][src] = 0
        for i in range(1, k+2):
            for f in flights:
                dp[i][f[1]] = min(dp[i][f[1]], dp[i-1][f[0]] + f[2])
        return -1 if dp[k+1][dst] > 1e9 else dp[k+1][dst]

class Solution:
    def numTeams(self, rating: List[int]) -> int:
        asc = dsc = 0
        for i,v in enumerate(rating):
            llc = rgc = lgc = rlc =0
            for l in rating[:i]:
                if l < v:
                    llc += 1
                if l > v:
                    lgc += 1
            for r in rating[i+1:]:
                if r > v:
                    rgc += 1
                if r < v:
                    rlc += 1
            asc += llc * rgc
            dsc += lgc * rlc            
        return asc + dsc

class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        nums.sort(key=lambda x:abs(x))
        res = [x**2 for x in nums]
        return res

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counter = Counter(nums)
        res = list(counter.items())
        res.sort(key=lambda x:(x[1], x[0]), reverse=True)
        r = [res[i][0] for i in range(k)]
        return r

class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        hard_cnt = Counter(tasks).items()
        res = 0

        def check(a):
            if a % 2 == 0 or a % 3 == 0:
                return True
            if (a%2)%3 != 0 and (a%3)%2 != 0:
                return False
            return True
        
        def minTime(cnt):
            res = 0
            three = cnt // 3
            if (cnt - three * 3)%2 != 0:
                two = (cnt - three*3 + 1) // 2
                res = three - 1 + two
            else:
                res = three + (cnt - three*3) // 2
            return res

        for h, cnt in hard_cnt:
            if (not check(cnt)):
                return -1
            else:
                res+=minTime(cnt)
            
        return res

class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:

        def helper(nums, k):
            res = 0
            l = 0
            cnt = defaultdict(int)
            cur_dif = 0
            for i in range(len(nums)):
                if cnt[nums[i]] == 0:
                    cur_dif+=1
                cnt[nums[i]]+=1
                if cur_dif == k:
                    res += 1
                while (cur_dif > k and l < i) :
                    cnt[nums[l]]-=1
                    if cnt[nums[l]] == 0:
                        cur_dif-=1
                    l+=1
                res += i - l + 1
            return res
        
        return helper(nums, k) - helper(nums, k - 1)

class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        sum_array = [0 for i in range(len(nums))]
        res = 0
        sum = 0
        sum_map = defaultdict(int)
        sum_map[0] = 1
        for i, num in enumerate(nums):
            sum += num
            sum_array[i] = sum
            sum_map[sum] += 1
            for key,val in sum_map.items():
                if (sum - key) % k == 0:
                    res += val
        return val

class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        intervals.sort(key=lambda x:x[0])
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i-1][1]:
                return false
        return true

class Solution:
    def distanceBetweenBusStops(self, distance: List[int], start: int, destination: int) -> int:
        sum_arr = [0 for i in range(len(distance))]
        sum = 0
        for i, d in enumerate(distance):
            sum+=d
            sum_arr[i] = sum
        return min(sum_arr[destination] - sum_arr[start], sum_arr[-1] - sum_arr[destination] + sum_arr[start])

class Solution:
    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        dif_arr = [0 for i in range(len(s))]
        flag = false
        for i in range(len(s)):
            dif_arr[i] = abs(ord(s[i]) - ord(t[i]))
            if dif_arr[i] <= maxCost:
                flag = true
        if (not flag):
            return 0
        l, r = 0, 0
        sum = dif_arr[0]
        res = 0
        for i in range(1, len(s)):
            s += dif_arr[i]
            while sum > maxCost and l <= i:
                sum -= dif_arr[l]
                l += 1
            
            if i >= l:
                res = max(res, i - l + 1)
        return res
            
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        l, r, res = 0, 0, 0
        dic = defaultdict(int)
        dif = 0
        res = 0
        for i in range(len(s)):
            if dic[s[i]] == 0:
                dif += 1
            dic[s[i]] += 1
            while (dif > 2 and l <= i) :
                dic[s[l]] -= 1
                if (dic[s[l]] == 0):
                    dif -= 1
                l += 1
            if (i >= l) :
                res = max(i - l + 1, res)
        return res

class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda x:x[0])
        res = 1
        bound = points[1]
        for i in range(1, len(points)) :
            p = points[i]
            if (p[0] > bound):
                res+=1
                bound = p[1]
            else:
                bound = min(bound, p[1])
        return res

class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        m = defaultdict(int)
        trips.sort(lambda x:x[1])
        for t in trips:
            p = t[0]
            m[t[1]] += p
            m[t[2]] -= p
        tmp = sorted(m.items(), key = lambda x:x[0])
        cur_cap = 0
        for t in tmp:
            cur_cap += t[1]
            if cur_cap > capacity:
                return False
        return True

class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        i, j = 0, 0
        m, n = len(firstList), len(secondList)
        res = []
        while (i < m and j < n) :
            if (firstList[i][1] < secondList[j][0]):
                i+=1
            elif (secondList[j][1] < firstList[i][0]):
                j+=1
            else:
                if (firstList[i][0] <= secondList[j][1] and firstList[i][1] >= secondList[j][1]):
                    i+=1
                elif (secondList[j][0] <= firstList[i][1] and secondList[j][1] >= ):
                    j+=1
                res.append([max(secondList[j][0], firstList[i][0]), min(firstList[i][1], secondList[j][1])])
        return res

class Solution:
    def longestPalindrome(self, s: str) -> str:
        dp = [ [0 for i in range(len(s))] for i in range(len(s))]
        for i in range(len(s)):
            dp[i][i] = 1
        res = float(-inf)
        for j in range(1, len(s)):
            for i in range(j):
                if s[i] == s[j]:
                    if j - i == 1:
                        dp[i][j] = 2
                    else:
                        dp[i][j] = max(dp[i][j], dp[i+1][j-1] + 2) if  dp[i+1][j-1] > 0 else 0
                res = max(res, dp[i][j])
        return res
                    
class Solution:
    def minimumHealth(self, damage: List[int], armor: int) -> int:
        d_sort = sorted(damage)
        res = sum(d_sort) + 1
        if armor == 0:
            return res
        i = bisect.bisect_left(d_sort, armor)
        if i != len(d_sort):
            res = res - armor
        else:
            res = res - d_sort[-1]
        return res

class Solution:
    def minimalKSum(self, nums: List[int], k: int) -> int:
        counter = defaultdict(int)
        for n in nums:
            counter[n] += 1
        res = 0
        for i in range(1, 10**8 + 1):
            if k <= 0 :
                break
            if counter[i] > 0:
                continue
            counter[i] += 1
            k -= 1
            res += i
        return res

class Solution(object):
    def minWindow(self, s, t):
        count = defaultdict(int)
        for c in t:
            count[c]+=1
        l, r = 0
        minLen = float(inf)
        res, cnt = 0
        while r < len(s):
            char = s[r]
            count[char] -= 1
            if count[char] >= 0:
                cnt += 1
            while cnt == len(t):
                minLen = min(minLen, r - l + 1)
                if r - l + 1 < minLen:
                    res = [l:r+1]
                count[s[l]]+=1
                if count[s[l]] > 0:
                    cnt-=1
                l+=1
            
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        l, r = 0, 0
        temp_sum = 0
        res = float(inf)
        for i, v in enumerate(nums):
            temp_sum += v
            if temp_sum >= target:
                while l <= i and temp_sum >= target:
                    res = min(res, i - l + 1)
                    temp_sum -= nums[l]
                    l += 1
        res = 0 if res == float(inf) else res
        return res

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        dq = collections.deque()
        res = []
        for i, v in enumerate(nums):
            if  (len(dq) != 0 and i - dq[0] + 1 > k) :
                dq.popleft()
            while (len(dq) > 0 and v >= nums[dq[-1]]):
                dq.pop()
            dq.append(i)
            if i + 1 >= k:
                res.append(nums[dq[0]])
        return res

class Solution:
    def getDescentPeriods(self, prices: List[int]) -> int:
        l = 0
        pre = prices[0]
        res = 1
        t_prices = prices[1:]
        for i, v in enumerate(t_prices):
            res += 1
            if l < i and pre - v == 1:
                res += i - l
            else:
                l = i
            pre = v
        return res
        
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        res = 0
        for i in range(1, 27):
            charCnt = [0 for i in range(26)]
            difCnt = 0
            start = 0
            for p in range(len(s)):
                isValid = True
                if charCnt[ord(s[p]) - ord('a')] == 0:
                    difCnt += 1
                charCnt[ord(s[p]) - ord('a')] += 1

                while start <= p and difCnt > i:
                    charCnt[ord(s[start]) - ord('a')] -= 1
                    if charCnt[ord(s[start]) - ord('a')] == 0:
                        difCnt -= 1
                    start += 1
                
                for j in range(26):
                    if (charCnt[j] > 0 and charCnt[j] < k) :
                        isValid = False
                
                if isValid:
                    res = max(res, p - start + 1)
        return res
        
class Solution:
    def minWindow(self, s1: str, s2: str) -> str:
        m, n = len(s1), len(s2)
        start, minLen = -1, float(inf)
        dp = [[-1 for j in range(n+1)] for i in range(m+1)]
        for i in range(m + 1):
            dp[i][0] = i
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = dp[i-1][j]
            if dp[i][n] != -1:
                len = i - dp[i][n]
                if len < minLen:
                    minLen = t_len
                    start = dp[i][n]
        if start == -1:
            return ""
        else:
            return s1[start:start+minLen]

class MedianFinder:

    def __init__(self):
        self.smallTop = []
        self.largeTop = []

    def addNum(self, num: int) -> None:
        heapq.heappush(self.largeTop, -num)
        if len(self.largeTop) >= len(self.smallTop):
            heapq.heappush(self.smallTop, -heapq.heappop(self.largeTop))

    def findMedian(self) -> float:
        if len(self.smallTop) == len(self.largeTop):
            return float(self.smallTop[0] - self.largeTop[0]) / 2
        else:
            return -self.largeTop[0]

class SummaryRanges:

    def __init__(self):
        self.intervals = []

    def addNum(self, val: int) -> None:
        newInterval = [val, val]
        res = []
        insertIdx = 0
        for interval in self.intervals:
            if newInterval[0] > 1 + interval[1]:
                res.append(interval)
                insertIdx+=1
            elif newInterval[1] + 1 < interval[0]:
                res.append(interval)
            else:
                newInterval[0] = min(newInterval[0], interval[0])
                newInterval[1] = max(newInterval[1], interval[1])
        res.insert(insertIdx, newInterval)
        self.intervals = res

    def getIntervals(self) -> List[List[int]]:
        return self.intervals


# Your SummaryRanges object will be instantiated and called as such:
# obj = SummaryRanges()
# obj.addNum(val)
# param_2 = obj.getIntervals()

# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        res = []
        q =  collections.deque()
        if root:
            q.append(root)
        while len(q) > 0:
            p = q.popleft()
            if p:
                res.append(str(p.val))
                res.append(" ")
                q.append(p.left)
                q.append(p.right)
            else:
                res.append("#")
                res.append(" ")
        #res.pop()
        return "".join(res)


    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if len(data) == 0:
            return None
        ref = data.split(" ")
        val = ref[0]
        idx = 0
        root = TreeNode(int(val))
        cur = root
        res = cur
        q = collections.deque()
        q.append(cur)
        while len(q) > 0:
            t = q.popleft()
            idx+=1
            if idx >= len(ref):
                break
            if ref[idx] != '#':
                t_left_node = TreeNode(int(ref[idx]))
                q.append(t_left_node)
                t.left = t_left_node
            idx+=1
            if idx >= len(ref):
                break
            if ref[idx] != '#':
                t_right_node = TreeNode(int(ref[idx]))
                t.right = t_right_node
                q.append(t_right_node)
        return res

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        q = collections.deque()
        q.append(root)
        res = []
        while len(q) > 0:
            t_size = len(q)
            t_res = []
            for i in range(t_size):
                t = q.popleft()
                t_res.append(t.val)
                if t.left:
                    q.append(t.left)
                if t.right:
                    q.append(t.right)
            res.append(t_res)
        return res

"""
class DirectedGraphNode:
     def __init__(self, x):
         self.label = x
         self.neighbors = []
"""

class Solution:
    """
    @param graph: A list of Directed graph node
    @return: Any topological order for the given graph.
    """
    def topSort(self, graph):
        # write your code here
        res = []
        if len(graph) == 0:
            return res
        inDegree = collections.defaultdict(int)
        for node in graph:
            for n in node.neighbors:
                inDegree[n]+=1
        indegre_zero_q = collections.deque()
        for node in graph:
            if inDegree[node] == 0:
                indegre_zero_q.append(node)
                res.append(node)
        while len(indegre_zero_q) > 0:
            t_node = indegre_zero_q.popleft()
            for n in t_node.neighbors:
                inDegree[n] -= 1
                if inDegree[n] == 0:
                    indegre_zero_q.append(n)
                    res.append(n)
        return res
        
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort(key=lambda x : (x[0], -x[1]))
        dp = []
        for i in range(0, len(envelopes)):
            l = 0
            r = len(dp)
            t = envelopes[i][1]
            while l < r:
                mid = (r - l) // 2 + l
                if dp[mid] < t:
                    l = mid + 1
                else:
                    r = mid
            if r >= len(dp):
                dp.append(t)
            else:
                dp[r] = t
        return len(dp)

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        st = []
        for i in range(nums):
            l = 0
            r = len(st)
            t = nums[i]
            while l < r :
                mid = (r - l)//2 + l
                if st[mid] < t:
                    l = mid + 1
                else:
                    right = mid
            if right >= len(st):
                st.append(t)
            else:
                st[right] = t
        return len(st)

class Solution:
    def minimumHealth(self, damage: List[int], armor: int) -> int:
        max_val = float(-inf)
        sum = 0
        for d in damage:
            max_val = max(max_val, d)
            sum += d
        if armor >= max_val:
            sum = sum - max_val + 1
        else:
            sum = sum - armor + 1
        return sum

class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        m, n = len(maze), len(maze[0])
        dirs = [[0,-1],[-1,0],[0,1],[1,0]]

        def helper(sx, sy, dx, dy):
            if sx == dx and sy == dy:
                return True
            res = False
            maze[sx][sy] = -1
            for dir in dirs:
                x, y = sx, sy
                while x >= 0 and x < m and y >= 0 and y < n and maze[x][y] != 1:
                    x += dir[0]
                    y += dir[1]
                x -= dir[0]
                y -= dir[1]
                if maze[x][y] != -1:
                    res = res | helper(x, y, dx, dy)
            return res

        return helper(start[0], start[1], destination[0], destination[1])

class Solution:
    def countingCars(self, nums: List[int], query: List[int]) -> List[int]:
        maxFreq = collections.defaultdict(int)
        curMax = float(-inf)
        curFreq = 0
        dp = []
        for i in range(len(nums)-1, -1, -1):
            if nums[i] > curMax:
                curMax = nums[i]
                curFreq = 1
            elif nums[i] == curMax:
                curFreq += 1
            else:
                curFreq = curFreq
            dp.append(curFreq)
        dp.reverse()
        res = []
        for q in query:
            res.append(dp[q - 1])
        return res
    
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1 

    # def maxSumLeftToRight(self, nums: List[int]) -> List[int]:
# 2. 给一个数组int[] nums，一个数x从数组开始走到结尾，每走一步都加上对应位置的nums[i]，求x的minimum让每一步的结果都>0，算最小前缀和就解决了

    def funAna(self, words: List[str], phrase: List[str]):
        dict = collections.defaultdict(int)
        for w in words:
            wl = list(w)
            wl.sort()
            formal_str = "".join(wl)
            dict[formal_str] += 1
    
    # 第一题fun anagrams，这个题有一个坑点是phrases里面的词可能不是原先在words列表里面的词，所以会多增加一个anagram candidate。也就是说这种情况下需要 乘的数字是在words列表里的anagram数量+1

class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        dirs = [[1, 0], [1, 1], [0, -1], [-1, -1], [-1, 0], [1, -1], [0, 1], [-1, 1]]
        m, n = len(grid), len(grid[0])
        queue = collections.deque()
        queue.append([0, 0])
        level = 0
        visited = [[1 for i in range(len(grid[0]))] for j in range(len(grid))]
        while len(queue) > 0:
            t_size = len(queue)
            level += 1
            for i in range(t_size):
                t_node = queue.popleft()
                if t_node[0] == grid[m][n]:
                    return level
                for dir in dirs:
                    next_x = t_node[0] + dir[0]
                    next_y = t_node[1] + dir[1]
                    if next_x >= 0 and next_x < m and next_y >=0 and next_y < n and visited[next_x][next_y] != -1:
                        queue.append([next_x, next_y])
                        visited[next_x][next_y] = -1
        return -1  

class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if (not root.left and not root.right) or not root:
            return True

        def check(l:TreeNode, r:TreeNode):
            if not l and not r:
                return True
            if not l or not r:
                return False
            if l.val != r.val:
                return False
            return check(l.left, r.right) and check(l.right, r.left)

        return check(root.left, root.right)

    def solution(arr):
        # res = 0
        # for i in range(len(arr) - 1):
        #     flag = 0
        #     for j in range(i+1, len(arr)):
        #         if flag == 0:
        #             if arr[j] > arr[i]:
        #                 flag = 1
        #             elif arr[j] < arr[i]:
        #                 flag = -1
        #             else:
        #                 break
        #             res += 1
        #             continue
        #         if flag == 1 and arr[j] < arr[j-1]:
        #             res += 1
        #             flag = -1
        #         elif flag == -1 and arr[j] > arr[j-1]:
        #             res += 1
        #             flag = 1
        #         else:
        #             break
        # return res
        
        l = 0
        flag = 0
        res = 0
        while l < len(arr) - 1:
            j = i + 1
            while j < len(arr):
                if flag == 0:
                    if arr[j] > arr[l]:
                        flag = 1
                    elif arr[j] < arr[l]:
                        flag = -1
                    else:
                        l = j-1
                        flag = 0
                        break
                    res += j - l
                    print(l, j)
                    j+=1
                    continue
                if flag == 1 and arr[j] < arr[j-1]:
                    res += j - l
                    print(l, j)
                    flag = -1
                elif flag == -1 and arr[j] > arr[j-1]:
                    res += j - l
                    print(l, j)
                    flag = 1
                else:
                    l = j-1 
                    flag = 0
                    break
            l+=1
        return res

class Solution:
    def countSawSubarrays(arr):
        n = len(arr)

        if n < 2:
            return 0

        start = 0
        end = 1
        count = 0
        sign = arr[end] - arr[start]

        while end < n:
            while end < len and arr[end] != arr[end - 1] and isNotSameSign(arr[end] - arr[end-1], sign):
                sign = -1 * sign
                end+=1
            size = end - start
            count = count + (size * (size - 1)/2)
            start = end - 1
            end = start + 1

        return count
    
    def countSawSubarrays(arr):
        n = len(arr)
        saw = [0] * n
        totalCount = 0
        preCount = 0
        goingUp = False

        for i in range(1, len(arr)):
            curCnt = 0
            preIdx = i - 1
            if arr[i] > arr[preIdx]:
                goingUp = True
            elif arr[i] < arr[preIdx]:
                goingUp = False
            else:
                continue

            curCnt = 1

            if preIdx >= 1:
                if goingUp:
                    if arr[preIdx - 1] > arr[preIdx]:
                        curCnt = preCount + curCnt
                else:
                    if arr[preIdx - 1] < arr[preIdx]:
                        curCnt = preCount + curCnt
            
            preCount = curCnt
            totalCount += curCnt
        
        return totalCount








class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        p = [1] * n
        q = [1] * n

        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    p[i] = max(p[i], q[j] + 1)
                elif nums[i] < nums[j]:
                    q[i] = max(q[i], p[j] + 1)
        
        return max(p[-1], q[-1])
            

    def wiggleMaxLength(self, nums: List[int]) -> int:
        p, q = 1, 1
        n = len(nums)

        for i in range(1, n):
            if nums[i] > nums[i-1]:
                p = q + 1
            elif nums[i] < nums[i-1]:
                q = p + 1

        return min(n, max(p, q))    

class Solution:
    def largestBSTSubtree(self, root: Optional[TreeNode]) -> int:
        err = - 10**4 - 1

        def helper(node:TreeNode):
            if not node:
                return (err, err, 0, 0)
            if not node.left and not node.right:
                return (node.val, node.val, 1, 1)

            lmin, lmax, lcnt, lres = helper(node.left)
            rmin, rmax, rcnt, rres = helper(node.right)

            res = max(lres, rres)

            if lmax < node.val and rmin > node.val and lcnt > 0 and rcnt > 0:
                t_res = lres + rres + 1
                res = max(t_res, res)
                return 

class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        def helper(node:TreeNode, pre:TreeNode ,cnt:int) -> int:
            if not node:
                return cnt
            if pre and node.val == pre.val + 1:
                res = cnt + 1
            return max(res, helper(root.left, root, res), helper(root.right, root, res))
        return helper(root, None, 0)
        
class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        res = 0
        def helper(root:TreeNode, parent:TreeNode):
            if not root:
                return (0, 0)
            left = helper(root.left, root)
            right = helper(root.right, root)
            res = max(res, 1 + left[0] + right[1])
            res = max(res, 1 + left[1] + right[0])
            inc, dec = 0, 0
            if root.val == parent.val + 1:
                inc = max(left[0], right[0]) + 1
            if root.val == parent.val - 1:
                dec = max(left[1], right[1]) + 1
            return (inc, dec)
        helper(root, root)
        return res

class Solution:
    def isNumChr(self, c):
        if '0' <= c <= '9':
            return True
        else:
            return False
    
    def isChr(self, c):
        if 'a' <= c <= 'z':
            return True
        else:
            return False

    def decodeString(self, s: str) -> str:

        def decode(s, idx):
            res = []
            num = 0
            n = len(s)
            while idx < n:
                if self.isChr(s[idx]):
                    res.append(s[idx])
                    idx+=1
                if self.isNumChr(s[idx]):
                    num = num * 10 + ord(s[idx]) - '0'
                    idx+=1
                if s[idx] == '[':
                    idx+=1
                    temp_str, next_idx = decode(s, idx)
                    for j in range(num):
                        res.append(temp_str)
                    num = 0
                    idx = next_idx
                if s[idx] == ']':
                    break
                
            return res, idx + 1

        return decode(s, 0)

class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return
        flatten(root.left)
        flatten(root.right)
        if root.left:
            t_left = root.left
            t_right = root.right
            root.left = None
            root.right = t_left
            cur = root
            pre = root.right
            while cur:
                pre = cur
                cur = cur.right
            pre.right = t_right

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # res = []
        # n = len(candidates)
        # def backtrack(pos: int, sum: int, temp: List[int]):
        #     if sum > target:
        #         return
        #     if sum == target:
        #         res.append(temp.copy())
        #         return
        #     if pos >= len(candidates): return
        #     for i in range(pos, len(candidates)):
        #         temp.append(candidates[i])
        #         backtrack(i, sum + candidates[i], temp)
        #         temp.pop()        
        # backtrack(0, 0, [])
        # return res
    
        def recursive(nums, perm=[], res=[]):
            if not nums:
                res.append(perm.copy())
            
            for i in range(len(nums)):
                newNums = nums[:i] + nums[i+1:]
                perm.append(nums[i])
                recursive(newNums, perm, res)
                perm.pop()
            
            return res
        
        return recursive(candidates)

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []

        def generate(left, right, tStr):
            if left > right or left < 0 or right < 0:
                return
            
            if left == 0 and right == 0:
                res.append(tStr)
            
            if left > 0:
                generate(left - 1, right, tStr + '(')
            
            if right > 0:
                generate(left, right - 1, tStr + ')')
            
            return 
        
        generate(n, n, '')

        return res

class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        left, right = 0, 0
        for i in range(len(s)):
            if s[i] == '(':
                left += 1
            
            if left == 0 and s[i] == ')':
                right += 1
            elif s[i] == ')':
                left -= 1
            
        
        res = []
        
        def isValid(s):
            cnt = 0
            for i in range(len(s)):
                if s[i] == '(':
                    cnt += 1
                elif s[i] == ')':
                    cnt -= 1
                if cnt < 0:
                    return False
            return True
        
        def helper(s: str, start: int, left: int, right:int):
            if left == 0 and right == 0 and isValid(s):
                res.append(s)
                return

            for i in range(start, len(s)):
                if left > 0 and s[i] == '(':
                    helper(s[:i]+s[i+1:], i, left - 1, right)
                if right > 0 and s[i] == ')':
                    helper(s[:i]+s[i+1:], i, left, right - 1)
        
        helper(s, 0, left, right)

        return res

    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []
        
        def isValid(s):
            if len(s) > 0 and int(s) <= 255:
                if len(s) > 1 and s[0] == '0':
                    return False
                return True
            return False

        def helper(s, section, temp):
            if len(s) == 0 and section == 4:
                res.append(temp)
            
            for i in range(1, 4):
                t = s[:i]
                if i <= len(s) and isValid(t):
                    if section == 3:
                        helper(s[i:], section+1, temp+t)
                    else:
                        helper(s[i:], section+1, temp+t+'.')
                else:
                    break
        
        helper(s, 0, '')
        
        return res

class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []

        def helper(start, temp_l):
            if len(temp_l) >= 2:
                res.append(temp_l.copy())

            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i-1]:
                    continue
                temp_l.append(nums[i])
                helper(i+1, temp_l)
                temp_l.pop()
        
        helper(0, [])

        return res


class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        n = len(s)
        
        dp = [[-1 for i in range(n)] for j in range(n)]

        for i in range(0, n):
            for j in range(0, i + 1):
                if s[i] == s[j] and (i - j <= 2 or dp[j+1][i-1] > 0):
                    dp[j][i] = 1

        def helper(pos:int, temp_list:List[str]):
            if pos >= n:
                res.append(temp_list.copy())
                return

            for i in range(pos, n):
                if dp[pos][i] > 0:
                    temp_list.append(s[pos:i+1])
                    helper(i+1, temp_list)
                    temp_list.pop()
        
        helper(0, [])

        return res

class Solution:
    def checkPartitioning(self, s: str) -> bool:
        n = len(s)
        
        dp = [[-1 for i in range(n)] for j in range(n)]

        for i in range(0, n):
            for j in range(0, i + 1):
                if s[i] == s[j] and (i - j <= 2 or dp[j+1][i-1] > 0):
                    dp[j][i] = 1
            
        for i in range(0, n -1):
            for j in range(i+1, n):
                if dp[0][i] > 0 and dp[i+1][j] > 0 and dp[j+1][n] > 0:
                    return True
        
        return False

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        max_word_len = max(wordDict, key=len, default='')
        dict_set = set(wordDict)
        res = []

        def backtrack(start:int, temp_l:List[str]):
            if start >= len(s):
                res.append(" ".join(temp_l.copy()))
                return
            
            boundary = min(start + max_word_len, len(s)+1)

            for i in range(start, boundary):
                temp_substr = s[start:i+1]
                if temp_substr in dict_set:
                    temp_l.append(temp_substr)
                    backtrack(i+1, temp_l)
                    temp_l.pop()
        
        backtrack(0, [])

        return res

    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        max_word_len = max(wordDict, key=len, default='')
        dict_set = set(wordDict)
        memo = defaultdict(list)
        res = []

        @lru_cache(maxsize=None)
        def _backtrack_topdown(s):
            if not s:
                return [[]]

            if s in memo:
                return memo[s]
            
            boundary = min(max_word_len, len(s))

            for i in range(boundary):
                temp_substr = s[:i+1]
                if temp_substr in dict_set:
                    for subsentence in _backtrack_topdown(s[i+1:]):
                        memo[s].append(temp_substr + subsentence)
            
            return memo[s]
        
        _backtrack_topdown(s)

        return [" ".join(words) for words in memo[s]]
    
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        dict_set = set(wordDict)
        dp = [[]] * (len(s) + 1)
        dp[0] = ['']

        for endIndex in range(1, len(s) + 1):
            sublist = []
            for startIndex in range(endIndex):
                temp_substr = s[startIndex:endIndex]
                if temp_substr in dict_set:
                    for subsentence in dp[startIndex]:
                        sublist.append(subsentence + ' ' + temp_substr)
            
            dp[endIndex] = sublist

        return dp[len(s)]

class Solution:
    def wordPatternMatch(self, pattern: str, s: str) -> bool:
        pattern_map = defaultdict(str)
        pattern_set = set()

        def helper(pos_p:int, pos_s:int):
            if pos_p == len(pattern) and pos_s == len(s):
                return True
            if pos_p == len(pattern) or pos_s == len(s):
                return False
            cur_c = pattern[pos_p]
            for i in range(pos_s+1, len(s) + 1):
                temp_substr = s[i:pos_s+1]
                if cur_c in pattern_map and temp_substr == pattern_map[cur_c]:
                    if helper(pos_p+1, i):
                        return True
                elif cur_c not in pattern_map or pattern_map[cur_c] != "":
                    if temp_substr in pattern_set:
                        continue
                    pattern_map[cur_c] = temp_substr
                    pattern_set.add(temp_substr)

                    
                    if helper(pos_p+1, i):
                        return True
                    pattern_map[cur_c] = ""
                    pattern_set.remove(temp_substr)
            return False
        
        return helper(0, 0)

class SolutionT126:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        res = []
        word_set = set(wordList)
        if endWord not in word_set:
            return res
        bfs_q = deque()
        temp_ladder = [beginWord]
        bfs_q.append(temp_ladder.copy())
        level, min_level = 1, float(-inf)
        used_words = set()
        while len(bfs_q) > 0:
            cur_ladder = bfs_q.popleft()
            if len(cur_ladder) > level:
                for w in cur_ladder:
                    word_set.remove(w)
                used_words.clear()
                level = len(cur_ladder)
                if level > min_level:
                    break
            cur_word = cur_ladder[-1]
            for i in range(len(cur_word)):
                changed_word = list(cur_word)
                for j in range(26):
                    cur_char = changed_word[i]
                    cur_chage = chr(ord(cur_char) + j)
                    changed_new_word = "".join(changed_word)
                    if changed_new_word not in word_set:
                        continue
                    nextPath = cur_ladder.copy()
                    nextPath.append(changed_new_word)
                    if changed_new_word == endWord:
                        res.append(nextPath)
                        min_level = level #bfs
                    else:
                        bfs_q.append(nextPath)
        return res

class Solution:
    def countDaysTogether(self, arriveAlice: str, leaveAlice: str, arriveBob: str, leaveBob: str) -> int:
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        prefix_month_days = []
        for i in range(len(month_days)):
            if i > 1:
                prefix_month_days.append(month_days[i-1] + prefix_month_days[i-1])
            elif i == 1:
                prefix_month_days.append(month_days[0])
            else:
                prefix_month_days.append(0)

        arrA_str = arriveAlice.split('-')
        leaA_str = leaveAlice.split('-')
        arrB_str = arriveBob.split('-')
        leaB_str = leaveBob.split('-')

        arrA_days = prefix_month_days[int(arrA_str[0]) - 1] + int(arrA_str[1])
        leaA_days = prefix_month_days[int(leaA_str[0]) - 1] + int(leaA_str[1])
        arrB_days = prefix_month_days[int(arrB_str[0]) - 1] + int(arrB_str[1])
        leaB_days = prefix_month_days[int(leaB_str[0]) - 1] + int(leaB_str[1])

        if arrB_days > leaA_days or arrA_days > leaB_days:
            return 0
        
        if (arrB_days <= arrA_days and leaB_days >= leaA_days) or (arrA_days <= arrB_days and leaA_days >= leaB_days):
            return min(leaA_days - arrA_days + 1, leaB_days - arrB_days + 1)

        if leaA_days >= leaB_days:
            return leaB_days - arrA_days+1
        else:
            return leaA_days - arrB_days+1

class Solution:
    def matchPlayersAndTrainers(self, players: List[int], trainers: List[int]) -> int:
        players.sort()
        trainers.sort()
        res = 0
        index_t = 0

        for p in players:
            if index_t >= len(trainers):
                break

            if p <= trainers[index_t] and index_t < len(trainers):
                res+=1
                index_t+=1
            else:
                while index_t < len(trainers) and trainers[index_t] < p:
                    index_t+=1
                
                if index_t < len(trainers) and trainers[index_t] >= p:
                    index_t+=1
                    res+=1
        
        return res

class Solution:
    def smallestSubarrays(self, nums: List[int]) -> List[int]:
        last = [0] * 32
        n = len(nums)
        res = [0] * 32
        for i in range(n - 1, -1, -1):
            for j in range(32):
                if nums[i] & (1 << j): # whether the jth digit at nums[i] is 1
                    last[j] = i
            res[i] = max(1, max(last) - i + 1)
        return res    

class Solution:
    def latestTimeCatchTheBus(self, buses: List[int], passengers: List[int], capacity: int) -> int:
        passengers.sort()
        buses.sort()

        bus_pas = defaultdict([])

        pas_idx = 0

        for b in buses:
            cur_dep = b
            cur_cap = capacity
            while pas_idx < len(passengers) and passengers[pas_idx] <= cur_dep and cur_cap > 0:
                cur_cap-=1
                pas_idx+=1
                bus_pas[b].append(passengers[pas_idx])

class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        res = []
        bfs_q = collections.deque()
        for i in range(n):
            if i == 0:
                for j in range(1, 10):
                    bfs_q.append(j)
                continue
            
            temp_len = len(bfs_q)

            for p in range(temp_len):
                cur_num = bfs_q.pop()
                last_digit = cur_num % 10
                if last_digit + k >=0 and last_digit + k <= 9:
                    bfs_q.append(cur_num * 10 + last_digit + k)
                if last_digit - k >=0 and last_digit - k <= 9:
                    bfs_q.append(cur_num * 10 + last_digit - k)
            
            if i == n - 1:
                res = list(bfs_q)
        
        return res

class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        monoSt = []
        fRes = []
        for i in range(len(temperatures)-1, -1, -1):
            if len(monoSt) == 0:
                monoSt.append(i)
                fRes.append(0)
                continue

            while len(monoSt) > 0 and temperatures[monoSt[-1]] <= temperatures[i]:
                monoSt.pop()

            if len(monoSt) == 0:
                monoSt.append(i)
                fRes.append(0)
            
            else:
                fRes.append(monoSt[-1] - i)
                monoSt.append(i)
        
        return reversed(fRes)

class Solution:
    def maxTaskAssign(self, tasks: List[int], workers: List[int], pills: int, strength: int) -> int:
        tasks.sort()
        workers.sort()
        res = 0
        w_idx = len(workers) - 1
        unworkable = []
        #怎么决定 谁吃pill？如果能不吃药就完成的工作，就不吃药
        #怎么标记 一些worker已经被用掉？栈 + idx？

        #tasks 1 3 3 worker 0 3 3 pill 1 strength 1
        #怎么省？ 按照上面的决定策略，1会用到3，但是1应该用pill, 让后面的3，3满足
        #从后向前？

        #遍历task 递增单调栈

        # for t in range(len(tasks)):
        #     while w_idx < len(workers) and workers[w_idx] < tasks[t]:
        #         unworkable.append(workers[w_idx])
        #         w_idx += 1

        #从后向前 [10,15,30], workers = [0,10,10,10,10], pills = 3, strength = 10, 会用0去补10
        for t in range(len(tasks)-1, -1, -1):
            if workers[]

class RandomizedSet:

    def __init__(self):
        self.my_set = defaultdict(int)
        self.my_arr = []

    def insert(self, val: int) -> bool:
        if val in self.my_set:
            return false
        else:  
            self.my_arr.append(val)
            self.my_set[val] = len(self.my_arr) - 1
            return true

    def remove(self, val: int) -> bool:
        if val in self.my_set:
            pre_idx = self.my_set[val]
            last_val = self.my_arr[-1]
            self.my_arr[-1], self.my_arr[pre_idx] = self.my_arr[pre_idx], self.my_arr[-1]
            self.my_arr.pop()
            self.my_set[last_val] = pre_idx
            return true
        else:
            return false
        
    def getRandom(self) -> int:
        return self.my_arr[random.randint(0, len(self.my_arr)-1)]

class RandomizedCollection:

    def __init__(self):
        self.my_set = defaultdict(set(int))
        self.my_arr = []

    def insert(self, val: int) -> bool:
        if val in self.my_set:
            self.my_arr.append([val])
            self.my_set[val].insert(len(self.my_arr) - 1)
            return False
        else:  
            self.my_arr.append([val])
            self.my_set[val] = len(self.my_arr) - 1
            return True

    def remove(self, val: int) -> bool:
        if val in self.my_set:
            pre_idx = self.my_set[val]
            last_list = self.my_arr[-1]
            last_val = last_list[0]
            self.my_arr[pre_idx].pop()
            if len(self.my_arr[pre_idx] > 0):
                return True
            else:
                self.my_arr[-1], self.my_arr[pre_idx] = self.my_arr[pre_idx], self.my_arr[-1]
                self.my_arr.pop()
                self.my_set[last_val] = pre_idx
                del self.my_set[val]
                return True
        else:
            return False
        
    def getRandom(self) -> int:
        return self.my_arr[random.randint(0, len(self.my_arr)-1)]

class Solution:
    def findSubarrays(self, nums: List[int]) -> bool:
        sum_set = set()
        for i in range(len(nums)-1):
            t_sum = nums[i] + nums[i+1]
            if t_sum in sum_set:
                return True
            else:
                sum_set.add(t_sum)
        return False

class Solution:
    def maximumRows(self, matrix: List[List[int]], numSelect: int) -> int:
        m, n = len(matrix), len(matrix[0])
        row_sum = [0 * m]
        res = 0
        back_up = []
        for i in range(m):
            for j in range(n):
                row_sum[i] += matrix[i][j]
            if row_sum[i] == 0:
                res += 1
            elif row_sum[i] == numSelect:
                t_bu = []
                for p in range(len(matrix[i])):
                    if matrix[i][p] == 1:
                        t_bu.append(p)
                back_up.append(t_bu)

class Solution:
    def findSmallestRegion(self, regions: List[List[str]], region1: str, region2: str) -> str:
        parents = {region[i]:region[0] for region in regions for i in range(1, len(region))}
        ancestor_history = {region1}
        while region1 in parents:
            region1 = parents[region1]
            ancestor_history.add(region1)
        while region2 not in ancestor_history:
            region2 = parents[region2]
        return region2

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right\

class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        bfs_q = collections.deque()
        level = 1
        res = 1
        temp_sum = root.val
        bfs_q.append(root)
        while(len(bfs_q) > 0):
            t_len = len(bfs_q)
            level += 1
            level_sum = 0
            for i in range(t_len):
                t_node = bfs_q.popleft()
                level_sum += t_node.val
                if t_node.left:
                    bfs_q.append(t_node.left)
                if t_node.right:
                    bfs_q.append(t_node.right)
            if level_sum > temp_sum:
                temp_sum = level_sum
                res = level
        return level

class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        res = []
        for i in range(len(nums)):
            while nums[i] != nums[nums[i] - 1]:
                val = nums[i]
                ano = nums[val - 1]
                nums[i], nums[val - 1] = ano, val
        
        for i in range(len(nums)):
            if nums[i] != i+1:
                res.append(nums[i])
        return res
        
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        #layer
        for i in range(n/2):
            for j in range(n - 1 - i):
                temp = matrix[i][j]
                matrix[i][j] = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j]
                matrix[n - 1 - i][n - 1- j] = matrix[j][n - 1 - i]
                matrix[j][n - 1 - i] = temp

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        m = len(matrix)
        n = len(matrix[0])
        if m == 0 or n == 0:
            return res
        loop = (n+1)//2 if m > n else (m+1)//2
        rLen, cLen = m, n
        for i in range(loop):
            for col in range(i, i + cLen):
                res.append(matrix[i][col])
            for row in range(i + 1, i + rLen):
                res.append(matrix[row][i + cLen - 1])
            if rLen == 1 or cLen == 1:
                break
            for col in range(i + cLen - 2, i - 1, -1):
                res.append(matrix[rLen - 1 - i][col])
            for row in range(i + rLen - 2, i, -1):
                res.append(matrix[row][i])

            rLen -= 2
            cLen -= 2

        return res

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(-1)
        dummy.next = head
        ptr = dummy
        num = 0
        while ptr.next != None:
            ptr = ptr.next
            num+=1
        pre = dummy
        while num >= k:
            cur = pre.next
            for i in range(k):
                t = cur.next
                cur.next = t.next
                t.next = pre.next
                pre.next = t
            pre = cur
            num -= k
        return dummy.next

class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        dummyLarge = ListNode(-1)
        dummyLarge.next = head
        lessTail = ListNode(-1)
        newStart = lessTail
        pre = dummyLarge
        while (pre.next != None) :
            if pre.next.val < x:
                lessTail.next = pre.next
                pre.next = pre.next.next
                lessTail = lessTail.next
                lessTail.next = None
            else:
                pre = pre.next
        lessTail.next = dummyLarge.next
        return newStart.next

class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dummy = ListNode(-1)
        pre = dummy
        dummy.next = head
        for i in range(left - 1):
            pre = pre.next
        cur = pre.next
        for i in range(left, right):
            t = cur.next
            cur.next = t.next
            t.next = pre.next
            pre.next = t
        return dummy.next


class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        nodeMap = defaultdict(Node)
        cur = head
        while cur != None:
            t = Node(cur.val)
            nodeMap[cur] = t
            cur = cur.next
        secondCur = head
        while secondCur != None:
            nodeMap[secondCur].next = nodeMap[secondCur.next]
            nodeMap[secondCur].random = nodeMap[secondCur.random]
            secondCur = secondCur.next
        return nodeMap[head]

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False for i in range(n + 1)]
        word_dict = defaultdict(wordDict)
        wordSplit = s.split()
        dp[0] = True
        for i in range(n):
            if dp[i] == False:
                continue
            for j in range(i+1, n+1):
                if dp[i] == True and "".join(wordSplit[i:j+1] in word_dict):
                    dp[j] == True
        return dp[n]

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False for i in range(n + 1)]
        # i is the length
        word_dict = set(wordDict)
        dp[0] = True
        #possible length iterate
        for i in range(n+1):
            for j in range(i):
                if dp[j] == True and s[j:i+1] in word_dict:
                    dp[i] = True
                    break
        return dp[n]

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordSet = set(wordDict)
        n = len(s)
        memo = [False * n]

        def dfs(pos:int):
            if pos >= len(s):
                return True
            if memo[pos] == True:
                return memo[pos]
            for i in range(pos + 1, n+1):
                if s[pos:i] in wordSet and dfs(i):
                    memo[pos] = True
                    return True
            memo[pos] = False
            return False
        dfs(s, 0)
        return memo[0]

class Solution: 
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:  
        if not head or not head.next: 
            return head 
        fast, slow, pre = head, head, head 
        while fast and fast.next:
            pre = slow
            slow = slow.next
            fast = fast.next.next
        pre.next = None

        return self.merge(self.sortList(head), self.sortList(slow))
    
    def merge(self, l, r):
        dummy = ListNode(-1)
        cur = dummy
        while (l != None and r != None):
            if l.val < r.val:
                cur.next = l
                l = l.next
            else:
                cur.next = r
                r = r.next
            cur = cur.next
        if l:
            cur.next = l
        if r:
            cur.next = r
        return dummy.next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(-1)
        dummy.next = head
        cur = head
        while cur.next:
            t = cur.next
            cur.next = t.next
            t.next = dummy.next
            dummy.next = t
        return dummy.next

class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head
        st = []
        while fast.next != None and fast.next.next != None:
            st.append(slow.val)
            slow = slow.next
            fast = fast.next.next
        if fast.next == None:
            st.pop()
        while len(st) > 0:
            if slow.val != st.pop():
                return false
            slow = slow.next
        return True

class Node:
    def __init__(self, val=0, left=None, right=None, random=None):
        self.val = val
        self.left = left
        self.right = right
        self.random = random

class Solution:
    def __init__(self) -> None:
        self.visited = {}

    def copyRandomBinaryTree(self, root: 'Optional[Node]') -> 'Optional[NodeCopy]':
        nodeMap = defaultdict(Node)
        if root == None:
            return None
        ptr = root
        st = []
        while ptr or len(str) > 0:
            if ptr:
                generNode = Node(ptr.val)
                nodeMap[ptr] = generNode
                st.append(ptr)
                ptr = ptr.left
            else:
                ptr = st.pop()
                ptr = ptr.right
        
        ptr = root
        st = []

        while ptr or len(str) > 0:
            if ptr:
                if ptr.left:
                    nodeMap[ptr].left = nodeMap[ptr.left]
                if ptr.right:
                    nodeMap[ptr].right = nodeMap[ptr.right]
                if ptr.random:
                    nodeMap[ptr].random = nodeMap[ptr.random]
                st.append(ptr)
                ptr = ptr.left
            else:
                ptr = st.pop()
                ptr = ptr.right
        
        return nodeMap[root]

class Solution:
    def __init__(self):
        self.visited = {}
    def copyRandomBinaryTree(self, node: 'Node') -> 'NodeCopy':
        if not node:
            return node
        if node in self.visited:
            return self.visited[node]
        clone_node = NodeCopy(node.val)
        self.visited[node] = clone_node
        clone_node.left = self.copyRandomBinaryTree(node.left)
        clone_node.right = self.copyRandomBinaryTree(node.right)
        clone_node.random = self.copyRandomBinaryTree(node.random)

        return clone_node

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_map = defaultdict(int)
        res = 0
        for n in nums:
            if n in num_map:
                continue
            left = num_map[n - 1] if (n - 1) in num_map else 0
            right = num_map[n + 1] if (n + 1) in num_map else 0
            sum = left + 1 + right
            num_map[n] = sum
            res = max(res, sum)
            num_map[n - left] = sum
            num_map[n + right] = sum
        return res
        
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        tSum = 0
        sum_map = defaultdict(int)
        res = 0
        for n in nums:
            tSum += N
            sum_map[tSum] += 1
            if (tSum - k) in sum_map:
                res += sum_map[tSum - k]
        return res

class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        res = []
        smallHalf = [] #max heap
        largeHalf = nums[:k] #min heap
        heapq.heapify(largeHalf)
        while len(smallHalf) < len(largeHalf):
            heapq.heappush(smallHalf, -heapq.heappop(largeHalf))
        
        removals = collections.Counter()

        idx = k - 1
        while idx < len(nums):
            res.append((largeHalf[0] - smallHalf[0]) * 0.5 if k % 2 == 0 else - smallHalf[0])
            idx += 1

            if idx == len(nums):
                break
            
            out_num = nums[i - k]
            in_num = nums[i]

            balance = 0
            balance += -1 if out_num <= -smallHalf[0] else 1
            removals[out_num] += 1

            #out num doesn't move out at this time

            # only add one item

            #balance just decide to push to where
            if smallHalf and in_num <= -smallHalf[0]:
                balance += 1
                heapq.heappush(smallHalf, -in_num)
            else:
                balance -= 1
                heapq.heappush(largeHalf, in_num)

            # adjust two halves
            if balance < 0:
                heapq.heappush(smallHalf, -heapq.heappop(largeHalf))
                balance += 1
            if balance > 0:
                heapq.heappush(largeHalf, -heapq.heappop(smallHalf))
                balance -= 1
            
            #lazy removal
            #only when the removal one is on the top, we remove it, since it doesn't influence the calculation of median, if it is not on the top, we use a dict to record which item should be removed
            while smallHalf and removals[-smallHalf[0]]:
                removals[-smallHalf[0]] -= 1
                heapq.heappop(smallHalf)
                
            while largeHalf and removals[largeHalf[0]]:
                removals[largeHalf[0]] -= 1
                heapq.heappop(largeHalf)   

        return res       

class Solution:
    def reorganizeString(self, s: str) -> str:
        res = []
        charCounter = collections.Counter()
        for c in s:
            charCounter[c] += 1
        
        (tk, tv) = charCounter.most_common()
        if tv > (len(s) + 1) // 2:
            return ""
        
        pq = [(-v, k) for k, v in charCounter.items()]
        heapq.heapify(pq)

        pre_char, pre_cnt = '', 0

        while pq:
            cur_cnt, cur_char = heapq.heappop(pq)
            res += [cur_char]
            cur_cnt += 1
            if pre_cnt < 0:
                heapq.heappush((pre_cnt, pre_char))
            pre_cnt, pre_char = cur_cnt, cur_char
        
        res = ''.join(res)
        return res

class Solution:
    def isValid(self, s: str) -> bool:
        parenthesesMap = {')':'(', ']':'[', '}':'{'}
        left = parenthesesMap.values()
        st = []
        for c in s:
            if c in left:
                st.append(c)
            elif c in parenthesesMap:
                if st[-1] != parenthesesMap[c]:
                    return False
                st.pop()
        return len(st) == 0
            
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        m, n = len(matrix), len(matrix[0])
        st = [0] * n
        res = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "1":
                    st[j] += 1
            t = self.help(st)
            res = max(res, t)
        return res
    
    def help(self, nums:List[int]) -> int:
        nums.append(0)
        st = []
        res = 0
        for i in range(len(nums)):
            while len(st) > 0 and nums[i] <= nums[st[-1]]:
                height = nums[st[-1]]
                st.pop()
                width = i - st[-1] - 1 if len(st) > 0 else i
                res = max(res, height * width)
            st.append(i)
        return res

    def shortestSubsequence(arr, K):
        n = len(arr)
        reminders = {}
        sum = [0] * n
        min_len = float(inf)
        for i in range(n):
            sum[i] = (sum[i-1] if i > 0 else 0) + arr[i]
            r = sum[i] % K
            if r in reminders:
                min_len = min(min_len, i - reminders[r])
            reminders[r] = i
        return min_len

class Solution:
    def calculate(self, s: str) -> int:
        num = 0
        sumSt = []
        res = 0
        n = len(s)

        flag = 1

        idx = 0

        while idx < n:
            if s[idx].isdigit():
                num = num*10 + int(s[idx])
            elif s[idx] in ['-', '+']:
                res += flag * num
                num = 0
                flag = [-1, 1][s[idx] == '+']
            elif s[idx] == '(':
                sumSt.append(res)
                sumSt.append(flag)
                res = 0
            elif s[idx] == ')':
                res += flag * num
                res *= sumSt.pop()
                res += sumSt.pop()
                num = 0
        
        return res + num * flag

class Solution:
    def calculate(self, s: str) -> int:
        st = []
        num = 0
        flag = '+'
        for c in s:
            if c.isdigit():
                num = num*10 + int(c)
            elif c in ['-', '+', '*', '/']:
                if flag == '+':
                    st.append(num)
                elif flag == '-':
                    st.append(-num)
                elif flag == '*':
                    t = st.pop()
                    st.append(t * num)
                elif flag == '/':
                    t = st.pop()
                    r = abs(t) // num
                    st.append(r if num > 0 else -r)
                num = 0
                flag = c
        return sum(st)

    def calculate(self, s: str) -> int:
        num = 0
        res = 0
        sign = 1
        st = []

        idx = 0
        while idx < len(s):
            if s[idx].isdigit():
                num = num * 10 + int(s[idx])
            elif s[idx] in ['-', '+']:
                res += sign * num
                flag = -1 if s[idx] == '-' else 1
                num = 0
            elif s[idx] == '(':
                st.append(res)
                st.append(flag)
                res = 0
            elif s[idx] == ')':
                res += flag * num
                res *= st.pop()
                res += st.pop()
                num = 0

            idx += 1
        
        return res + flag * num

class Solution:
    def decodeString(self, s: str) -> str: 
        num = 0
        str = []
        numSt = []
        strSt = []
        idx = 0
        while idx < len(s):
            if s[idx].isdigit():
                num = num * 10 + int(s[idx])
            elif s[idx].isalpha():
                str.append(s[idx])
            elif s[idx] == '[':
                numSt.append(num)
                strSt.append("".join(str))
                str = []
            elif s[idx] == ']':
                tNum = numSt.pop()
                tStr = "".join(str)
                str = []
                str.append(strSt.pop())
                for i in range(tNum):
                    str.append(tStr)
        return "".join(str)

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        st = []
        for i in range(len(nums)):
            if len(st) == 0 or nums[i] > st[-1]:
                st.append(nums[i])
            else:
                l = 0
                r = len(st)
                while l < r:
                    mid = (r - l) // 2 + l
                    if st[mid] < nums[i]:
                        l += 1
                    else:
                        r = mid
                st[l] = nums[i]
        return len(st)

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        dq = collections.deque()
        res = []
        for i in range(len(nums)):
            while len(dq) > 0 and i - dq[0] >= k:
                dq.popleft()
            while len(dq) > 0 and nums[i] >= nums[dq[-1]]:
                dq.pop()
            dq.append(i)
            res.append(dp[0])
        return res

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
            #forward 0, 1, 2, ..., i - 1
            for k in range(i+1):
                if k == i and sum[i] <= M:
                    dp[i] = max_in_sub[0][i]
                elif i > k and (sum[i] - sum[i - k - 1]) <= M:
                    dp[i] = min(dp[i], max_in_sub[i-k][i] + dp[i - k - 1])
        
        return dp[-1]


class Solution:
    def nextLargerNodes(self, head: Optional[ListNode]) -> List[int]:
        st = []
        p = head
        cnt = 0
        nums = []
        while p:
            cnt+=1
            nums.append(p.val)
            p = p.next

        res = [0] * cnt
        ptr = head
        idx = 0
        while ptr:
            while len(st) > 0 and ptr.val > nums[st[-1]]:
                cur_idx = st.pop()
                res[cur_idx] = ptr.val
            st.append(idx)
            ptr = ptr.next
            idx += 1
        
        return res


class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        root = [0 for i in range(n)]
        for i in range(n):
            root[i] = i
        
        def findRoot(root:List[int], p):
            while p != root[p]:
                p = root[p]
            return p
        
        for edge in edges:
            p = findRoot(root, edge[0])
            q = findRoot(root, edge[1])
            if p != q:
                root[p] = q
        
        print(root)

        return len(root)

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if (len(grid) == 0):
            return 0
        row, col = len(grid), len(grid[0])
        self.count = sum(grid[i][j] == "1" for i in range(row) for j in range(col))
        self.root = [i for i in range(row * col)]

        def union(x, y):
            pRoot = findRoot(x)
            qRoot = findRoot(y)
            if pRoot == qRoot:
                return
            self.root[qRoot] = pRoot
            self.count -= 1

        def findRoot(p):
            while p != self.root[p]:
                p = self.root[p]
            return p
        
        for i in range(row):
            for j in range(col):
                if grid[i][j] == '0':
                    continue
                idx = i * row + j

class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        self.root = [i for i in range(m * n)]
        self.count = 0
        self.grid = [[0 for i in range(m)] for j in range(n)]
        res = []
    
        def union(x, y):
            pRoot = findRoot(x)
            qRoot = findRoot(y)
            if pRoot == qRoot:
                return
            self.root[qRoot] = pRoot
            self.count -= 1

        def findRoot(p):
            while p != self.root[p]:
                p = self.root[p]
            return p
        
        def checkBoundary(x, y):
            if x >= 0 and x < m and y >=0 and y < n:
                return True
            else:
                return False
        
        dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        for p in positions:
            self.grid[p[0]][p[1]] = 1
            self.count += 1
            for i in range(4):
                newX = p[0] + dir[i][0]
                newY = p[1] + dir[i][1]
                if checkBoundary(newX, newY) and self.grid[newX][newY] == 1:
                    union(newX * m + newY, p[0] * m + p[1])
            res.append(self.count)
        
        return res

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        startTimes = sorted([t[0] for t in intervals])
        endTimes = sorted([t[1] for t in intervals])
        endIdx = 0
        cnt = 0
        res = 0
        for i in range(len(startTimes)):
            cnt += 1
            while (endTimes[endIdx] <= startTimes[i]):
                endIdx += 1
                cnt -= 1
            res = max(res, cnt)
        return res
    
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        map = collections.defaultdict(int)
        for interval in intervals:
            map[interval[0]] += 1
            map[interval[1]] -= 1
        res = 0
        cnt = 0
        calculated = list(map.items())
        calculated.sort(key=lambda i:i[0])
        for k, v in calculated:
            cnt += v
            res = max(res, cnt)
        return res
    
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        pq = []
        intervals.sort(key=lambda i:i[0])
        res = 0
        for interval in intervals:
            while len(pq) > 0 and interval[0] >= pq[0]:
                heapq.heappop(pq)
            heapq.heappush(pq, interval[1])
            res = max(res, len(pq))
        return res

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        l, r, sum = 0, 0, 0
        res = len(nums) + 1
        while r < len(nums):
            while r < len(nums) and sum < target:
                sum += nums[r]
                r += 1
            while l < r and sum >= target:
                res = min(r - l + 1, res)
                sum -= nums[l]
                l += 1
        return res if res < len(nums) + 1 else 0

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        sum = [0 for i in range(n + 1)] #max index is n
        res = n + 1
        # i -> sum[i + 1]
        for i in range(n + 1):
            sum[i + 1] = nums[i] + sum[i]
        
        #find index, the first one great or equal
        def find(left, right, target):
            while left < right:
                mid = sum[(right - left) // 2 + left]
                if mid < target:
                    left = mid + 1
                else:
                    right = mid
            return right

        for i in range(n + 1):
            #iterate from 0 elements to n elments, index is 0, 1, 2, ... n
            right = find(i + 1, n, sum[i] + target)
            if right == n and sum[n] - sum[i] < target:
                break
            res = min(res, right - i)
        
        return res

class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        n = len(nums)
        left = min(nums)
        right = max(nums)
        sum = [0 for i in range(n + 1)]
        while right - left >= 1e-5:
            minSum = 0
            mid = (right - left) /2 + left
            check = False
            for i in range(1, n + 1):
                sum[i] = sum[i - 1] + nums[i - 1] - mid
                if i - k >= 0 :
                    minSum = min(sum[i - k], minSum)
                if i - k >= 0 and sum[i] - minSum >= 0:
                    check = True
                    break
            
            if check:
                left = mid
            else:
                right = mid
        return left

class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        res = 0
        st = []
        mod = 10**9 + 7
        INF = float(inf)
        nums = [-INF] + arr + [-INF]
        for i in range(len(nums)):
            while len(st) > 0 and nums[i] < nums[st[-1]]:
                end = st[-1]
                st.pop()
                nextEnd = st[-1]
                #子数组个数 （end - nextEnd) 
                res += nums[end] * (end - nextEnd) * (i - end)
            st.append(i)
    
    def sumSubarrayMins(self, arr: List[int]) -> int:
        res = 0
        mod = 10**9 + 7
        #递增栈
        st = []
        INF = float(inf)
        A = [-INF] + arr + [-INF]
        for k, v in enumerate(A):
            while st and v < A[st[-1]]:
                j = st[-1]
                st.pop()
                i = st[-1]
                # 更更小a 更小i 小j    新值k
                # 以arr[j]为最小的子数组的个数 (j - i) * (k - j)
                # 可以理解成 以j为起点，左边取一个，右边从1 取到最大 k - j
                # 然后左边取两个，...， 左边最多取（j - i)个
                res += A[j] * (j - i) * (k - j)
            st.append(k)
        return res % mod

class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:
        st = []
        INF = float(inf)
        arr = [-INF] + nums + [-INF] 
        res = 0
        for i, v in enumerate(arr):
            while len(st) > 0 and v < arr[st[-1]]:
                end = st[-1]
                st.pop()
                new_end = st[-1]
                res -= arr[end] * (end - new_end) * (i - end)
            st.append(i)

        st = []
        arr = [INF] + nums + [INF]
        for i, v in enumerate(arr):
            while len(st) > 0 and v > arr[st[-1]]:
                end = st[-1]
                st.pop()
                new_end = st[-1]
                res += arr[end] * (end - new_end) * (i - end)
            st.append(i)
        return res

class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        root = [i for i in range(n)]

        def union(x, y):
            rootX = findRoot(x)
            rootY = findRoot(y)
            if rootX == rootY:
                return False
            else:
                root[rootY] = rootX
                return True

        def findRoot(i):
            while i != root[i]:
                i = root[i]
            return i
        
        for d in dislikes:
            if not union(d[0], d[1]):
                return False
            
        return True

class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        root = [i for i in range(n)]

        def findRoot(i):
            while i != root[i]:
                i = root[i]
            return i

        for i in range(len(graph)):
            x = i
            y = graph[i][0]
            rootX = findRoot(x)
            rootY = findRoot(y)

            for j in range(1, len(graph[i])):
                rootTemp = findRoot(graph[i][j])
                if rootTemp == rootX:
                    return False
                root[rootTemp] = rootY

        return True

class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        n = len(board)
        cell = [None] * (n**2 + 1)
        columns = list(range(0, n))
        label = 1
        for row in range(n-1, -1, -1):
            for column in columns:
                cell[label] = (row, column)
                label += 1
            columns.reverse()
        q = deque([1])
        dist = [-1] * (n**2 + 1)
        while q:
            cur = q.popleft()
            for next in range(cur + 1, min(cur + 6, n**2) + 1):
                row, col = cell[next]
                destination = (board[row][col] if board[row][col] != -1 else next)
                if dist[destination] == -1:
                    dist[destination] = dist[cur] + 1
                    q.append(destination)
        return dist[n**2]

class Solution:
    def rankTeams(self, votes: List[str]) -> str:
        ranking = [[] for i in range(26)]
        for s in votes:
            for index, c in enumerate(list(s)):
                ranking[index].append(c)
        
        ranked = set()
        res = []

        for i in range(len(ranking)):
            characters = list(Counter(ranking[i]).items())
            characters.sort(key=lambda c:(c[1], c[0]))
            for c_tuple in characters:
                if c_tuple[0] not in ranked:
                    res.append(c_tuple[0])
        
        return "".join(res)

class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        common_interval = []
        s1_idx, s2_idx = 0, 0
        while s1_idx < len(slots1) and s2_idx < len(slots2):
            s1_itv = slots1[s1_idx]
            s2_itv = slots2[s2_idx]

            while s1_itv[0] >= s2_itv[1]:
                s2_idx += 1
                s2_itv = slots2[s2_idx]

            while s2_itv[0] >= s1_itv[1]:
                s1_idx += 1
                s1_itv = slots1[s1_idx]
            
            # if s1_itv[0] <= s2_itv[0] and s1_itv[1] >= s2_itv[1]:
            #     common_interval.append(s2_itv)
            #     s1_idx += 1
            #     s2_idx += 1
            #     continue

            # if s2_itv[0] <= s1_itv[0] and s2_itv[1] >= s1_itv[1]:
            #     common_interval.append(s1_itv)
            #     s1_idx += 1
            #     s2_idx += 1
            #     continue

            new_itv = [max(s1_itv[0], s2_itv[0]), min(s1_itv[1], s2_itv[1])]
            common_interval.append(new_itv)
            s1_idx += 1
            s2_idx += 1
        
        for c in common_interval:
            if c[1] - c[0] >= duration:
                return c
        
        return []

class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        common_interval = []
        s1_idx, s2_idx = 0, 0
        while s1_idx < len(slots1) and s2_idx < len(slots2):
            s1_itv = slots1[s1_idx]
            s2_itv = slots2[s2_idx]

            while s1_itv[0] >= s2_itv[1] and s2_idx < len(slots2):
                s2_idx += 1
                if s2_idx >= len(slots2):
                    break
                s2_itv = slots2[s2_idx]

            while s2_itv[0] >= s1_itv[1] and s1_idx < len(slots1):
                s1_idx += 1
                if s1_idx >= len(slots1):
                    break
                s1_itv = slots1[s1_idx]
            
            if s1_idx >= len(slots1) or s2_idx >= len(slots2):
                return []
            
            if s1_itv[0] <= s2_itv[0] and s1_itv[1] >= s2_itv[1]:
                common_interval.append(s2_itv)
                s2_idx += 1
                continue

            if s2_itv[0] <= s1_itv[0] and s2_itv[1] >= s1_itv[1]:
                common_interval.append(s1_itv)
                s1_idx += 1
                continue

            new_itv = [max(s1_itv[0], s2_itv[0]), min(s1_itv[1], s2_itv[1])]
            common_interval.append(new_itv)
            if s1_itv[1] <= s2_itv[1]:
                s1_idx += 1
            if s1_itv[1] >= s2_itv[1]:
                s2_idx += 1
        
        for c in common_interval:
            if c[1] - c[0] >= duration:
                return [c[0], c[0] + duration]
        
        return []
        
class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        i, j = 0, 0
        m, n = len(slots1), len(slots2)
        intersect_interval = []
        slots1.sort()
        slots2.sort()

        while i < m and j < n:
            if slots1[i][1] <= slots2[j][0]:
                i += 1
            elif slots2[j][1] <= slots2[i][0]:
                j += 1
            else:
                if slots1[i][0] <= slots2[j][1] and slots1[i][1] >= slots2[j][1]:
                    tmp_itv = [max(slots1[i][0], slots2[j][0]), min(slots1[i][1], slots2[j][1])]
                    if tmp_itv[1] - tmp_itv[0] >= duration:
                        return [tmp_itv[0], tmp_itv[0] + duration]
                    intersect_interval.append([max(slots1[i][0], slots2[j][0]), min(slots1[i][1], slots2[j][1])])
                    j += 1
                else:
                    if tmp_itv[1] - tmp_itv[0] >= duration:
                        return [tmp_itv[0], tmp_itv[0] + duration]
                    intersect_interval.append([max(slots1[i][0], slots2[j][0]), min(slots1[i][1], slots2[j][1])])
                    i += 1
        
        return []

class Solution:
    def minIncrementForUnique(self, nums: List[int]) -> int:
        nums.sort()
        add_set = set()
        rest = []
        for n in nums:
            if n not in add_set:
                add_set.add(n)
            else:
                rest.append(n)
        
        idx = 0
        res = 0

        for i in range(rest[0] + 1, 10**5 + 1):
            if idx < len(rest) and i not in add_set and i > rest[idx]:
                add_set.add(i)
                res += i - rest[idx]
                idx += 1
                if idx >= len(rest):
                    break
        
        return res

class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        residual_map = defaultdict(int)
        res = 0
        for t in time:
            residual = t % 60
            if 60 - residual in residual_map:
                res += residual_map[60 - residual]
            residual_map[residual] += 1
        return res

class Solution:
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        if startFuel >= target:
            return 0
        
        #错误的greedy 策略，每次选，当前可达范围内最大的一个station

        #错误样例 
        # 1000
        # 299
        # [[13,21],[26,115],[100,47],[225,99],[299,141],[444,198],[608,190],[636,157],[647,255],[841,123]]

        #永远停留在440，到不了更远，中间加几次正向油，推进一下

        stations.sort()
        cur_range = startFuel
        next_max_stat = 0
        next_max_range = startFuel

        idx = 0
        res = 0
        
        while idx < len(stations):
            while idx < len(stations) and stations[idx][0] <= cur_range:
                tmp_next_range = stations[idx][0] + cur_range - stations[idx][0] + stations[idx][1]
                if tmp_next_range >= target:
                    return res + 1
                if next_max_range <= tmp_next_range:
                    next_max_range = tmp_next_range
                    next_max_stat = idx
                idx += 1

            if next_max_range <= cur_range:
                return -1
            
            res += 1
            idx = next_max_stat + 1
            cur_range = next_max_range

        return -1 if cur_range < target else 0
            
class Solution:
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        stations.sort()
        dp = [startFuel] + 0 * len(stations)

        for idx, (location, fuel) in enumerate(stations):
            for t in range(idx, -1, -1):
                if dp[t] >= location:
                    dp[t + 1] = max(dp[t+1], dp[t] + fuel)
        
        for i, d in enumerate(dp):
            if d >= target:
                return i
            
        return -1

class Solution:
    def maxPerformance(self, n: int, speed: List[int], efficiency: List[int], k: int) -> int:
        res = 0
        mod = 10 ** 9 + 7

        for i in range(len(efficiency)):
            bench_eff = efficiency[i]
            temp_speed_sum = speed[i]
            team_mem_cnt = 1

            heap = []
            for j in range(len(speed)):
                if efficiency[j] >= bench_eff and j != i:
                    heapq.heappush(heap, -speed[j])
                
            while team_mem_cnt < k:
                new_speed = -heapq.heappop()
                temp_speed_sum += new_speed
            
            res = max(res, (temp_speed_sum * bench_eff) % mod)
        
        return res

class Solution:
    def maxPerformance(self, n: int, speed: List[int], efficiency: List[int], k: int) -> int:
        mod = 10 ** 9 + 7
        member = zip(efficiency, speed)
        member = sorted(member, key=lambda k:-k[0])

        heap = []
        res = 0

        max_speed_sum = 0
        for i in range(len(member)):
            bench_eff = member[i][0]
            cur_speed = member[i][1]

            # when we have iterated k person
            # the first k iteration will include all members since the previous efficieny is higher and the amount is less than k
            if i > k - 1:
                max_speed_sum -= heapq.heappop(heap)
            heapq.heappush(heap, cur_speed)

            # this is a must, since we based on this guy
            max_speed_sum += cur_speed
            res = max(res, max_speed_sum * bench_eff)
        
        return res % mod

class Solution:
    def maxPerformance(self, n: int, speed: List[int], efficiency: List[int], k: int) -> int:
        modulo = 10 ** 9 + 7

        # build tuples of (efficiency, speed)
        candidates = zip(efficiency, speed)
        # sort the candidates by their efficiencies
        candidates = sorted(candidates, key=lambda t:t[0], reverse=True)

        speed_heap = []
        speed_sum, perf = 0, 0
        for curr_efficiency, curr_speed in candidates:
            # maintain a heap for the fastest (k-1) speeds
            if len(speed_heap) > k-1:
                speed_sum -= heapq.heappop(speed_heap)
            heapq.heappush(speed_heap, curr_speed)

            # calculate the maximum performance with the current member as the least efficient one in the team
            speed_sum += curr_speed
            perf = max(perf, speed_sum * curr_efficiency)

        return perf % modulo

class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = [[0 for i in range(len(text1) + 1)] for j in range(len(text2) + 1)]
        
        for i in range(1, len(text1) + 1):
            for j in range(1, len(text2) + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = max(dp[i - 1][j - 1] + 1, dp[i][j])
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j - 1])
        
        return dp[-1][-1]

class Solution:
    def maximumSubsequenceCount(self, text: str, pattern: str) -> int:
        pattern_set = set(list(pattern))
        pre_first = 0
        after_second = 0

        for t in text:
            if t == pattern[1]:
                after_second += 1

        total_second = after_second
        res = 0

        for t in text:
            if t in pattern_set:
                if t == pattern[0]:
                    pre_first += 1
                    if pre_first > 0:
                        res += after_second
                else:
                    after_second -= 1
                # insert.append((pre_first, after_second))
        
        max_insert = max(pre_first, total_second)

        return res + max_insert


    def solve(nums: List[int], k):
        number_set = set(nums)
        nums.sort()
        res = 0
        for i in range(len(nums)):
            cur = nums[i]
            rest = k - cur
            if rest < 0:
                continue
            else:
                return

class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        res = []
        idx = 0
        enumerate
        email_dic = collections.defaultdict()
        for acc in accounts:
            acc_list = acc[1:]
            merge = False
            merge_idx = -1
            for email in acc_list:
                if email in email_dic:
                    merge = True
                    merge_idx = email_dic[email]
            if merge:
                for email in acc_list:
                    if email not in email_dic:
                        email_dic[email] = merge_idx
                        res[merge_idx].append(email)
            else:
                res.append(acc)
                for email in acc_list:
                    email_dic[email] = idx
                idx += 1
        res.sort(key=lambda x:x[0])
        for r in res:
            r = r[0] + sorted(r[1:])
        return res

class UF:
    def __init__(self, N) -> None:
        self.parents = list(range(N))
    def find_root(self, node):
        while self.parents[node] != node:
            node = self.parents[node]
        return node
    def union(self, child, parent):
        self.parents[self.find_root(child)] = self.parents[self.find_root(parent)]
    
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        uf = UF(len(accounts))
        # UF 是用来找 用户的合并关系的

        email_userIdx_map = collections.defaultdict()
        # map 是用来 存储email和用户的归属关系的
        for idx, (_, *emails) in enumerate(accounts):
            for email in emails:
                if email in email_userIdx_map:
                    uf.union(idx, email_userIdx_map[email])
                else:
                    email_userIdx_map[email] = idx
        
        ans = collections.defaultdict(list)
        for email, idx in email_userIdx_map.items():
            ans[uf.find_root(idx)].append(email)
        res = []
        for idx, emails in ans.items():
            res.append([accounts[idx][0]] + sorted(emails))
        return res

class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        map = collections.defaultdict(int)
        l, maxCnt = 0, 0
        res = 0
        for r in range(len(s)):
            char = s[r]
            map[char] += 1
            maxCnt = max(maxCnt, map[char])
            if r - l + 1 - maxCnt <= k:
                res = max(r - l + 1, res)
            else:
                map[s[l]] -= 1
                l += 1
        return res

class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        charCnt = collections.defaultdict(int)
        l, res = 0, 0
        dif = 0
        for r in range(len(s)):
            if charCnt[s[r]] == 0:
                dif += 1
            charCnt[s[r]] += 1
            while dif > k:
                charCnt[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
        return res

class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        l, res, maxCnt = 0, 0, 0
        curCnt = 0
        for r in range(len(nums)):
            if nums[r] == 1:
                curCnt += 1
                maxCnt = max(maxCnt, curCnt)
            # if r - l + 1 - maxCnt < k:
            #     res = max(res, r - l + 1)
            # else:
            #     if nums[l] == 1:
            #         curCnt -= 1
            #     l += 1
            while r - l + 1 - maxCnt 
        return res
    
class Solution:
    def predictTheWinner(self, nums: List[int]) -> bool:
        n = len(nums)
        dp = [[0 for i in range(n)] for i in range(n)]
        #dp[i][j] from i - j, player 1 
        #dp[i][j] =  dp[i-1][j], dp[i][j-1]
        for i in range(n):
            dp[i][i] = nums[i]
        for len in range(1, n):
            for j in range(len, n):
                return i,j

class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        course_pre_counter = collections.defaultdict(int)
        course_top = collections.defaultdict(list)
        taken = set()
        untaken = set()
        for rel in relations:
            course_pre_counter[rel[1]] += 1
            course_top[rel[0]].append(rel[1])
            untaken.add(rel[0])
            untaken.add(rel[1])
        total = len(untaken)
        round = 0
        while len(taken) != total:
            temp_taken = []
            for c in untaken:
                if course_pre_counter[c] == 0:
                    temp_taken.append(c)
            if len(temp_taken) == 0:
                return -1
            for c in temp_taken:
                for post in course_top[c]:
                    course_pre_counter[post] -= 1
                taken.add(c)
                untaken.remove(c)
            round += 1
        return round
    
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        bfs_queue = collections.deque(int)
        temp_res = [[] for i in range(-101, 101)]
        if root:
            bfs_queue.append((0, root))
        while len(bfs_queue) > 0:
            temp_len = len(bfs_queue)
            for i in range(temp_len):
                idx, node = bfs_queue.popleft()
                temp_res[idx].append(node.val)
                if node.left:
                    bfs_queue.append((idx - 1, node.left))
                if node.right:
                    bfs_queue.append((idx + 1, node.right))
        res = []
        for i in range(-101, 101):
            if len(temp_res[i]) > 0:
                res.append(temp_res[i])
        return res
