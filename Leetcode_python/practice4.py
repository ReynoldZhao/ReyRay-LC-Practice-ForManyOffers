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



        
        

        
        







        
            









            









