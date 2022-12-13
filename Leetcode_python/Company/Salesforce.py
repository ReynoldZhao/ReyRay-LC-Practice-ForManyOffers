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

class Solution1:
    def closestNumbers(self, n: int, arr: List[int]):
        arr.sort()
        minimumDiff = float('inf')
        for i in range(len(arr) - 1):
            minimumDiff = min(minimumDiff, abs(arr[i + 1] - arr[i]))
        res = []
        for i in range(len(arr) - 1):
            if abs(arr[i + 1] - arr[i]) == minimumDiff:
                res.append(arr[i])
                res.append(arr[i + 1])

class Solution:
    def breakPalindrome(self, palindrome: str) -> str:
        if (len(palindrome) <= 1):
            return ""
        res = ""
        halfStr = palindrome[: len(palindrome)//2 + 1] if len(palindrome) % 2 == 0 else palindrome[: len(palindrome)//2 ]
        flag = False
        idx = 0
        for i in range(len(halfStr)):
            if halfStr[i] != "a":
                flag = True
                idx = i
                break
        if not flag:
            # all a
            return palindrome[:len(palindrome) - 1] + "b"
        else:
            return palindrome[:idx] + "a" + palindrome[idx+1:]

class SolutionT39:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        n = len(candidates)
        def backtrack(pos: int, sum: int, temp: List[int]):
            if sum == target:
                res.append(temp.copy())
                return
            if pos >= len(candidates): return
            for i in range(pos, len(candidates)):
                sum += candidates[i]
                temp.append(candidates[i])
                backtrack(i, sum + candidates[i], temp)
                temp.pop()
                sum -= candidates[i]          
        backtrack(0, 0, [])
        return res

class SolutionT216:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        if n > k * 9 or n < k * 1: return []
        dict = [i for i in range(1, 10)]
        res = []
        def backtrack(sum: int, k: int, i: int, temp:List[int]):
            if sum > n:
                return
            if sum == n and k == 0:
                res.append(temp.copy())
                return
            for j in range(i, 10):
                backtrack(sum + j, k - 1, j, temp + [j])
        backtrack(0, k, 1, [])
        return res

class Solution:
    def findMin(self, nums: List[int]) -> int:
        l , r = 0, len(nums) - 1
        while l < r:
            mid = (r - l) // 2 + l
            if nums[mid] < nums[r]:
                mid = r
            elif nums[mid] > nums[r]:
                mid = l + 1
        return nums[r]



class SolutionT162:
    def findPeakElement(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        nums.insert(0, float('-inf'))
        nums.append(float('-inf'))
        while l < r:
            mid = (r - l) // 2 + l
            if nums[mid] < nums[mid + 1]:
                l = mid + 1
            else:
                r = mid
        return mid

    def findLocalMinimum(sef, nums: List[List[int]]) ->int:
        s_r, s_c = 0, 0
        m, n = len(nums), len(nums[0])

        def isOut(r, c):
            if r < 0 or r >= m or c < 0 or c >=n:
                return False
            return True
    
        res = []
        while s_r < m and s_c < n:
            cur = nums[s_r][s_c]
            
        
        return res

class SolutionT1102:
    def maximumMinimumPath(self, grid: List[List[int]]) -> int:
        heap = []
        m, n = len(grid), len(grid[0])
        dir = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        visited = set()
        heapq.heappush(heap, (-grid[0][0], 0, 0))
        visited.add(0)

        while heap:
            val, x, y = heapq.heappop(heap)
            if x == m - 1 and y == n - 1:
                return -val
            for i in range(4):
                tx = x + dir[i][0]
                ty = y + dir[i][1]
                if tx < 0 or tx >= m or ty < 0 or ty >= n or (tx * n + ty in visited): continue
                visited.add(tx * n + ty)
                heapq.heappush(heap, (min(grid[tx][ty], -val), tx, ty))
        return -1

class SolutionT826:
    def maxProfitAssignment(self, difficulty: List[int], profit: List[int], worker: List[int]) -> int:
        com = zip(difficulty, profit)
        temp_dif = [d[0] for d in com]
        worker.sort()
        res = 0

        i, best = 0, 0

        for w in worker:
            while i < len(com) and w >= com[i][0]:
                best = max(best, com[i][1])
                i+=1
            res+=best
        return res

class SolutionT1283:
    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        nums.sort()
        l, r = 0, nums[-1]
        def calc(d):
            sum = 0
            for n in nums:
                if n % d == 0: sum += int(n // d)
                else : sum += int (n // d) + 1
            return sum
        while l < r:
            mid = (r - l) // 2 + l
            if calc(mid) > threshold:
                l = mid + 1
            else:
                r = mid
        return r


