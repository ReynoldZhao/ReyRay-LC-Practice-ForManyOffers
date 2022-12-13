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
import numpy as np

class Solution:
    def countingPersons(self, statementCounts: List[int], statements: List[List[int]]):
        comments = collections.defaultdict(list)
        sum = 0
        n = len(statementCounts)
        for i in range(len(statementCounts)):
            for s in statements[sum : sum + statementCounts[i] - 1]:
                comments[i+1].append(s)
            sum += statementCounts[i]
        
        res = 0
        for i in range(len(1, n + 1)):
            queue = collections.deque()
            flag = True
            bucket = [-1] * (n + 1)
            bucket[i] = 1
            queue.append(i)
            # person in queue is always good
            while len(queue) > 0:
                t = queue.popleft()
                for com in comments[t]:
                    #good said truth: this person is bad
                    if com[1] == 0:
                        # conflict
                        if bucket[com[0]] == 1:
                            flag = False
                            break

                        bucket[com[0]] = 0
                        continue

                    #good said truth: this person is good
                    idx = com[0]
                    # not conflict
                    if bucket[idx] == -1:
                        bucket[idx] = 1
                        queue.append(idx)

                    # conflict
                    if bucket[idx] == 0:
                        flag = False
                        break
                if not flag:
                    break
            if flag:
                res += 1
        return res

class Solution1:
    def chooseAFlask(self,n : int, requirements: List[int], markings: List[List[int]]):
        map = collections.defaultdict(List)
        requirements.sort()
        for m in markings:
            map[m[0]].append(m[1])
        res = 0
        waste = float('inf')
        for k, v in map.items():
            v.sort()
            sum = 0
            if v[-1] < requirements[-1]: continue
            for r in requirements:
                idx = bisect.bisect_left(v, r)
                sum += v[idx] - r
            if sum < waste:
                waste = sum
                res = k
        return res

class Solution2:
    def distanceMetric(self, nums: List[int]):
        map = collections.defaultdict(list)
        n = len(nums)
        left = [0] * n
        right = [0] * n
        for i in range(n):
            if nums[i] not in map : 
                left[i] = 0
                map[nums[i]] = [1, i]
            else :
                temp = map[nums[i]]
                left[i] = temp[0] * (i - temp[1]) + left[temp[1]]
                map[nums[i]] = [temp[0] + 1, i]
        map.clear()
        for i in range(n - 1, -1, -1):
            if nums[i] not in map : 
                right[i] = 0
                map[nums[i]] = [1, i]
            else :
                temp = map[nums[i]]
                right[i] = temp[0] * (math.abs(i - temp[1])) + right[temp[1]]
                map[nums[i]] = [temp[0] + 1, i]
        res = []
        for i in range(n) :
            res.append(left[i] + right[i])
        return res

class Solution3:

    def countSentences(self, wordSet: List[str], sentences: List[str]):
        map = collections.defaultdict(int)

        def encodeKey(word: str):
            t = [0] * 26
            for c in word:
                t[ord(c) - ord('a')]+=1
            key_s = []
            for i in range(26):
                key_s.append('#')
                key_s.append(str(t[i]))
            key_s = "".join(key_s)
            return key_s

        for word in wordSet:
            key_s = encodeKey(word)
            map[key_s]+=1

        res = []
        for sentence in sentences:
            temp = 1
            sentence = sentence.split()
            for word in sentence:
                key_s = encodeKey(word)
                temp *= map[key_s]
            res.append(temp)
        return res

class Solution4:
    def numOfways(self, n : int):
        if n == 1 or n == 2: return 1
        if n == 3 : return 2
        count = 0

        def helper(n: int, count: int):
            if n == 0:
                count = (count + 1) % (1e9 + 7)
            if n >= 1:
                helper(n - 1, count)
            if n >= 3:
                helper(n - 3, count)
        
        helper(n - 3, count)
        helper(n - 1, count)
        return count % (1e9 + 7)
    
    def numOfways(self, n : int):
        dp = [0] * (n + 1)
        dp[1] = dp[2] = 1
        dp[3] = 2
        for i in range(4, n + 1):
            dp[i] = (dp[i - 1] % (1e9 + 7)) + (dp[i - 3] % (1e9 + 7))
            dp[i] = dp[i] % (1e9 + 7)
        return dp[n]

class Solution4:
    def numOfWays(self, n : int):
        #q = np.array([[3, 2, 1], [2, 1, 1], [1, 1, 0]])
        q = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0]])
        res = self.mat_pow(q, n)
        return res[0][0]
    
    def mat_pow(self, a: List[List[int]], n: int):
        ret = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0]])
        while n > 0 :
            if ((n & 1) == 1) :
                ret = np.dot(ret, a)
            n = n >> 1
            a = np.dot(a, a)
        return ret
    
    def numOfWays1(self, n : int):
        dp = [0] * (n + 1)
        dp[1] = dp[2] = 1
        dp[3] = 2
        for i in range(4, n + 1):
            dp[i] = (dp[i - 1] % (1e9 + 7)) + (dp[i - 3] % (1e9 + 7))
            dp[i] = dp[i] % (1e9 + 7)
        return int(dp[n])

class Soltuion5:
    def carParkingRoof(self, cars: List[int], k: int):
        cars.sort()
        res = float('inf')
        for i in range(k - 1, len(cars)):
            res = min(res, cars[i] - cars[i - k - 1] + 1)
        return res

class Solution6:
    def sumValues(self, parents: List[int], startPoint: List[int], jumpLength: List[int]):
        maxJump = 0
        for jump in jumpLength:
            maxJump = max(maxJump, jump)
        #  dpParent[i][j] jumpL为i，第j号的parent，
        dpParent = [[0 for i in range(len(parents))] for j in range(maxJump)]
        # dp[i][j] -> jumpLength 是 i，从第j号开始的 sum
        dp = [[0 for i in range(len(parents))] for j in range(maxJump)]
        for i in range(len(dpParent)):
            for j in range(len(dpParent[0])):
                #jumpL = 1, dpP 直接放 parent
                if i == 0:
                    dpParent[i][j] = parents[j]
                    print(dpParent[i][j])
                    idx = max(dpParent[i][j], 0)

                    # 计算dp sum，j（当前值） + dp[i][idx] jumpL 为i，爹为idx的sum 
                    dp[i][j] = j + dp[i][idx]
                    continue
                #j 位置为0
                if j == 0:
                    dpParent[i][j] = -1
                else :
                    # dpParent[i][j] jumpL为i，第j号的parent
                    # = pParent[i - 1][dpParent[0][j]] jumpL为 i-1，第dpParent[0][j]的爹
                    # dpParent[0][j]其实就是当前位置只跳一层level的爹
                    dpParent[i][j] = dpParent[i - 1][dpParent[0][j]]
                idx = max(dpParent[i][j], 0)
                dp[i][j] = j + dp[i][idx]
        res = []
        for i in range(len(startPoint)):
            res.append(dp[jumpLength[i] - 1][startPoint[i]])
        return res

class Solution7:
    def kAnagrams(self, s: str, l: List[str], k: int) -> List[str]:
        cnt = collections.Counter(s)
        res = []
        for t in l:
            cnt_t = collections.Counter(t)
            if sum((cnt - cnt_t).values()) <= k:
                res.append(t)
        return res

        










        


        