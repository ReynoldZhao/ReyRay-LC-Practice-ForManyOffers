from _typeshed import Self
from cmath import pi
import collections
import heapq
from typing import Collection, List, Optional
from collections import *
import sys


class Solution:
    def biS(self, nums, target):
        l, r = 0, len(nums)
        while (l < r):
            mid = l + (r - l) // 2
            if (nums[mid] < target):
                l = mid + 1
            else:
                r = mid
        return r

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        lb = self.biS(nums, target)
        ub = self.biS(nums, target+0.5)
        if (lb == ub) or (lb == -1 or ub == -1):
            return [-1, -1]
        else:
            return [lb, ub-1]

class Solution:
    
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        t_max = 0
        for i in range(len(piles)):
            t_max = max(t_max, piles[i])

        def check(e):
            res = 0
            for p in piles:
                res += p//e if (p%e == 0) else p//e + 1
            if res <= h:
                return True
            else:
                return False

        l, r = 1, t_max
        while (l < r):
            mid = l + (r-l)//2
            if (not check(mid)):
                l = mid + 1
            else:
                r = mid
        return r
        
class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        def helper(nestedInt, depth):
            res = 0
            for ni in nestedInt:
                if ni.isInteger():
                    res += ni.getInteger() * depth
                else:
                    res += helper(ni.getList(), depth+1)
            return res
        return helper(nestedList, 1)
                
    def depthSumInverse(self, nestedList: List[NestedInteger]) -> int:       
        # def getDep(nestedList):
        #     dep = 1
        #     for ni in nestedList:
        #         if (not ni.isInteger()):
        #             return dep + getDep(ni)
        
        def getMaxDep(nestedList):
            temp_max = 1
            for ni in nestedList:
                if (not ni.isInteger()):
                    t_max_dep = 1 + getMaxDep(ni.getList())
                    temp_max = max(temp_max, t_max_dep)
            return temp_max

        max_dep =  getMaxDep(nestedList)

        def helper(nestedInt, depth, max_dep):
            res = 0
            for ni in nestedInt:
                if ni.isInteger():
                    res += ni.getInteger() * (max_dep - depth + 1)
                else:
                    res += helper(ni.getList, depth+1, max_dep)
            return res
        
        return helper(nestedList, 1, max_dep)


class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
            def countNum(s):
                cOne, cZero = 0, 0
                for i in range(len(s)):
                    if s[i] == '1':
                        cOne += 1
                    elif s[i] == '0':
                        cZero += 1
                    else:
                        continue
                return (cZero, cOne)

            memo = defaultdict(int)

            def helper(tempList, pos, m, n):
                temp_res = -float(inf)
                for i in range(pos, len(strs)):
                    ts = strs[i]
                    z, o = countNum(ts)
                    if z > m or o > n:
                        continue
                    r1 = 1 + helper(tempList + [ts], i + 1, m-z, n-o)
                    r2 = helper(tempList, i+ 1, m, n)
                    temp_res = max(temp_res, max(r1, r2))
                if temp_res ==  -float(inf):
                    temp_res = 0
                return temp_res

            z, o = countNum(strs[0])
            if z > m or o > n:
                return 0
            r1 = helper([0], 1,  m - z, n - o)
            r2 = helper([], 1, m, n)

            return max(1 + r1, r2)

    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        def countNum(s):
            cOne, cZero = 0, 0
            for i in range(len(s)):
                if s[i] == '1':
                    cOne += 1
                elif s[i] == '0':
                    cZero += 1
                else:
                    continue
            return (cZero, cOne)
        
        countMap = defaultdict((int))
        for s in strs:
            countMap[s] = countNum(s)

        dp = [[0 for i in range(n + 1)] for i in range(m + 1)]

        for i in range(len(strs)):
            z, o = countNum(strs[i])
            for p in range(m, 0, -1):
                for q in range(n, 0, -1):
                    if z > p or o > q:
                        continue
                    dp[p][q] = max(dp[p][q], 1 + dp[p - z][q - o])
        
        return dp[m][n]

class Solution:
    def subarraysWithMoreZerosThanOnes(self, nums: List[int]) -> int:
        #以i结尾的0，1数量
        oneL, zeroL = [0 for i in range(len(nums + 1))], [0 for i in range(len(nums + 1))]
        res = [0 for i in range(len(nums + 1))]

        if nums[0] == 1:
            oneL[1] = 1
            res[1] = 1
        else:
            zeroL[1] = 1

        for i in range(1, len(nums)):
            if nums[i] == 1:
                oneL[i + 1] = oneL[i-1] + 1
                zeroL[i] = zeroL[i-1]
            else:
                oneL[i] = oneL[i-1]
                zeroL[i] = zeroL[i-1] + 1
        
        #res[i] 以i结尾，1 > 0 的subarray 数量
        t = 0
        for i in range(1, len(nums)):
            if oneL[i-1] > zeroL[i-1]:
                res[i] = res[i-1]
            for j in range(i-1, -1, -1):
                if oneL[i] - oneL[j] > zeroL[i] - zeroL[j]:
                    t +=1
        
        # oneL, zeroL = [0 for i in range(len(nums))], [0 for i in range(len(nums))]
        # res = [0 for i in range(len(nums))]

        # if nums[0] == 1:
        #     oneL[0] = 1
        #     res[0] = 1
        # else:
        #     zeroL[1] = 1

        # for i in range(1, len(nums)):
        #     if nums[i] == 1:
        #         oneL[i + 1] = oneL[i-1] + 1
        #         zeroL[i] = zeroL[i-1]
        #     else:
        #         oneL[i] = oneL[i-1]
        #         zeroL[i] = zeroL[i-1]

class Solution:
    def subarraysWithMoreZerosThanOnes(self, nums: List[int]) -> int:
        MOD = 1000000007
        sum = 0
        dp = [0, 0]
        #how many end pos have the n 1 more than 0
        mp = defaultdict(int)
        mp[0] = 1
        ans = 0
        for num in nums:
            tdp = dp
            if num == 1:
                sum += 1
            else:
                sum -= 1
            dp[0] = mp[sum]
            if num == 1:
                dp[1] = (tdp[0] + tdp[1] + 1) % MOD
            else:
                dp[1] = (tdp[1] - dp[0] + MOD) % MOD
            mp[sum]+=1
            ans = (ans + dp[1]) % MOD
        return ans
    
    def subarraysWithMoreZerosThanOnes(self, nums: List[int]) -> int:
        diff_sum = 0 # count of 1 - count of zero
        MOD = 1000000007
        dp, res = 0, 0

        mp = defaultdict(int)
        mp[0] = 1 #从-1位开始

        for num in nums:
            if num == 1:
                diff_sum += 1
                dp = dp + mp[diff_sum - 1] #dp[i] = dp[i-1] + 
            else:
                diff_sum -= 1
                dp = dp - mp[diff_sum]
            res = (res + dp) % MOD
            mp[diff_sum] += 1
        
        return res % MOD
    
    def subarraysWithMoreZerosThanOnes(self, nums: List[int]) -> int:
        MOD = 10** 9 + 7
        counts = collections.Counter({0:1})
        res = s = dp = 0
        for n in nums:
            if n:
                s += 1
                dp += counts[s - 1]
            else:
                s -= 1
                dp -= counts[s]
            res = (res + dp) % MOD
            counts[s] += 1
        
        return res % MOD

    def subarraysWithMoreZerosThanOnes(self, nums: List[int]) -> int:
        nums = [0] + [i if i == 1 else -1 for i in nums]
        n = len(nums)
        for i in range(1, n):
            nums[i] += nums[i-1]
        
        cnt = Counter()
        res, pre, precnt, mod = 0, 0, 0, int(10**9+7)
        for cur in nums:
            if cur == pre + 1:
                precnt += cnt[pre] % mod
            else:  # cur == pre - 1
                precnt -= cnt[pre-1] % mod
            res += precnt
            cnt[cur] += 1
            pre = cur
            
        return res % mod
    
    def subarraysWithMoreZerosThanOnes(self, nums: List[int]) -> int:
        count = defaultdict(int)
        diff_sum = 0
        dp = 0
        count[0] = 1
        res = 0
        MOD = 10** 9 + 7

        for num in nums:
            if num:
                diff_sum += 1
                dp = dp + count[diff_sum - 1] #dp[n-1]的肯定都能用 因为是1
        #count[diff_sum - 1] 指 some j --- n这部分的，由于新增一个1，新增的子数组
            else:
                diff_sum -= 1
                dp = dp - count[diff_sum] #dp[n-1]的不一定全部能用，
        #count[diff_sum] 指 some j --- n这部分，由于新增一个0，
            res = (res + dp) % MOD
        
        return res

class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        calMap = defaultdict(defaultdict)
        for i in range(len(equations)):
            dividend = equations[i][0]
            divisor = equations[i][1]
            calMap[dividend][divisor] = values[i]
            calMap[divisor][dividend] = 1 / values[i]
        
        res = []

        # def bfs(dvd, dvs, last, past_set):
        #     res = -1.0
        #     t_iter = calMap[dvd]
        #     if dvs in t_iter:
        #         res = last * calMap[dvd][dvs]
        #     else:
        #         for k,v in t_iter:
        #             if k in past_set:
        #                 continue
        #             else:
        #                 t_set = past_set
        #                 t_set.add(k)
        #                 ret = bfs(k, dvs, last * v, t_set)
        #                 if t != -1:
        #                     break
        #     calMap[dvd][dvs] = res
        #     return res
        
        def bfs(dvd, dvs, last, past_set):
            past_set.add(dvd)
            ret = -1.0
            t_iter = calMap[dvd]
            if dvs in t_iter:
                ret = last * t_iter[dvs]
            else:
                for k,v in t_iter.items():
                    if k in past_set:
                        continue
                    ret = bfs(k, dvs, last*v, past_set)
                    if ret != -1.0:
                        break
            return ret

        for q in queries:
            if q[0] not in calMap or q[1] not in calMap:
                res.append(-1.00000)
            
            elif q[0] == q[1]:
                res.append(1.00000)
            
            else:
                past_set = set()#set(q[0])
                t = bfs(q[0], q[1], 1, past_set)
                res.append(t)
        
        return res

class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:

        gid_weight = {}

        def find(node_id):
            if node_id not in gid_weight:
                gid_weight[node_id] = (node_id, 1)
            group_id, node_weight = gid_weight[node_id]
            # The above statements are equivalent to the following one
            #group_id, node_weight = gid_weight.setdefault(node_id, (node_id, 1))

            if group_id != node_id:
                # found inconsistency, trigger chain update
                new_group_id, group_weight = find(group_id)
                gid_weight[node_id] = \
                    (new_group_id, node_weight * group_weight)
            return gid_weight[node_id]

        def union(dividend, divisor, value):
            dividend_gid, dividend_weight = find(dividend)
            divisor_gid, divisor_weight = find(divisor)
            if dividend_gid != divisor_gid:
                # merge the two groups together,
                # by attaching the dividend group to the one of divisor
                gid_weight[dividend_gid] = \
                    (divisor_gid, divisor_weight * value / dividend_weight)

        # Step 1). build the union groups
        for (dividend, divisor), value in zip(equations, values):
            union(dividend, divisor, value)

        results = []
        # Step 2). run the evaluation, with "lazy" updates in find() function
        for (dividend, divisor) in queries:
            if dividend not in gid_weight or divisor not in gid_weight:
                # case 1). at least one variable did not appear before
                results.append(-1.0)
            else:
                dividend_gid, dividend_weight = find(dividend)
                divisor_gid, divisor_weight = find(divisor)
                if dividend_gid != divisor_gid:
                    # case 2). the variables do not belong to the same chain/group
                    results.append(-1.0)
                else:
                    # case 3). there is a chain/path between the variables
                    results.append(dividend_weight / divisor_weight)
        return results

for loop
