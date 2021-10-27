from _typeshed import Self
import collections
from typing import Collection, List, Optional
from collections import *
import sys

class Solution:
    def countSubArraysBySum(arr, s, k):
        map = collections.defaultdict(deque)
        sum , res = 0 , 0
        def cleanWindow(sum, index):
            indices = map[sum]
            while indices and index - indices[0] > k:
                indices.popleft()
        for i in range(len(arr)):
            sum += arr[i]
            if sum - s in map:
                cleanWindow(sum - s, i)
                res += len(map[sum - s])
            cleanWindow(sum, i)
            map[sum].append(i)
        return res

    def countSubArraysBySum(arr, s, k):
        dp = defaultdict(deque)
        runningSum = 0
        subArraysCount = 0
        dp[0] = deque([-1])
        def cleanWindow(sum, currIndex):
            indices = dp[sum]
            while indices and currIndex - indices[0] > k:
                indices.popleft()
        for i in range(len(arr)):
            runningSum += arr[i]
            complement = runningSum - s
            if complement in dp:
                cleanWindow(complement, i)
                subArraysCount += len(dp[complement])
            cleanWindow(runningSum, i)
            dp[runningSum].append(i)
        return subArraysCount

from heapq import *

def meanAndChessboard(matrix, queries):
    heapBlack = []
    heapWhite = []
    m, n = len(matrix), len(matrix[0])
    for i in range(m) :
        for j in range(n) :
            if ((i + j) % 2 == 0) :
                heapWhite.heappush()

class MyHashMap:

    def __init__(self):
        self.data = [[None] for i in range(1000001)]

    def put(self, key: int, value: int) -> None:
        self.data[key] = value

    def get(self, key: int) -> int:
        val = self.data[key]
        return val if val else -1

    def remove(self, key: int) -> None:         
        self.data[key] = None