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

class SolutionT1710:
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        
        





