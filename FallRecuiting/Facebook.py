from _typeshed import Self, SupportsReadline
from abc import abstractproperty
from ast import Index
import collections
from operator import le
from typing import Collection, List, Optional
from collections import *
import sys
import bisect
import heapq
import math
import re
import time

class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        left, right = 0, 0
        res = []
        for i in range(len(s)):
            if s[i] == '(':
                left+=1
                res.append(s[i])
            elif s[i] == ')':
                left-=1
                if left < 0:
                    left = 0
                    continue
                else:
                    res.append(s[i])
            else:
                res.append(s[i])
        res_2 = []
        for i in range(len(res)-1, -1, -1):
            if res[i] == ')':
                right+=1
                res_2.append(res[i])
            elif res[i] == '(':
                right-=1
                if right < 0:
                    right = 0
                    continue
                else:
                    res_2.append(res[i])
            else:
                res_2.append(res[i])
        res_2.reverse()
        r = "".join(res_2)
        return r
