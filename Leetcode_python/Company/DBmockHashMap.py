from abc import abstractproperty
from ast import Index
import collections
from typing import Collection, List, Optional
from collections import *
import sys
import bisect
import heapq
import math
import re
from typing_extensions import Unpack


import time
class mockHashMap:
    def __init__(self):
        self.res_dict = {}
        self.start_time = time.time()
        self.putCallCount = 0
        self.putCallTrack = [] # Each Element in the list is the call times in ith 5 minutes
        self.getCallCount = 0
        self.getCallTrack = []

    def put(self, key, val):
        if key not in res_dict:
            res_dict[key] = []
            res_dict[key].append(val)
        else:
            res_dict[key].append(val)

        if (time.time()-start_time)%300 == 0:
            self.putCallTrack.append(self.putCallCount)
            self.putCallCount = 0
        self.putCallCount += 1

    def get(self, key):
        if (time.time()-start_time)%300 == 0:
            self.getCallTrack.append(self.getCallCount)
            self.putCallCount = 0
        self.getCallCount += 1

        return res_dict[key]  

    def measure_put_load():
        last_5_min_call = self.putCallCount[-1]
        return last_5_min_call/300
    
    def measure_get_load():
        last_5_min_call = self.getCallCount[-1]
        return last_5_min_call/300

class mockHashMap1:
    def __init__:
        self.putBuffer =[0] * 300

    def get(self, k):
        ...
        current_time = time.time()
        diff = min(current_time - self.last_time, 300)
        self.last_time = current_time
        if diff == 0:
            self.putBuffer[self.last_idx] += 1
        else:
            for i in range(diff - 1):
                idx = (self.last_idx + 1 + i) % 300
                self.putBuffer[idx] = 0
            idx = (self.last_idx + diff) % 300
            self.putBuffer[idx] = 1
            self.last_idx = idx

    def measure_put_load(self):
        return sum(self.putBuffer) / 300
