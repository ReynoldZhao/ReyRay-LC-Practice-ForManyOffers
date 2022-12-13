from _typeshed import Self, SupportsReadline
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
import time

class newString:

    default_max_size = 20
    block_size = 5
    total_size = 0
    index_to_block_map = collections.defaultdict(list)
    index_list = []

    def __init__(self, N = 25) -> None:
        self.default_max_size = N
        self.block_size = int(math.sqrt(N))
        self.total_size = 0

    def findBlockIndex(self, i : int) -> int:
        l = 0
        r = len(self.index_list) - 1
        while l <= r:
            mid = (r - l) // 2 + l
            if self.index_list[mid] == i:
                return mid
            elif self.index_list[mid] <i:
                l = mid + 1
            else:
                r = mid - 1
        return l - 1

    def get(self, pos: int) -> str:
        try:
            if pos < 0 or pos > self.total_size or len(self.index_to_block_map) == 0:
                raise Exception("Argument", pos)
        except Exception as err:
            print(err)

        bi = self.findBlockIndex(pos)
        startIndex = self.index_list[bi]
        list = self.index_to_block_map[startIndex]

        return list[pos - startIndex]

    def insert(self, i: int, c: str) -> None:
        try:
            if i < 0 or i > self.total_size or len(self.index_to_block_map) == 0:
                raise Exception("Argument", i)
        except Exception as err:
            print(err)  
        
        if len(self.index_to_block_map) == 0:
            list = [c]
            self.index_to_block_map[0] = list.copy()
            self.index_list.append(0)
        
        else:
            bi = self.findBlockIndex(i)
            startIndex = self.index_list[bi]
            self.index_to_block_map[startIndex].insert(i - startIndex, c)

        poppertail = None
        for k in range(bi, len(self.index_list)):
            blockList = self.index_to_block_map[self.index_list[k]]

            if poppertail:
                blockList.insert(0, poppertail)
            
            if k != len(self.index_list) - 1:
                poppertail = blockList[-1]
                blockList.pop()
            
        lastStartIndex = self.index_list[-1]
        lastBlockList = self.index_to_block_map[lastStartIndex]

        if len(lastBlockList) > self.block_size:
            newStartIndex = lastStartIndex + self.block_size
            newBlockList = []

            while len(lastBlockList) > self.block_size:
                newBlockList.insert(0, lastBlockList[-1])
                lastBlockList.pop()
            
            self.index_to_block_map[newStartIndex] = newBlockList.copy()
            self.index_list.append(newStartIndex)

        self.total_size += 1

    def delete(self, i: int):
        try:
            if i < 0 or i > self.total_size:
                raise Exception("Arguments Error: Index out of boundary", i)
        except Exception as err:
            print(err)  

        bi = self.findBlockIndex(i)
        starIndex = self.index_list[bi]
        list = self.index_to_block_map[starIndex]
        del list[i - starIndex]
        
        poppedHead = None
        for k in range(len(self.index_list) - 1, bi - 1, -1):
            blockList = self.index_to_block_map[self.index_list[k]]
            if poppedHead :
                blockList.append(poppedHead)
            
            if k != bi:
                poppedHead = blockList[0]
                del blockList[0]
        
        lastStartIndex = self.index_list[len(self.index_list) - 1]
        lastBlockList = self.index_to_block_map[lastStartIndex]

        if len(lastBlockList) == 0:
            del self.index_list[-1]
            del self.index_to_block_map[lastStartIndex]

        self.total_size -= 1


        

    