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

class newString:

    default_max_str_size = 20
    index_to_block_map = defaultdict(list)
    index_list = []
    block_size = 5
    total_size = 0

    def __init__(self, N = 20) -> None:
        self.default_max_str_size = N
        self.block_size = int(math.sqrt(N))
    
    def output(self) -> str :
        res = []
        for i in range(len(self.index_list)):
            list = self.index_to_block_map[self.index_list[i]]
            for c in list:
                res.append(c)

            if i != len(self.index_list) - 1:
                res.append(",")
        return "".join(res)

    #index_list[i] 放的是第i个block的startindex
    #lower_bound
    def findBlockIndex(self, i) -> int:
        l, r = 0, len(self.index_list) - 1
        while l <= r:
            mid = (r - l) // 2 + l
            if self.index_list[mid] == i:
                return mid
            elif self.index_list[mid] < i:
                l = mid + 1
            else:
                r = mid - 1
        return l - 1
    
    def get(self, pos: int) -> str:
        try:
            if pos < 0 or pos > self.total_size or len(self.index_to_block_map) == 0:
                raise Exception("Arguments Error: Index out of boundary", pos)
        except Exception as err:
            print(err)
        
        bi = self.findBlockIndex(pos)
        starIndex = self.index_list[bi]
        list = self.index_to_block_map[starIndex]

        return list[pos - starIndex]
    
    def insert(self, i: int, c: str) -> None:
        try:
            if i < 0 or i > self.total_size:
                raise Exception("Arguments Error: Index out of boundary", i)
        except Exception as err:
            print(err)
        
        if len(self.index_to_block_map) == 0:
            list = [c]
            self.index_to_block_map[0] = list.copy()
            self.index_list.append(0)
        
        else:
            bi = self.findBlockIndex(i)
            # print(bi)
            starIndex = self.index_list[bi]
            # print(i - starIndex)
            self.index_to_block_map[starIndex].insert(i - starIndex, c)

            #总是将前面的block都放满，才会增殖新的block，如果插入到非最后一个blocks，则会将从插入的block开始，一直到最后一个block，都进行前一个block的最后一个，放到下一个block的第一个的操作，这样保证除最后一个block外，前面的容量都是满的

            poppedTail = None
            for k in range(bi, len(self.index_list)):
                blockList = self.index_to_block_map[self.index_list[k]]

                if poppedTail :
                    blockList.insert(0, poppedTail)
                
                if k != len(self.index_list) - 1:
                    poppedTail = blockList[-1]
                    blockList.pop()
            
            lastStartIndex = self.index_list[len(self.index_list) - 1]
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

ns = newString(50)
ns.insert(0, 'a')
ns.insert(0, 'b')
ns.insert(0, 'c')
ns.insert(0, 'd')
ns.insert(0, 'e')
ns.insert(0, 'f')
ns.insert(0, 'g')
print(ns.output())

ns.insert(0, 'h')
ns.insert(1, 'i')
ns.insert(2, 'j')
ns.insert(3, 'k')
ns.insert(4, 'l')
ns.insert(5, 'f')
ns.insert(6, 'm')
ns.insert(7, 'n')
ns.insert(12, 'x')
print(ns.output())

print("3" + ns.get(3))
print("5" + ns.get(5))
print("12" + ns.get(12))

ns.delete(12)
print(ns.output())
ns.delete(1)
print(ns.output())
ns.delete(2)
print(ns.output())
ns.delete(0)
print(ns.output())

ns.delete(0)

ns.delete(0)

ns.delete(0)
print(ns.output())