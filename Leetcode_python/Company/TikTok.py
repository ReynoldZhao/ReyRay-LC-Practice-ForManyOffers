from abc import abstractproperty
from ast import Index
import collections
from typing import Collection, List, Optional
from collections import *
import sys
import bisect
import heapq
import math

e1 = False #more than 2 children
e2 = False #duplicate edges
e3 = False #cycle
e4 = False #multiple roots
e5 = False # any other error

class MyNode:
    def __init__(self, val = 0, children = 0, left = None, right = None) -> None:
        self.val = val
        #self.treeSize = treeSize
        self.children = children
        self.left = left
        self.right = right

class Solution:
    def isThisTree(self, pairStr: str):
        # pairStr = "(A,B) (A,C) (B,G) (C,H) (E,F) (B,D) (C,E)" 
        pairslist = pairStr.split(" ")
        pairs = []
        for p in pairslist:
            pairs.append((p[1], p[3]))
        pairs.sort(key=lambda x : x[1])

        base = ord('A')

        #放的就是自己 rootNodes[i] -》 node（i)
        rootNodes = [MyNode(i) for i in range(26)]

        #roots[i] -> node(i)的直接爹
        roots = [i for i in range(26)]

        edges = set()
        addedNodes = set()
        for pair in pairs:
            #duplicate edges
            if pair[0] + "2" + pair[1] in edges:
                return "E2"

            n1 = ord(pair[0]) - base
            n2 = ord(pair[1]) - base

            #cycle
            ancestor1 = self.findRoot(roots, n1)
            ancestor2 = self.findRoot(roots, n2)
            if ancestor1 == ancestor2:
                return "E3"
            # if n2 in addedNodes and n1 in addedNodes:
            #     return "E3"

            #只是当前pair的结点
            parentNode = rootNodes[n1]
            childNode = rootNodes[n2]

            #more than 2 children
            if parentNode.children >= 2:
                return "E1"
            
            # childRoot = self.findRoot(roots, childNode)

            childRoot = rootNodes[roots[n2]] #n2的直系爹

            #multiple roots1 一个儿子接了两个爹
            if childRoot.val != n2:
                return "E4"

            addedNodes.add(n1)
            addedNodes.add(n2)

            roots[n2] = n1
            parentNode.children += 1

            # parentNode.treeSize == childNode.treeSize
            # if rootNodes[ancestor1] != parentNode:
            #     rootNodes[ancestor1] += childNode.treeSize

            if parentNode.left == None:
                parentNode.left = childNode
            else :
                parentNode.right = childNode
        
        realRoot = 0
        flag = False
        for i in range(26) :
            if i in addedNodes:
                if roots[i] == i:
                    if flag :
                        #multiple roots
                        return "E4"
                    realRoot = i
                    flag = True
        realRootNode = rootNodes[realRoot]
        res = []
        def inorder(root: MyNode) :
            res.append("(")
            res.append(chr(base + ord(root.val)))
            if not root.left and not root.right:
                res.append(")")
                return
            if not root.left:
                inorder(root.left)
            if not root.right:
                inorder(root.right)
        return res

            
    # 找到的是总爹
    def findRoot(self, roots: List[int], node: int):
        return node if roots[node] == node else self.findRoot(roots, roots[node])

    # def findRoot(self, roots: List[MyNode], node: MyNode):
    #     return node if roots[node.val] == node.val else self.findRoot(roots, roots[node])

class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        words.sort(key = lambda x : len(x))
        dp = collections.defaultdict(int)
        res = 1
        for word in words:
            curL = 1
            for i in  range(len(word)):
                predecessor = word[:i] + word[i + 1:]
                if predecessor in dp:
                    preL = dp[predecessor]
                    curL = max(curL, preL + 1)
            dp[word] = curL
            res = max(res, curL)
        return res