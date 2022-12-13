from _typeshed import WriteableBuffer
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

class SolutionT472:
    #词间DP
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        wordSet = set(words)
        res = []

        def dfs(word: str) -> bool:
            n = len(word)
            dp = [0] * (n + 1)
            dp[0] = 1
            for i in range(n):
                if dp[i] == 0: continue
                for j in range(i + 1, n + 1):
                    if j - i < n and word[i:j] in wordSet: dp[j] = 1
                if dp[n] == 1: 
                    return True
            return False

        
        for w in words:
            if dfs(w):
                res.append(w)
        return res

class SolutionT140:
    #Top - down DP
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        wordSet = set(wordDict)
        # table to map a string to its corresponding words break
        # {string: [['word1', 'word2'...], ['word3', 'word4', ...]]}
        memo = defaultdict(list)

        #@lru_cache(maxsize=None)    # alternative memoization solution
        def _wordBreak_topdown(s):
            """ return list of word lists """
            if not s:
                return [[]]  # list of empty list

            if s in memo:
                # returned the cached solution directly.
                return memo[s]

            for endIndex in range(1, len(s)+1):
                word = s[:endIndex]
                if word in wordSet:
                    # move forwards to break the postfix into words
                    for subsentence in _wordBreak_topdown(s[endIndex:]):
                        memo[s].append([word] + subsentence)
            return memo[s]

        # break the input string into lists of words list
        _wordBreak_topdown(s)

        # chain up the lists of words into sentences.
        return [" ".join(words) for words in memo[s]]

#Bottom-up
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        # quick check on the characters,
        #   otherwise it would exceed the time limit for certain test cases.
        if set(Counter(s).keys()) > set(Counter("".join(wordDict)).keys()):
            return []

        wordSet = set(wordDict)

        dp = [[]] * (len(s)+1)
        dp[0] = [""]

        for endIndex in range(1, len(s)+1):
            sublist = []
            # fill up the values in the dp array.
            for startIndex in range(0, endIndex):
                word = s[startIndex:endIndex]
                if word in wordSet:
                    for subsentence in dp[startIndex]:
                        sublist.append((subsentence + ' ' + word).strip())

            dp[endIndex] = sublist

        return dp[len(s)]

class SolutionT140:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        res = []
        n = len(s)
        wordSet = set(wordDict)

        def backtrack(i: int, temp_out: List[str]):
            if i >= len(s):
                res.append(" ".join(temp_out))
            for j in range(i + 1, n):
                if s[i:j] in wordSet:
                    temp_out.append(s[i:j])
                    backtrack(j, temp_out)
                    temp_out.pop()
        backtrack(0, [])

        return res

