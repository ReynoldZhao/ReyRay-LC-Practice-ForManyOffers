import collections
from typing import Collection
import sys

class Solution:
    """
    @param: s: a string
    @param: dict: a set of n substrings
    @return: the minimum length
    """
    def minLength(self, s, dict):
        visited = {}
        queue = collections.deque()
        queue.append(s)
        ans = sys.maxsize
        while len(queue) > 0:
            s = queue.popleft()
            for word in dict:
                pos = s.find(word)
                while pos != -1:
                    if s == word:
                        ans = 0
                        break
                    temp = s[:pos] + s[pos + len(word):]
                    if temp not in visited:
                        visited[temp] = 1
                        ans = min(ans, len(temp))
                        queue.append(temp)

                    pos = s.find(word, pos + 1)
            if ans == 0:
                break
        return ans        

        