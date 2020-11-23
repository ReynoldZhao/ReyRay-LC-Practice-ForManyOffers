from collections import Counter, defaultdict

class Solution0103:
    def replaceSpaces(self, S: str, length: int) -> str:
        return "%20".join(S[:length].split(" "))

class Solution:
    def canPermutePalindrome(self, s: str) -> bool:
        return sum(1 for k, v in collections.Counter(s).items() if v % 2 != 0) <= 1

