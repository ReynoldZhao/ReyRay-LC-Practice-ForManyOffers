# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

from collections import Counter, defaultdict

#大师级的python代码，括号生成
def generateParenthesis(self, n):
    def generate(p, left, right):
        if right >= left >= 0:
            if not right:
                yield p
            for q in generate(p + '(', left-1, right): yield q
            for q in generate(p + ')', left, right-1): yield q
    return list(generate('', n, n))

#Improved version of this. Parameter open tells the number of "already opened" parentheses, and I continue the recursion as long as I still have to open parentheses (n > 0) and I haven't made a mistake yet (open >= 0).
def generateParenthesis(self, n, open=0):
    if n > 0 <= open:
        return ['(' + p for p in self.generateParenthesis(n-1, open+1)] + \
               [')' + p for p in self.generateParenthesis(n, open-1)]
    return [')' * open] * (not n)
#T22. Generate Parentheses

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        carry = 0
        root = n = ListNode(0)

        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:

            carry, val = divmod(v1+v2+carry, 10)
            n.next = ListNode

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        map = {}
        j = 0
        for i in range(len(s)):
            if s[i] in map:
                j = max(j, map[s[i]]+1)
            else:
                map[s[i]] = i

class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        m = len(nums1)
        n = len(nums2)

        return findKth(nums1, nums2, (m + n + 1)/2 ) + findKth(nums1, nums2, (m + n + 2)/2 )
    
    def findKth(slef, nums1, nums2, k):
        if(not a) return nums2[k-1]
        if(not b) return nums1[k-1]
        if(k==1) return min(nums1[0], nums2[0])
        index1 = min(len(a), k/2)
        index2 = min(len(b),k - k/2)
        if(nums1[index1] < nums2[index2]):
            findKth(nums[index1:], nums2, k - index1)

class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if(len(s)<2) return s
        for i in range(len(s)):
            left = i, right = i
            while(right<len(s) and s[right] == s[right+1]): 
                right += 1
            while(left >= 0 and right < len(s) and s[])

def maxArea(self, height):
    L, R, width, res = 0, len(height)-1, len(height)-1, 0
    for w in range(width, 0, -1):

def letterCombinations(self, digits):
    if digits == '': return []
    kvmaps = {
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz'
    }
    return reduce(lambda acc, digit: [x + y for x in acc for y in kvmaps[digit]], digits, [''])

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        def index(node):
            if not node：
                return 0
            i = index(node.next) + 1
            if i > n:
                node.next.val = node.val
            return i
        
        index(head)
        return head.next

    class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        dict = {"]":"[", "}":"{", ")":"("}
        for char in s:
            if char in dict.values():
                stack.append(char)
            elif char in dict.keys():
                if stack == [] or stack.pop() != dict[char] return False
            elif
                return False
            return stack == []

    class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1 or not l2:
            return l1 or l2
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        if l2.val < l1.val:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        def generator(p, left, right, res = []):
            if left:
                generator(p + '(', left - 1, right)
            if right > left: 
                generator(p + ')', left, right - 1)
            if right:
                res.append(p)

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

from Queue import PriorityQueue
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        dummy = listNode(None)
        cur = dummy
        q = PriorityQueue()
        for node : lists:
            if node:
                q.put((node.val, node))
        while q.size() > 0:
            cur.next = q.get()[1]
            cur = cur.next
            if cur.next:
                q.put((cur.next.val, cur.next))
        return dummy.next

class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        dfs(res, [], nums)
        return res

    def dfs(self, res, temp, nums):
        if len(temp) = len(nums):
            res.appen(temp)
        for i in range(len(nums)):
            if nums[i] in temp:
                continue
            temp.append(nums[i])
            dfs(res, temp, nums)
            temp.remove(nums[i])

class SolutionT3(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        start = maxLength = 0
        usedChar = {}

        for i in range(len(s)):
            if s[i] in usedChar and start <= usedChar[s[i]]:
                start = usedChar[s[i]] + 1
            else :
                maxLength = max(maxLength, i - start + 1)
            usedChar[s[i]] = i
        
        return maxLength

class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        s = s.strip(" ")
        if not s or (s[0] not in ["+", "-"] and not s[0].isdigit()):
            return 0

        op = 1
        tmp = ""
        for i, char in enumerate(s):
            if i == 0:
                if char == "-":
                    op = -1
                    continue
                elif char == "+":
                    pass 
                    continu
            if char.isdigit():
                tmp += char
            else :
                break

        if temp:
            res = op * int(tmp)
        else:
            res = 0
        
        INT_MAX = 2 ** 31 - 1
        INT_MIN = - 2 ** 31  
        if res > INT_MAX：
            return INT_MAX
        elif res < INT_MIN:
            return INT_MIN
        else :
            return res

class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        dict = {}
        for c in s:
            dict[c] = dict.get(c, 0) + 1
        for i, c in enumerate(s):
            if dict[c] == 1:
                return i
        return -1

class SolutionT344(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        tmep = c for c in reversed(s) if c in 'aeiouAEIOU'
        return re.sub('(?i)[aeiou]', lambda m:next(temp), s)

class SolutionT6(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows == 1 or numRows >= len(s):
            return s
        
        L = [''] * numRows
        index, step = 0, 1

        for x in s:
            L[index] += x
            if index == 0:
                step = 1
            elif index == numRows - 1：
                step = -1
            index += step
        
        return ''.join(L)

class Solution(object):
    def myAtoi(self, s):
        """
        :type str: str
        :rtype: int
        """
        ###better to do strip before sanity check (although 8ms slower):
        #ls = list(s.strip())
        #if len(ls) == 0 : return 0
        if len(s) == 0 : return 0
        ls = list(s.strip())

        sign = -1 if ls[0] == '-' else 1
        if ls[0] in ['-', '+'] : del ls[0]
        res, i = 0, 0
        while i < len(ls) and ls[i].isdigit():


 class SolutionT12:
    def intToRoman(self, num: int) -> str:           
        d = {1000: 'M', 900: 'CM', 500: 'D', 400: 'CD', 100: 'C', 90: 'XC', 50: 'L', 40: 'XL', 10: 'X', 9: 'IX', 5: 'V', 4: 'IV', 1: 'I'} 

        res = ""

        for i in d :
            res += (num//d) * d[i]
            num %= i

        return res

     def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ""
        shortest = min(strs, key = len)
        for i, ch in enumerate(shortest) :
            for other in strs:
                if other[i] != ch:
                    return shortest[:i]
        return shortest

class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        if(num1 == "0" or num2 == "0"):
            return "0"

        if len(num1) < len(num2):
            num1, num2 = num2, num1
        res = ""
        num1 = num1[::-1]
        nums2 = nums2[::-1]
        list1 = list(num1), list2 = list(num2)
        m = len(list1), n = len(list2)
        l = [0 for i in range(m+n)]
        for i in

        list2 = list(num2)
        for i in range(len(list2)):
            temp = multi(num1, list2[-i-1])
            res = plus(res, temp)
        return res
    
    def multi(self, s, ch) :
        res = ""
        l = list(s)
        c = ord(ch) - ord("0")
        next = 0
        for i in range(len(l)):
            number = ord(l[-i-1]) - ord("0")
            t = number * c + next
            res = str(t % 10) + res
            next = t // 10
        return res

    
    def plus(self, s1, s2):
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        l1 = list(s1)
        l2 = list(s2)
        add = 0
        res = ""
        for i in range(len(l1)):
            if i < len(l2) :
                temp = ord(l1[-i-1]) - ord("0") + ord(l2[-i-1]) - ord("0") + add
                add = temp // 10
                res = str(temp % 10) + res
            else:
                temp = ord(l1[-i-1]) - ord("0") + add
                add = temp // 10
                res = str(temp % 10) + res
        if add == 1:
            res = "1" + res
        return res
    
class SolutionT30 (object):
    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        wordbag = Counter(words)
        wordLen, numWords = len(words[0]), len(word)
        totalLen, res = wordLen * numWords, []
        for i in range(len(s) - totalLen + 1):
            seen = defaultdict(int)
            for j in range(i, i + totalLen, wordLen):
                curWord = s[j:j+wordLen]
                if curWord in wordbag:
                    seen[curWord] += 1
                    if (seen[curWord] > wordbag[curWord]):
                        break
                else:
                    break
            if seen == wordbag:
                res.append(i)
        return res

        wordBag = Counter(words)   # count the freq of each word
        wordLen, numWords = len(words[0]), len(words)
        totalLen, res = wordLen*numWords, []
        for i in range(len(s)-totalLen+1):   # scan through s
            # For each i, determine if s[i:i+totalLen] is valid
            seen = defaultdict(int)   # reset for each i
            for j in range(i, i+totalLen, wordLen):
                currWord = s[j:j+wordLen]
                if currWord in wordBag:
                    seen[currWord] += 1
                    if seen[currWord] > wordBag[currWord]:
                        break
                else:   # if not in wordBag
                    break    
            if seen == wordBag:
                res.append(i)   # store result
        return res

class SolutionT76(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        charbag = collections.Counter(t)
        left = 0, cnt = 0
        res = "", minLen = 2**32 - 1
        check = charbag
        for i, c in enumerate(s) :
            if c in charbag :
                check[c]-=1
                if check[c] >= 0:
                    cnt+=1
            else :
                continue
            while(cnt == len(t)):
                if minLen > i - left + 1:
                    minLen = i - left + 1
                    res = s[left:left + minLen]
                if s[left] in charbag :
                    check[left]+=1
                    if check[left] > 0 :
                        cnt-=1
                left+=1
        return res 

        charbag = collections.Counter(t)
        left, cnt = 0, 0
        res, minLen ="", 2**32 - 1
        check = charbag
        for i, c in enumerate(s) :
            check[c]-=1
            if check[c] >= 0:
                cnt+=1
            while(cnt == len(t)):
                if minLen > i - left + 1:
                    minLen = i - left + 1
                    res = s[left:left + minLen]
                check[s[left]]+=1
                if check[s[left]] > 0 :
                    cnt-=1
                left+=1                
        return res 

class SolutionT468(object):

    def validIPAddress(self, IP):
        """
        :type IP: str
        :rtype: str
        """
        def isIPv4(s):
            try: return str(int(s)) == s and 0 <= int(s) <= 255
            except: return False

        def isIPv6(i):
            #if len(i) <= 0 or len(i) > 4: return False
            try: return 0 < len(i) <= 4 and int(i, 16) > 0 or i[0] != "-"
            except: return False

        if IP.count(".") == 3 and all(isIPv4(s) for s in IP.split(".")):
            return "IPv4"
        if IP.count(":") == 7 and all(isIPv6(s) for s in IP.split(":")):
            return "IPv6"
        return "Neither"

class SolutionT1422(object):
    def maxScore(self, s):
        """
        :type s: str
        :rtype: int
        """
        l = len(s)
        zero = s.count("0")
        cur_zero = 0
        cur_one = 0
        res = 0
        for i in range(len(s)):
            if s[i] == "0" :
                cur_zero += 1
            else :
                cur_one += 1
            res = max(res, cur_zero + l - zero - cur_one)
        return res

 class Solution(object):
    def findDuplicate(self, paths):
        """
        :type paths: List[str]
        :rtype: List[List[str]]
        """
        map = collections.defaultdict(list)
        for line in paths:
            data = line.split()
            root = data[0]
            for file in data[1:]:
                name, _, content = file.partition('(')
                M[content[:-1]].append(root + '/' + name)
        return [x for x in M.values if len(x) > 1]

class SolutionT1436(object):
    def destCity(self, paths):
        """
        :type paths: List[List[str]]
        :rtype: str
        """
        A, B = map(set, zip(*paths))
        return (B - A).pop()
        
        return list(set(list(zip(*paths))[1]) - set(list(zip(*paths))[0]))[0]

        return list(set([path[1] for path in paths])) - set([path[0] for path in paths]))[0]

class SolutionT1417:
    def reformat(self, s: str) -> str:
        letters = [c for c in s if c.isalpha()]
        digits = [c for c in s if c.isdigit()]
        if abs(len(digits) - len(letters)) > 1: return ''
        if len(letters) > len(digits): letters, digits = digits, letters
        return "".join(map(lambda x: x[0] + x[1], zip_longest(letters, digits, fillvalue = '')))

class SolutionT1443:
    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        adj = [] for _ in range(n)
        for u, v in edges:
            adj[u].append(v)
            adj[v],append(u)
        
        visited = set()
        def dfs(node):
            if node in visited:
                return 0
            visited.add(node)
            secs = 0
            for child in adj[node]:
                secs += dfs(child)
            if secs > 0:
                return secs + 2
            return 2 if hasApple[node] else 0

    return max(dfs(0) - 2, 0)

    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        graph = {}
        for a, b in edges
            graph[b] = a
        res = 0
        for i in range(n):
            if hasApple[i]:
                p = i
                while p !=0 and graph[p] >= 0:
                    temp = graph[p]
                    graph[p] = -1
                    res += 2
                    p = temp
        return res

class SolutionT373:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        res = []
        heap = []
        if len(nums1) == 0 or len(nums2) == 0 or k == 0:
            return res
        i = 0
        while i < len(nums1) and i < k :
            heapq.heappush(heap, (nums1[i] + nums2[0], nums1[i], nums2[0], 0))
            i += 1
        while k and h :
            cur = heappop(h)
            res.append(cur[0])
            if cur[3] == len(nums2) - 1:
                continue
                heapq.heappush(heap, (cur[1] + nums2[cur[3] + 1], cur[1], nums[cur[3] + 1], cur[3] + 1))
                k -= 1
        return res

    def kSmallestPairs(self, nums1, nums2, k):
        return sorted(itertools.product(nums1, nums2), key = sum)[:k]

        return map(list, sorted(itertools.product(nums1, nums2), key = sum)[:k])

        return map(list, heapq.nsmallest(k, itertools.product(nums1, nums2), key = sum))

        return heapq.nsmallest(k, ([u,v] for u in nums1 for v in nums2), key = sum)

        streams = map(lambda u: ([u + v, u, v] for v in nums2), nums1)
        streams = heapq.merge(*streams)
        return [suv[1:] for suv in itertools.islice(stream, k)]

    def kSmallestPairs(self, nums1, nums2, k):
        queue = []
        def push(i, j):
            if i < len(nums1) and j < len(nums2):
                heapq.heappush(queue, [nums1[i] + nums2[j], i, j])
        push(0, 0)
        pairs = []
        while queue and len(pairs) < k:
            _, i, j = heapq.heappop(queue)
            pairs.append([nums1[i], nums2[j]])
            push(i, j + 1)
            if j == 0:
                push(i + 1, 0)
        return pairs

class SolutionT1451(object):
    def arrangeWords(self, text):
        """
        :type text: str
        :rtype: str
        """
        text = text.lower()
        arr = text.split()
        arr.sort(cmp = lambda x,y:cmp(len(x),len(y)))
        res = " ".join(arr)
        res = res.capitalize()
        return res

class SolutionT1452(object):
    def peopleIndexes(self, favoriteCompanies):
        """
        :type favoriteCompanies: List[List[str]]
        :rtype: List[int]
        """
        sets = [set(l) for l in favoriteCompanies]
        ans = []
        for i in range(len(favoriteCompanies)):
            flag = True
            for j in range(len(favoriteCompanies)):
                if i != j and sets[i].issubset(sets[j]):
                    flag = False
                if flag == False:
                    res.append(i)
                    break
        return ans

class Solution(object)T1455 :
    def isPrefixOfWord(self, sentence, searchWord):
        """
        :type sentence: str
        :type searchWord: str
        :rtype: int
        """
        word = sentence.split()
        if(len(searchWord) == 0) return 1
        if(len(word) == 0) return -1
        n = len(searchWord)
        for i, word in enumerate(word):
            if len(word) < len(searchWord):
                continue
            else :
                if word[:n] == searchWord:
                    return i
        return -1

class Solution(object)T1456 :
    def maxVowels(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        vowels = {'a', 'e', 'i', 'o', 'u'}
        res = 0
        for i in range(len(s)):
            count = 0
            for j in range(k):
                if (i + j) < len(s) and s[i + j] in vowels:
                    count = count + 1
            res = max(res, count)
        return res

        vowels = {'a', 'e', 'i', 'o', 'u'}
        res = 0
        dp = [0]
        count = 0
        for i in range(len(s)):
            if s[i] in vowels:
                count+=1
            dp.append(count)
        for i in range(len(s) - k + 1):
            res = max(res, dp[i+k]) - dp[i])
        return res

class Solution(object)T1461:
    def hasAllCodes(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: bool
        """
        return len({s[i:i + k] for i in xrange(len(s) - k + 1)}) == 2 ** k


class SolutionT17:
    def printNumbers(self, n: int) -> [int]:
        def dfs(x):
            if x == n:
                s = ''.join(num[self.start:])
                if s != '0':
                    res.append(int(s))
                if n - self.start == self.nine:
                    self.start -= 1
                return 
            for i in range(10):
                if i == 9:
                    self.nine += 1
                num[x] = str(i)
                dfs(x+1)
            self.nine -= 1
        num, res = ['0']*n, []
        self.nine = 0
        self.start = n - 1
        dfs(0)
        return res

class SolutionT493:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        d = {}
        def dfs(cur, i, d):
            if i < len(nums) and (cur, i) not in d:
                d[(cur, i)] = dfs(cur + nums[i], i + 1, d) + dfs(cur - nums[i], i+1, d) 
            return d.get((cur, i), int(cur == S))
        return dfs(0, 0, d)

    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        if sum(nums) < S or (sum(nums) + S) % 2 == 1: return 0
        P = (sum(nums) + S) // 2
        dp = [1] + [0 for _ in range(P)]
        for num in nums:
            for j in range(P, num-1, -1):
                dp[j] += dp[j - num] #j - num组成j - num有多少种可能 + num 都到j 
        return dp[P]

    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        length, dp = len(nums), {(0, 0):1}
        summary = sum(nums)
        for i in range(1, length + 1) :
            for j in range(-sum, sum+1):
                # true index in nums is i-1
                dp[(i, j)] = dp.get((i-1, j -nums[i-1]), 0) + dp.get((i-1, j + nums[i+1]), 0)
        return dp.get((length, S), 0)

