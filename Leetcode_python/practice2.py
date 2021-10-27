
from _typeshed import Self
import collections
import heapq
from typing import Collection, List, Optional
from collections import *
import sys

class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res, stack = [], []
        p = root
        while len(res) > 0 or p:
            while p :
                stack.append(p)
                p = p.left
            p = stack.pop()
            res.append(p)
            if p.right :
                p = p.right
                stack.append(p)
        return res

class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root == None:
            return []
        res, stack = [], [root]
        while len(stack) > 0:
            p = stack.pop()
            res.append(p.val)
            if p.left :
                stack.append(p.left)
            if p.right:
                stack.append(p.right)
        return res[::-1]

#105. Construct Binary Tree from Preorder and Inorder Traversal
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if len(preorder) == 0:
            return None
        if len(inorder) == 0:
            return None
        root_val = preorder[0]
        root = TreeNode(root_val)

        index = inorder.index(root_val)
        linorder = inorder[:index]
        rinorder = inorder[index + 1:]
        lpreorder = preorder[1 : len(linorder) + 1]
        rpreorder = preorder[1 + len(lpreorder):]

        root.left = self.buildTree(lpreorder, linorder)
        root.right = self.buildTree(rpreorder, rinorder)

        index = inorder.index(preorder.pop(0))
        root = TreeNode(inorder[index])
        root.left = self.buildTree(preorder, inorder[:index])
        root.right = self.buildTree(preorder, inorder[index+1:])
        return root

#106. Construct Binary Tree from Inorder and Postorder Traversal
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if inorder:
            index = inorder.index(postorder[-1])
            root = TreeNode(inorder[index])

            linorder = inorder[0:index]
            rinorder = inorder[index+1:]
            lporder = postorder[0:len(linorder)]
            rporder = postorder[len(linorder):-1]
            root.left = self.buildTree(linorder, lporder)
            root.right = self.buildTree(rinorder, rporder)
            return root
        else :
            return None

    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        map = {}
        for i, val in enumerate(inorder) : 
            map[val] = i
        def recur(low, high):
            if (low > high): return None
            root = TreeNode(postorder.pop())
            index = map[root.val]
            root.right = recur(index+1, high)
            root.left = recur(low, index-1)
            return root
        return recur(0, len(inorder) - 1)

class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not postorder:
            return None
        root = TreeNode(preorder[0])
        if len(postorder) == 1:
            return root
        index = preorder.index(postorder[-2])
        root.left = self.constructFromPrePost(preorder[1:index], postorder[:index - 1])
        root.right = self.constructFromPrePost(preorder[index:], postorder[index-1:-1])
        return root

class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left

    def next(self) -> int:
        node = self.stack.pop()
        if node.right:
            while node.right:
                self.stack.append(node.right)
                node = node.right
        return node.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0

class MyTreeNode:
    def __init__(self, val=0, count = 1, left=None, right=None):
        self.val = val
        self.count = count
        self.left = left
        self.right = right
        
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        myroot = self.build(root)
        return self.help(myroot, k)

    def build(self, root: Optional[MyTreeNode]) -> MyTreeNode:
        if not root:
            return None
        node = MyTreeNode(root.val)
        node.left = self.build(root.left)
        node.right = self.build(root.right)
        if node.left: node.count = node.left.count
        if node.right: node.count = node.right.count
        return node

    def help(self, root: Optional[MyTreeNode], k: int) -> int:
        if root.left:
            cnt = root.left.count
            if k <= cnt:
                return help(root.left, k)
            elif k > cnt + 1:
                return help(root.right, k - cnt - 1)
            return root.val
        else :
            if k == 1:
                return root.val
            else:
                return help(root.right, k - 1)
    
    def fun(self, str):
        i = 0

class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        res = root
        while root:
            if root.val > p.val:
                res = root
                root = root.left
            elif root.val < p.val:
                res = root
                root = root.right
            return res  

class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the minimum subtree
    """
    def findSubtree(self, root):
        # write your code here
        self.res = None
        self.minSum = float('inf')
        self.helper(root)
        return self.res
    
    def helper(self, node):
        if node == None:
            return 0
        sum = self.helper(node.left) + self.helper(node.right) + node.val
        if sum <= self.minSum:
            self.minSum = sum
            self.res = node
        return sum

class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the maximum average of subtree
    """
    average, node = 0, None

    def findSubtree2(self, root):
        # write your code here
        self.helper(root)
        return self.node
    
    def helper(self, root):
        if root == None:
            return 0, 0
        
        left_sum, left_size = self.helper(root.left)
        right_sum, right_size = self.helper(root.right)

        sum, size = left_sum + right_sum + root.val, \
                    left_size + right_size + 1
        
        if self.node == None or sum * 1.0 / size > self.average :
            self.node = root
            self.average = sum * 1.0 / size
        
        return sum, size

class Solution:
    """
    @param root: the root of binary tree.
    @return: An integer
    """
    def maxPathSum2(self, root):
        # write your code here
        if root == None:
            return 0
        
        leftMax = self.maxPathSum2(root.left)
        rightMax = self.maxPathSum2(root.right)

        return max(leftMax, rightMax) + root.val 

class Solution:
    res = 0

    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        self.dfs(root, root.val, 0, self.res)
        return self.res
    
    def dfs(self, root, v, out, res) :
        if root == None:
            return 
        if (root.val == v + 1) : out+=1
        else : out = 1
        res = max(res, out)
        self.dfs(root.left, root.val, out, res)
        self.dfs(root.right, root.val, out, res)

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordDict = set(wordList)
        queue = collections.deque([beginWord, 1])
        while queue:
            word, step = queue.popleft()
            if word == endWord:
                return step
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    newWord = word[i:] + c + word[i+1:]
                if newWord in wordDict:
                    queue.append([newWord, step + 1])
                    wordDict.remove(newWord)
            
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root == None:
            return None
        if root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left == None and right == None:
            return root
        return right if left == None else left

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None

class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        s = set()
        tq, tp = q, p
        while (tq and tp) :
            if tq in s :
                return tq
            if tp in s :
                return tp
            s.add(tq)
            s.add(tp)
            tq = tq.parent
            tp = tp.parent

# Question 什么时候放进去的是深拷贝也就是，C++中的&
class Solution1:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        self.dfs(candidates, 0, target, [], res)
        return res
    
    def dfs(self, nums, index, target, path, res) :
        if target < 0:
            return
        if target == 0:
            res.append(path[:])
            return 
        for i in range(index, len(nums)) :
            path.append(nums[i])
            self.dfs(nums, i, target - nums[i], path[:], res)
            path.pop()

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        self.dfs(candidates, 0, target, [], res)
        return res
    
    def dfs(self, nums, index, target, path, res) :
        if target < 0:
            return
        if target == 0:
            res.append(path)
            return 
        for i in range(index, len(nums)) :
            self.dfs(nums, i, target - nums[i], path + [nums[i]], res)

class Solution2:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        def dfs(index, target, path) :
            if target < 0:
                return
            if target == 0:
                res.append(path)
                return 
            for i in range(index, len(candidates)) :
                path.append(candidates[i])
                dfs( i, target - candidates[i], path)
                path.pop()
        dfs(0, target, [])
        return res

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        self.dfs(nums, [], res)
        return res
    
    def dfs(self, nums, path, res: List[int]):
        res.append(path)
        for i in range(len(nums)):
            self.dfs(nums[i+1:], path + [nums[i]], res)

#visited 有问题
class SolutionT47:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        sorted(nums)
        visited = [0 for i in range(len(nums))]
        def dfs(idx, path, res):
            if idx == len(nums):
                res.append(path)
            for i in range(0, len(nums)) :
                if visited[i] == 1 or (i > 0 and nums[i] == nums[i-1] and visited[i-1] == 0):
                    continue
                visited[i] = 1
                dfs(idx+1, path+[nums[i]], res)
                visited[i] = 0
        dfs(0, [], res)
        return res

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        sorted(nums)
        visited = [0 for i in range(len(nums))]
        self.dfs(nums, 0, [], visited, res)
        return res

    def dfs(self, nums, idx, path, visited, res):
        if idx == len(nums):
            res.append(path)
        for i in range(0, len(nums)) :
            if visited[i] == 1 or (i > 0 and nums[i] == nums[i-1] and visited[i-1] == 0):
                continue
            visited[i] = 1
            self.dfs(idx+1, path+[nums[i]], res)
            visited[i] = 0

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        self.dfs(nums, [], res)
        return res

    def dfs(self, nums, path, res):
        if not nums:
            res.append(path)
            return
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            self.dfs(nums[:i] + nums[i+1:], path + [nums[i]], res)

class SolutionT301:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        cnt1, cnt2 = 0, 0
        for t in s:
            if t == '(': cnt1+=1
            elif t == ')':
                if cnt1 == 0: cnt2+=1
                else: cnt1 -= 1

        res = []
        self.dfs(0, s, cnt1, cnt2, res)
        return res
   
    def isValid(self, str):
        cnt = 0
        for t in str:
            if t == '(': cnt+=1
            elif t == ')' :
                cnt -= 1
                if cnt < 0: return False
        return cnt == 0
    
    def dfs(self, index, str, cnt1, cnt2, res):
        if cnt1 == 0 and cnt2 == 0:
            if self.isValid(str) : res.append(str)
            return 
        for i in range(index, len(str)):
            if i > index and str[i - 1] == str[i]:
                continue
            if cnt1 > 0 and str[i] == '(':
                self.dfs(i, str[:i] + str[i+1:], cnt1 - 1, cnt2, res)
            if cnt2 > 0 and str[i] == ')':
                self.dfs(i, str[:i] + str[i+1:], cnt1, cnt2 - 1, res)

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        queenCol = [-1 for i in range(n)]
        self.dfs(0, queenCol, res)
        return res
    
    def isValid(self, queenCol, row, col):
        for i in range(row):
            if col == queenCol[i] or (abs(row - i) == abs(col - queenCol[i])): return False
        return True

    def dfs(self, curRow, queenCol, res):
        n = len(queenCol)
        if curRow == n:
            print(queenCol)
            temp = []
            for i in range(n):
                pos = queenCol[i] + 1
                temp.append(["."*(pos - 1) + "Q" + "."*(n - pos)])
            res.append(temp)
            return
        for i in range(n):
            if self.isValid(queenCol, curRow, i) == True:
                queenCol[curRow] = i
                self.dfs(curRow + 1, queenCol, res)
                queenCol[curRow] = -1

class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        res = []
        self.dfs(nums, 0, [], res)
        return res
    
    def dfs(self, nums, idx, out, res) :
        if len(out) >= 2:
            res.append(out)
        intSet = set()
        for i in range(idx, len(nums)) :
            if nums[i] in intSet or (len(out) > 0 and out[-1] > nums[i]) :
                continue
            intSet.add(nums[i])
            self.dfs(nums, i + 1, out + [nums[i]], res)

class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        words = s.split(" ")
        if len(pattern) != len(words):
            return False
        dict_ch = {}
        dict_w = {}
        for ch, word in zip(pattern, words):
            if ch not in dict_ch and word not in dict_w:
                dict_ch[ch] = word
                dict_w[word] = ch
            elif (ch not in dict_ch and word in dict_w):
                return False
            elif (ch in dict_ch and dict_ch[ch] != word):
                return False
        return True

class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        map = collections.defaultdict(int)
        left, res = 0, -sys.maxsize
        for i in range(len(str)):
            map[str[i]]+=1
            while len(map) > k :
                map[str[left]]-=1
                if (map[str[left]] == 0):
                    del map[str[left]]
                left+=1
        res = max(res, i - left + 1)

            
class Solution:
    def countSubArraysBySum(arr, s, k):
        map = collections.defaultdict(deque)
        sum , res = 0 , 0
        def cleanWindow(sum, index):
            indices = map[sum]
            while indices and index - indices[0] > k:
                indices.popleft()
        for i in range(len(arr)):
            sum += arr[i]
            if sum - s in map:
                cleanWindow(sum - s, i)
                res += len(map[sum - s])
            cleanWindow(sum, i)
            map[sum].append(i)
        return res

    def countSubArraysBySum(arr, s, k):
        dp = defaultdict(deque)
        runningSum = 0
        subArraysCount = 0
        dp[0] = deque([-1])
        def cleanWindow(sum, currIndex):
            indices = dp[sum]
            while indices and currIndex - indices[0] > k:
                indices.popleft()
        for i in range(len(arr)):
            runningSum += arr[i]
            complement = runningSum - s
            if complement in dp:
                cleanWindow(complement, i)
                subArraysCount += len(dp[complement])
            cleanWindow(runningSum, i)
            dp[runningSum].append(i)
        return subArraysCount

class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        res, n = 0, len(s)
        for cnt in range(1, 27):
            start, i, uniqueCnt = 0, 0, 0
            charCnt = [0 for i in range(26)]
            while i < n :
                isValid = True
                if charCnt[ord(s[i]) - ord('a')] == 0:
                    uniqueCnt+=1
                charCnt[ord(s[i]) - ord('a')] += 1
                i+=1
                while uniqueCnt > cnt :
                    charCnt[ord(s[start]) - ord('a')] -= 1
                    if charCnt[ord(s[start]) - ord('a')] == 0:
                        uniqueCnt-=1
                    start += 1
                for j in range(26) :
                    if charCnt[j] > 0 and charCnt[j] < k:
                        isValid = False
                if isValid:
                    res = max(res, i - start)
        return res

class Solution:
    def wordPatternMatch(self, pattern: str, s: str) -> bool:
        map = collections.defaultdict(int)
        def helper(i, j):
            if i >= len(pattern) and j >= len(s) :
                return True
            if i >= len(pattern) or j >= len(s) :
                return False
            c = pattern[i]
            for idx in range(j, len(s)):
                temp = s[j:idx + 1]
                print(c, i, temp)
                if c in map and map[c] == temp:
                    if helper(i + 1, idx + 1): return True
                elif c not in map:
                    if temp in map.values():
                        continue;
                    map[c] = temp
                    if helper(i + 1, idx + 1): return True
                    del map[c]
            return False
        return helper(0, 0)

class SolutionT442:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        res = []
        for i in range(len(nums)):
            # idx = abs(nums[i]) - 1
            # if nums[idx] < 0:
            #     res.append(idx + 1)
            # nums[idx] = -nums[idx]
            if nums[i] != nums[nums[i] - 1]:
                nums[i], nums[nums[i] - 1] = nums[nums[i] - 1], nums[i]
                i -= 1
        for i in range(len(nums)):
            if nums[i] != i + 1:
                res.append(nums[i])
        return res

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def convert(self, s: str, numRows: int) -> str:
        temp = ["" for i in range(numRows)]
        i = 0
        n = len(s)
        while i < len(s) :
            for p in range(numRows) :
                if i >= n: break
                temp[p]+=(s[i])
                i += 1
            for q in range(numRows - 2, 0, -1):
                if i >= n: break
                temp[q]+=(s[i])
                i += 1
        return "".join(temp)
    
    def addTwoNumbers(self, l1, l2):
        carry = 0
        root = n = ListNode(0)
        l1 = self.reverseNode(l1)
        l2 = self.reverseNode(l2)
        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1+v2+carry, 10)
            n.next = ListNode(val)
            n = n.next
        return self.reverseNode(root.next)
    
    def reverseNode(self, l: Optional[ListNode]):
        if not l:
            return None
        dummy = ListNode(0)
        cur = l
        while cur :
            l = l.next
            cur.next = dummy.next
            dummy.next = cur
            cur = l
        return dummy.next

def maximalPalindrome(s):
    dict = collections.defaultdict(int)
    res = ""
    heap = []
    for i in range(len(s)) :
        dict[s[i]]+=1
    for ch, count in dict.items():
        if count % 2 == 0:
            res += ch * (count//2)
        elif count % 2 != 0:
            heapq.heappush(heap, (-count+ord(ch), ch, count))
    tempL = list(res)
    tempL.sort()
    middle = "z"
    if len(heap) == 0:
        res = "".join(tempL)
        return res + res[::-1]
    else :
        temp = heapq.heappop(heap)
        while len(heap) > 0:
            item = heapq.heappop(heap)
            cnt = item[2] // 2
            for i in range(cnt):
                tempL.append(item[1])
            if ord(item[1]) <= ord(middle):
                middle = item[1]
            #tempL.append([item[1] for i in range(cnt)])
        insertCnt = temp[2] // 2
        #tempL.append([temp[1] for i in range(insertCnt)])
        for i in range(insertCnt):
            tempL.append(temp[1])
        tempL.sort()
        res = "".join(tempL)
        if ord(temp[1]) <= ord(middle):
            middle = temp[1]
        res = res + middle + res[::-1]
        return res

class SolutionT25:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(-1)
        pre = dummy
        cur = pre
        num = 0
        while cur.next :
            num+=1
            cur = cur.next
        while num >= k:
            cur = pre.next
            for i in range(1, k):
                t = cur.next
                cur.next = t.next
                t.next = pre.next
                pre.next = t
                cur = cur.next
            pre = cur
            num -= k
        return dummy.next


class SolutionT82:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(-1)
        dummy.next = head
        pre = dummy
        while pre.next :
            cur = pre.next
            while cur.next and cur.val == cur.next.val:
                cur = cur.next
            if (cur != pre.next) :
                pre.next = cur.next
            else :
                pre = pre.next
        return dummy.next

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class SolutionT138:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return None
        dict = collections.defaultdict(Node)
        res = Node(head.val)
        dict[head] = res
        cur = head.next
        copy_cur = res
        while cur:
            t = Node(cur.val)
            copy_cur.next = t
            dict[cur] = t
            cur = cur.next
            copy_cur = copy_cur.next
        cur, copy_cur = head, res
        while cur:
            copy_cur.random = dict[cur.random]
            copy_cur = copy_cur.next
            cur = cur.next
        return res

class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next
        secondHead = slow.next
        slow.next = None
        t1 = self.sortList(head)
        t2 = self.sortList(secondHead)
        dummy = ListNode(0)
        pre = dummy
        while t1 and t2 :
            if t1.val < t2.val:
                pre.next = t1
                t1 = t1.next
            else:
                pre.next = t2
                t2 = t2.next
            pre = pre.next
        if t1:
            pre.next = t1
        if t2:
            pre.next = t2
        return dummy.next

class MyHashMap:

    def __init__(self):
        self.data = [[None] for i in range(1000001)]

    def put(self, key: int, value: int) -> None:
        self.data[key] = value

    def get(self, key: int) -> int:
        val = self.data[key]
        return val if val else -1

    def remove(self, key: int) -> None:         
        self.data[key] = None

class SolutionT128:
    def longestConsecutive(self, nums: List[int]) -> int:
        dict = collections.defaultdict(int)
        res = 0
        for num in nums:
            if num in dict:
                continue
            left = dict[num-1] if num - 1 in dict else 0
            right = dict[num+1] if num + 1 in dict else 0
            sum = left + right + 1
            dict[num] = sum
            res = max(res, sum)
            dict[num - left] = sum
            dict[num + right] = sum
        return res

class SolutionT953:
    def isAlienSorted(self, words, order):
        m = {c:i for i, c in enumerate(order)}
        words = [[m[c] for c in w] for w in words]
        return all(w1 <= w2 for w1, w2 in zip(words, words[1:]))

class SolutionT480:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        small = [] #minHeap put larget half
        large = [] #maxHeap put smaller half
        for i in range(k):
            if len(small) == len(large):
                heapq.heappush(small, -heapq.heappushpop(large, -nums[i]))
            else :
                heapq.heappush(large, -heapq.heappushpop(small, nums[i]))
        ans = [float(large[0])] if k & 1 else [(large[0] + small[0]) / 2.0]
        to_remove = defaultdict(int)
        for i in range(k, len(nums)) :
            heapq.heappush(large, -heapq.heappushpop(small, nums[i]))
            out_num = nums[i - k]
            to_remove[out_num]+=1
            while large and to_remove[large[0]]:
                to_remove[-large[0]]-=1
                heapq.heappop(large)
            while small and to_remove[small[0]]:
                to_remove[small[0]]-=1
                heapq.heappop(small)
            if k % 2:
                ans.append(float(small[0]))
            else:
                ans.append((small[0] + large[0]) / 2.0)

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        cur = head = ListNode(0)
        heap = []
        for l in lists:
            if l is not None:
                heapq.heappush(heap, (l.val, l))
        while len(heap) > 0 :
            _, top = heapq.heappop()
            cur.next = top
            cur = cur.next
            if top.next is not None:
                heapq.heappush(heap, (top.next.val, top.next))
        return head.next

class Solution:
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        stack, num, sign = [], 0, '+'
        
        for i in range(len(s)):
            
            if s[i].isdigit():
                num = (num * 10) + int(s[i])
            if s[i] in '+-*/' or i == len(s) - 1:
                
                if sign == '+':
                    stack.append(num)
                if sign == '-':
                    stack.append(-num)
                if sign == '*':
                    stack.append(stack.pop() * num)
                if sign == '/':
                    p = stack.pop()
                    res = abs(p) // num
                    stack.append(res if p >= 0 else -res)
                num = 0
                sign = s[i]
                
        return sum(stack)

class Solution:
    def calculate(self, s: str) -> int:
        stack, res, sign , n= [], 0, 1, len(s)
        for i in range(len(s)) :
            if s[i].isdigit():
                num = 0
                while i < n and s[i].isdigit():
                    num = num * 10 + int(s[i])
                    i += 1
                res += sign * num
                i -= 1
            elif s[i] == '+':
                sign = 1
            elif s[i] == '-':
                sign = -1
            elif s[i] == '(':
                stack.append(num)
                stack.append(sign)
                res = 0
                sign = 1
            elif s[i] == ')':
                res *= stack[-1]
                stack.pop()
                res += stack[-1]
                stack.pop()
        return res

class SolutionT1249:
    def minRemoveToMakeValid(self, s: str) -> str:
        str = list(s)
        stack = []
        for i in range(len(s)):
            if str[i] == "(":
                stack.append(i)
            elif str[i] == ")":
                if len(stack) == 0:
                    str[i] = ""
                else :
                    stack.pop()
        while len(stack) > 0 :
            str[stack[-1]] = ""
            stack.pop()
        return "".join(str)

class SolutionT394:
    def decodeString(self, s: str) -> str:
        res = ""
        strStack = []
        numStack = []
        num = 0
        cur = ""
        for i in range(len(s)):
            if s[i].isdigit():
                num = num * 10 + int(s[i])
            elif s[i].isalpha():
                cur = cur + s[i]
            elif s[i] == '[':
                strStack.append(cur)
                numStack.append(num)
                cur, num = "", 0
            elif s[i] == ']':
                t = cur * numStack[-1]
                numStack.pop()
                cur = strStack[-1] + t
                strStack.pop()
        return cur

class SolutionT84:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        res = float('-inf')
        stack.append(-1)
        for i in range(len(heights)) :
            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                tempH = heights[stack[-1]]
                idx = stack[-1]
                stack.pop()
                res = max(res, tempH * (i if len(stack) == 0 else i - idx - 1))
            stack.append(i)

        while stack[-1] != -1:
            curH = heights[stack[-1]]
            stack.pop()
            curWid = len(heights) - 1 - stack[-1]
            res = max(res, curH * curWid)
        return res

class Solution:
    def nextLargerNodes(self, head: Optional[ListNode]) -> List[int]:
        res = []
        stack = []
        index = 0
        while head is not None:
            res.append(0)
            curVal = head.val
            while len(stack) > 0 and curVal > stack[-1][0]:
                res[stack[-1][1]] = curVal
                stack.pop()
            stack.append([curVal, index])
            index += 1
            head = head.next
        return res

class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = -1, right = 0, res = 0
        while (right < len(nums)) :
            if nums[right] == 0:
                k-=1
            while k < 0 and left <= right:
                if nums[left] == 0 :
                    k+=1
                left+=1
            res = max(res, right - left + 1)
            right += 1
        return res

    def helper(largest, K):
        if flag == False:
            return -1
        if largest == 0:
            flag = False:
            return -1
        if K == 0:
            return 0
        if K > largest:
            t = largest
            visited[largest] = 1
            while visited[largest] == 1 :
                largest -= 1
            
            return 1 + helper(largest, K - t)
        elif K <= largest:
            largest = K
            while visited[largest] == 1:
                largest -= 1
            if largest == K:
                return 1
            else :
                ret

def solution(A):
    # write your code in Python 3.6
    dict = collections.defaultdict(list)
    res = float('-inf')
    for i in A:
        digitSum = 0
        t = i
        while t > 0:
            digitSum += t % 10
            t /= 10
        dict[digitSum].append(i)
        if dict[digitSum] >= 2:
            dict[digitSum].sort(reverse=True)
            res = max(res, dict[digitSum][0] + dict[digitSum][1])
    return res if res != float('-inf') else -1