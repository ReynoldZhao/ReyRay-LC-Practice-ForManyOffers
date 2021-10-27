from _typeshed import Self
from abc import abstractproperty
from ast import Index
import collections
from typing import Collection, List, Optional
from collections import *
import sys
import bisect
import heapq
import math

#Connected Group
class Solution(object):
    def numIslands(self, grid):
        if len(grid) == 0 : return 0
        row, col = len(grid), len(grid[0])
        self.count = sum(grid[i][j] == '1' for i in range(row) for j in range(col))
        root = [i for i in range(row * col)]

        def find(x):
            return root[x] if root[x] == x else find(root[x])
        
        def union(x, y):
            proot, qroot = find(x), find(y)
            if proot == qroot: return
            root[proot] = qroot
            self.count -= 1

        for i in range(row):
            for j in range(col):
                if grid[i][j] == '0':
                    continue
                index = i * col + j
                if j < col - 1 and grid[i][j+1] == '1':
                    union(index, index + 1)
                if i < row - 1 and grid[i + 1][j] == '1':
                    union(index, index + col)
        return self.count
    
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        dirX = [-1, 0, 1, 0]
        dirY = [0, 1, 0, -1]
        res = 0

        def dfs(i, j):
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == '0':
                return 
            grid[i][j] = '0'
            for x in range(4):
                dfs(i + dirX[x], j + dirY[x])

        for i in range(m) :
            for j in range(n):
                if grid[i][j] == '1':
                    dfs(i, j)
                    res+=1
        return res

#Web Pagination
def web(items, sortParameter, sortOrder, itemsPerpage, pageNumber):
    items.sort(key = lambda x : x[1], reversed=(sortOrder==1))
    start = itemsPerpage * pageNumber
    res = []
    if start >= len(items): return res
    for i in range(start, min(start + itemsPerpage, len(items))):
        res.append(items[i][0])
    return res


job i 
endTime[i]
startTime < endTime[i] discarded
job j
startTime[j] >= endTime[i]

job         0   1   2   3
starttime   1   2   3   3
endtime     3   4   5   6
        0
       / \
     1    2
    / \   / \
   2   X  3  X
  / \
 3   X
class SolutionT1235:
#递归

    #memo 存的是以i为起点，向后，最大的profit
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        jobs = list(zip(startTime, endTime, profit))
        #找结束时间最小，profit最大
        jobs.sort(key=lambda x:x[0])
        startTime = [t[0] for t in jobs]
        self.memo = [-1 for i in range(50001)]
        self.helper(jobs, 0, 0, 0)
        return self.helper(jobs, startTime, 0)
    

    def helper(self, jobs: List[(int)], startTime: List[int], pos: int) -> int:
        # startTime 应该就是这个递归要处理的，关于合不合规应该在递归函数里完成

        # 递归里接受的位置，应该是已经处理完了的，合规了的，而不需要再处理
        if pos >= len(jobs):
            return 0
        if self.memo[pos] != -1:
            return self.memo[pos]
        curProfit = jobs[pos][2]
        endTime = jobs[pos][1]
        nextIdx = bisect.bisect_left(startTime, endTime, pos + 1, len(jobs))
        profit1 = self.helper(jobs, startTime, nextIdx) + curProfit
        profit2 = self.helper(jobs, startTime, pos + 1)
        maxProfit = max(profit1, profit2)
        self.memo[pos] = maxProfit
        return maxProfit

    def helper(self, temp: List[(int)], start: int, endtime: int) -> int:
        if start >= len(temp):
            return 0
        if (self.memo[start] > 0): return self.memo[start]
        idx = start
        while idx < len(temp) and temp[idx][0] < endtime:
            idx += 1
        if (idx >= len(temp)): return 0
        curEndTime = temp[idx][1]
        curProfit = temp[idx][2]
        profit1 = curProfit + self.helper(temp, idx + 1, curEndTime)
        profit2 = self.helper(temp, idx + 1, endtime)
        maxProfit = max(profit1, profit2)
        self.memo[start] = maxProfit
        return maxProfit 

[0 1 2 3 4 5 6]
             *
maxProfit = 0
[0 1 2 3 4 5 6]
         *
              ^ Binary Search next unconflict job
        1.schduled      curProfit = profit[4] + memo[6] schduled 4 
         2.skip 4       cur maxmimum profit memo[4 + 1]
         memo[4] = max(curProfit, memo[4 + 1])
memo[0]
class SolutionT1235:
#迭代

    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        jobs = list(zip(startTime, endTime, profit))
        #找结束时间最小，profit最大
        jobs.sort(key=lambda x:x[0])
        startTime = [t[0] for t in jobs]
        self.memo = [-1 for i in range(50001)]
        return self.findMaxProfit(jobs, startTime)
    
    def finMaxProfit(self, jobs: List[(int)], startTime: List[int]):
        n = len(jobs)

        #其实还是每个位置向后找，只不过从最右边开始
        for i in range(n-1, -1, -1):
            curProfit = 0
            nextIdx = bisect.bisect_left(startTime, jobs[i][1])
            if nextIdx != n:
                curProfit = jobs[i][2] + self.memo[nextIdx]
            else:
                curProfit = jobs[i][2]
            
            if i == n - 1:
                self.memo[i] = curProfit
            else:
                self.memo[i] = max(curProfit, self.memo[i+1]) #cur是要，memo[i+1]是不要

        return self.memo[0]

class Solution:
#数据结构 堆

    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        jobs = list(zip(startTime, endTime, profit))
        jobs.sort()
        return self.findMaxProfit(jobs)
    
    def findMaxProfit(self, jobs: List[(int)]) -> int:
        n, maxProfit = len(jobs), 0
        heap = []
        for i in range(n):
            start, end, profit = jobs[i]
            
            while len(heap) > 0 and start >= heap[0][0]:
                maxProfit = max(maxProfit, heap[0][1])
                heapq.heappop(heap)
            
            heapq.heappush(heap, [end, profit + maxProfit])
        
        while len(heap) > 0 :
            maxProfit = max(maxProfit, heap[0][1])
            heapq.heappop(heap)
        
        return maxProfit

class TreeNode():
    def __init__(self, key=None, val=None, child = []):
        self.key = key
        self.val = val
        self.children = []

class Solution:
    def compute_diff(self, old_tree: TreeNode, new_tree: TreeNode)-> int:
        if not old_tree and not new_tree:
            return 0
        elif not old_tree:
            return self.count_nodes(new_tree)
        elif not new_tree:
            return self.count_nodes(old_tree)
        elif old_tree.key != new_tree.key:
            return self.count_nodes(old_tree) + self.count_nodes(new_tree)
        count = 0
        if old_tree.val != new_tree.val:
            count += 1
        new_tree_children = {c.key: c for c in new_tree.children}
        for old_child in old_tree.children:
            count += self.compute_diff(old_child, new_tree_children.pop(old_child.key, None))
        for _, remaining_new_tree_child in new_tree_children.items():
            count += self.count_nodes(remaining_new_tree_child)
        return count
    
    def count_nodes(self, node: TreeNode):
        if not node:
            return 0
        count = 1
        for c in node.children:
            count += self.count_nodes(c)
        return c
    
class Solution:
    def minSteps(self, s: str, t: str) -> int:
        s_bucket = [0 for i in range(26)]
        t_bucket = [0 for i in range(26)]
        for c in s:
            s_bucket[ord(c) - ord('a')]+=1 
        for c in t:
            t_bucket[ord(c) - ord('a')]+=1
        sum = 0
        for i in range(26):
            sum += t_bucket[i] - s_bucket[i] if t_bucket[i] - s_bucket[i] < 0 else 0
        return -sum

class Dijsktra:
    def ShortestPath(self, n: int, edges: List[(int)]) -> List[bool]:
        graph = collections.defaultdict(list)
        ans = ["NO" for i in range(n)]
        heap = [] #(weight, vertex)
        dist = [1e9 for i in range(n + 1)] # dist[i] 未加入点 到已加入点集合/到原始点的距离
        parent = [[] for i in range(n + 1)] #parent[v] 装的是最短时，从哪些点到点v, 用于
        map = collections.defaultdict(int) # 标记edge在答案中的位置
        i, u, v ,w = 0, 0, 0, 0
        for i, item in enumerate(edges):
            _from, to, weight = item
            graph[_from].append([to, weight])
            graph[to].append([_from, weight])
            map[(_from, to)] = i
            map[(to, _from)] = i
        heapq.heappush(heap, (0, 1))
        dist[1] = 0
        while len(heap) > 0:
            #取最小weight的vertex
            u = heap[0][1]
            heapq.heappop(heap)
            #松弛
            #邻接矩阵中，最小weight点能到的所有点
            for item in graph[u]:
                vertex = item[0]
                weight = item[1]
                if (dist[vertex] > dist[u] + weight):
                    #松弛成功，之前到点vertex的parent都不算最短
                    parent[vertex].clear()
                if (dist[vertex] >= dist[u] + weight):
                    dist[vertex] = dist[u] + weight
                    # 加入到vertex的最短行列
                    print(u)
                    print(parent[vertex])
                    parent[vertex].append(u)
                    heapq.heappush(heap, (dist[vertex], vertex))
        print(parent)
        
        def trace_path(n: int, parent: list[List[int]], ans: List[int], map):
            print(parent[n])
            for p in parent[n]:
                ans[map[(p, n)]] = "YES"
                trace_path(p, parent, ans, map)

        trace_path(n, parent, ans, map)

        return ans

(1,2,1)
(2,3,1)
(3,5,1)
(1,4,1)
(4,5,2)
(3,4,2)
(2,4,4)

4 5
(1,2,1),
(1,3,1),
(1,4,1),
(2,3,1),
(2,4,1)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class SolutionT124:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        ans = float('-inf')

        def dfs(root: Optional[TreeNode]) -> int:
            nonlocal ans
            if root is None:
                return 0
            left = max(dfs(root.left), 0)
            right = max(dfs(root.right), 0)
            curMax = max(left + root.val + right, max(left + root.val, right + root.val))
            ans = max(ans, curMax)
            return max(left + root.val, right + root.val)
        
        dfs(root)

        return ans

class MyTreeNode:
    def __init__(self, isLeaf = False, val = 0, left = None, right = None) -> None:
        self.isLeaf = isLeaf
        self.val = val
        self.left = left
        self.right = right
    
class SolutionMaxPath:
    def maxPath(self, root: Optional[MyTreeNode]) -> int:
        self.res = float('-inf')
        if root is None:
            return 0

        def helper(root: Optional[MyTreeNode]) -> MyTreeNode:
            if root is None:
                return MyTreeNode(False, 0)
            if root.left is None and root.right is None:
                return MyTreeNode(root.val, True if root.isLeaf else False)
            leftNode = helper(root.left)
            rightNode = helper(root.right)
            if leftNode.isLeaf and rightNode.isLeaf:
                curMax = max(leftNode.val + rightNode.val + root.val, root.val + max(leftNode.val, rightNode.val))
                self.res = max(self.res, curMax)
                return MyTreeNode(True, max(leftNode.val, rightNode.val) + root.val)
            
            if leftNode.isLeaf:
                return MyTreeNode(True, leftNode.val + root.val)
            elif rightNode.isLeaf:
                return MyTreeNode(True, rightNode.val + root.val)
            else:
                return MyTreeNode(False, 0)


        helper(root)
        return self.res

class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
        i, patches = 0, 0
        miss = 1
        while miss <= n:
            if i < len(nums) and nums[i] <= miss:
                miss += nums[i]
                i+=1
            else :
                miss += miss
                patches += 1
        return patches

class Interval:
    def __init__(self, start: int = None, end: int = None):
        self.start = start
        self.end = end

class SolutionT759:
    def employeeFreeTime(self, schedule: '[[Interval]]') -> '[Interval]':
        times = sorted([i for s in schedule for i in s], key=lambda x:x.start)
        res, end = [], times[0].end
        for i in times[1:]:
            if i.start > end:
                res.append(Interval(end, i.start))
            end = max(end, i.end)
        return res

class SolutionT163:
    def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[str]:
        # formats range in the requested format
        def formatRange(lower, upper):
            if lower == upper:
                return str(lower)
            return str(lower) + "->" + str(upper)

        result = []
        prev = lower - 1
        for i in range(len(nums) + 1):
            curr = nums[i] if i < len(nums) else upper + 1
            if prev + 1 <= curr - 1:
                result.append(formatRange(prev + 1, curr - 1))
            prev = curr
        return result

class SolutionT329:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        memo = [[0] * n for j in range(m)]
        visited = [[False] * n for j in range(m)]
        res = float('-inf')
        dirX = [0, 1, 0, -1]
        dirY = [1, 0, -1, 0]

        def isNotOut(i: int, j: int):
            if i < 0 or i >= m or j < 0 or j >= n:
                return False
            return True

        def dfs(i: int, j: int) -> int:
            if memo[i][j] > 0:
                return memo[i][j]
            curLen = 0
            visited[i][j] = True
            for p in range(4):
                nextX = i + dirX[p]
                nextY = j + dirY[p]
                if isNotOut(nextX, nextY) and matrix[nextX][nextY] == matrix[i][j] and visited[nextX][nextY] == False:
                    curLen = max(curLen, dfs(nextX, nextY))
            memo[i][j] = 1 + curLen
            return memo[i][j]


        for i in range(m):
            for j in range(n):
                res = max(res, dfs(i, j))
            
        return res
    
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        memo = [[0] * n for j in range(m)]
        res = 0
        dirX = [0, 1, 0, -1]
        dirY = [1, 0, -1, 0]

        def isNotOut(i: int, j: int):
            if i < 0 or i >= m or j < 0 or j >= n:
                return False
            return True
        
        def dfs(i: int, j: int, visited: List[List[int]]):
            visited[i][j] = True
            maxLen, val = 0, matrix[i][j]
            for p in range(4):
                nextX = i + dirX[p]
                nextY = j + dirY[p]
                if isNotOut(nextX, nextY) and not visited[nextX][nextY] and val == matrix[nextX][nextY]:
                    maxLen = max(maxLen, dfs(nextX, nextY, visited))
            visited[i][j] = False
            return maxLen+1

        for i in range(m):
            for j in range(n):
                visited = [[False] * n for j in range(m)]
                res = max(res, dfs(matrix, i, j, visited))
        return res
    
    #def dfs(self, matrix: List[List[int]], i: int, j: int, visited: List[List[int]]):

class solution:
    class City:
        def __init__(self, ) -> None:
            

[7, 9, 6],
[9, 9, 9],
[2, 9, 1]

[1, 1, 1, 2, 4],
[5, 1, 5, 3, 1],
[3, 4, 2, 1, 1]

[9, 9, 9, 9, 9, 9, 9],
[9, 9, 8, 9, 9, 9, 9],
[9, 9, 9, 12, 9, 9, 9],
[9, 9, 9, 12, 9, 9, 9],
[9, 9, 9, 12, 9, 9, 9],
[9, 9, 9, 12, 9, 9, 9]

class Solution:
    def nearestShop(self, city: List[List[str]], location: List[List[int]]) -> List[int]:
        dirX = [0, 1, 0, -1]
        dirY = [1, 0, -1, 0]
        m, n = len(city), len(city[0])
        dist = [[1e9] * n for i in range(m)]
        visited = [[False] * n for i in range(m)]
        res = []

        def isNotOut(i: int, j: int):
            if i < 0 or i >= m or j < 0 or j >= n or city[i][j] != ' ':
                return False
            if visited[i][j] == True:
                return False
            return True

        # def bfs(i: int, j: int, visited: List[List[bool]]):
        #     queue = collections.deque(list)
        #     queue.append([i, j])
        #     step = 1
        #     while len(queue) > 0:
        #         size = len(queue)
        #         for p in range(size):
        #             node = queue[0]
        #             queue.popleft()
        #             x, y = node[0], node[1]
        #             #visited[x][y] = True
        #             for p in range(4):
        #                 tx = x + dirX[p]
        #                 ty = y + dirY[p]
        #                 if isNotOut(tx, ty) and city[tx][ty] == ' ' and not visited[tx][ty]:
        #                     dist[tx][ty] = min(dist[tx][ty], abs(i - tx) + abs(j - ty))
        #                     #dist[tx][ty] = min(dist[tx][ty], step)
        #                     visited[tx][ty] = True
        #                     queue.append([tx, ty]) 
        #         step+=1

        queue = collections.deque()
        for i in range(m):
            for j in range(n):
                if city[i][j] == 'D':
                    # visited = [[False] * n for i in range(m)]
                    # bfs(i, j, visited)
                    dist[i][j] == 0
                    queue.append([i, j])

        while len(queue) > 0:
            size = len(queue)
            for p in range(size):
                node = queue[0]
                queue.popleft()
                x, y = node[0], node[1]
                #visited[x][y] = True
                for p in range(4):
                    tx = x + dirX[p]
                    ty = y + dirY[p]
                    if not isNotOut(tx, ty): continue
                    dist[tx][ty] = dist[x][y] + 1
                    visited[tx][ty] = True
                    queue.append([tx, ty])
                    # if isNotOut(tx, ty) and city[tx][ty] == ' ' and not visited[tx][ty]:
                    #     dist[tx][ty] = min(dist[tx][ty], abs(i - tx) + abs(j - ty))
                    #     #dist[tx][ty] = min(dist[tx][ty], step)
                    #     visited[tx][ty] = True
                    #     queue.append([tx, ty])     
        
        for q in location:
            res.append(dist[q[0]][q[1]])
        
        return res

['X', ' ', ' ', 'D', ' ', ' ', 'X', ' ', 'X'],
['X', ' ', 'X', 'X', ' ', ' ', ' ', ' ', 'X'],
[' ', ' ', ' ', 'D', 'X', 'X', ' ', 'X', ' '],
[' ', ' ', ' ', 'D', ' ', 'X', ' ', ' ', ' '],
[' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X'],
[' ', ' ', ' ', ' ', 'X', ' ', ' ', 'X', 'X']

[200, 200],
[1, 4],
[0, 3],
[5, 6],
[5, 8]

class SolutionT36:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        repeat = set()
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.': continue
                t = '(' + str(board[i][j]) + ')'
                row = str(i) + t
                col = t + str(j)
                cell = str(i//3) + t + str(j//3)
                if row in repeat or col in repeat or cell in repeat:
                    return False
                repeat.add(row)
                repeat.add(col)
                repeat.add(cell)
        return True

class SolutionT37:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        n = len(board)

        rows, cols, boxes = collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(set)

        for r in range(n):
            for c in range(n):
                if board[r][c] == '.':
                    continue

                v = int(board[r][c])
                rows[r].add(v)
                cols[c].add(v)
                boxes[(r // 3) * 3 + c // 3].add(v)


        def is_valid(r, c, v):
            box_id = (r // 3) * 3 + c // 3
            return v not in rows[r] and v not in cols[c] and v not in boxes[box_id]


        def backtrack(r, c):
            if r == n - 1 and c == n:
                return True
            elif c == n:
                c = 0
                r += 1

            # current grid has been filled
            if board[r][c] != '.':
                return backtrack(r, c + 1)

            box_id = (r // 3) * 3 + c // 3
            for v in range(1, n + 1):
                if not is_valid(r, c, v):
                    continue

                board[r][c] = str(v)
                rows[r].add(v)
                cols[c].add(v)
                boxes[box_id].add(v)

                if backtrack(r, c + 1):
                    return True

                # backtrack
                board[r][c] = '.'
                rows[r].remove(v)
                cols[c].remove(v)
                boxes[box_id].remove(v)

            return False


        backtrack(0, 0)

    def solveSudoku(self, board: List[List[str]]) -> None:
        n = len(board)

        rows, cols, boxes = collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(set)

        for r in range(n):
            for c in range(n):
                if board[r][c] == '.':
                    continue

                v = board[r][c]
                rows[r].add(v)
                cols[c].add(v)
                boxes[(r // 3) * 3 + c // 3].add(v)


        def is_valid(r, c, v):
            box_id = (r // 3) * 3 + c // 3
            return v not in rows[r] and v not in cols[c] and v not in boxes[box_id]

        def backtrack(r: int, c: int) :
            if r == n :
                return True
            if c >= n:
                return backtrack(r + 1, 0)

            if board[r][c] != '.':
                return backtrack(r, c + 1)

            box_id = (r // 3) * 3 + c // 3
            for i in range(ord('1'), ord('9') + 1):
                char = chr(i)
                if not is_valid(r, c, char): continue
                board[r][c] = char

                rows[r].add(v)
                cols[c].add(v)
                boxes[box_id].add(v)

                if backtrack(r, c + 1):
                    return True

                board[r][c] = '.'
                rows[r].remove(v)
                cols[c].remove(v)
                boxes[box_id].remove(v)
            return False

        backtrack(0, 0)

class Solution:

    def swapOnce(self, s: str, l: List[str], k: int) -> List[str]:
        res = []
        for t in l:
            idx, n = 0, len(s)
            if len(t) != n:
                continue
            compare = list(zip(s, t))
            temp = []
            for item in compare:
                if item[0] != item[1]:
                    temp.append(item)
            if len(temp) == 2 and temp[0][1] == temp[1][0] and temp[1][0] == temp[0][1]:
                res.append(t)
        return t




    def kAnagrams(self, s: str, l: List[str], k: int) -> List[str]:
        cnt = collections.Counter(s)
        res = []
        for t in l:
            cnt_t = collections.Counter(t)
            if sum((cnt - cnt_t).values()) <= k:
                res.append(t)
        return res

class Token:
    def __init__(self, day = 0, min = 0) -> None:
        self.day = day
        self.min = min
    
    def formatTime(self) -> str:
        minutes = self.min % 60
        minstr = "0"+str(minutes) if minutes >=0 and minutes <= 9 else str(minutes)
        hour = self.min//60
        hourstr = "0"+str(hour) if hour >=0 and hour <= 9 else str(hour)
        string = str(self.day) + hourstr + minstr
        return string

class Solution:

    def __init__(self) -> None:
        self.map = collections.defaultdict(int)
        self.map["mon"] = 1
        self.map["tue"] = 2
        self.map["wed"] = 3
        self.map["thu"] = 4
        self.map["fri"] = 5
        self.map["sat"] = 6
        self.map["sun"] = 7

    def convertToken(self, start: str, end: str):
        t1 = self.parseToken(start)
        t2 = self.parseToken(end)
        res = []
        if t2.day < t1.day:
            t2.day += 7
        print(t1.day, t1.min, t1.day * 24 * 60 + t1.min)
        print(t2.day, t2.min, t2.day * 24 * 60 + t2.min)
        while t1.day * 24 * 60 + t1.min <= t2.day * 24 * 60 + t2.min :
            res.append(t1.formatTime())
            t1.min += 5
            if (t1.min >= 24 * 60) :
                t1.day += 1
                t1.min = t1.min % (24 * 60)
        return res

    def parseToken(self, s: str):
        parts = s.split()
        day = self.map[parts[0]]
        min = 12* 60 if parts[2] == "pm" else 0
        hours = int(parts[1].split(":")[0])
        mins = int(parts[1].split(":")[1])
        mins = math.ceil(mins/5) * 5
        if mins == 60:
            mins = 0
            hours += 1
            if hours == 12 and parts[2] == "pm":
                min = 0
                hours = 0
                day += 1
                if day > 7:
                    day = day - 7
        Token(day, min + hours * 60 + mins)
        return Token(day, min + hours * 60 + mins)

class Solution:
    def countOrders(self, n: int) -> int:
        #slot {" "} * 2n
        #用recursion的方法，按顺序放
        #第一个位置肯定放pickup， 有当前pickup的数量 p 种放法，其余的位置有recursion(p - 1, d)种
        #所以就是 p * recursion(p - 1, d)
        #d同理

        #其实每次recursion就是在一个位置上放了个数
        def recursionSolve(p ,d):
            count = 0
            if p == 0:
                return math.factorial(d)
            
            #pickup 
            count += p * recursionSolve(p - 1, d + 1)

            #deliver:
            if d > 0:
                count += d * recursionSolve(p, d - 1)
            
            return count
        
        return recursionSolve(n, 0) % (1e9 + 7) 
            
    def displayOrders(self, n: int) -> List[List[str]]:
        res = []
        self.slot = []
        pool = [("p" + str(i)) for i in range(1, n+1)]

        def recursion(p , d, pool: List[str]):
            if p == 0 and d == 0:
                self.res.append(self.slot)
                return 
            size = len(pool)
            for i in range(size):
                item = self.pool[i]
                type = item[:1]
                id = int(item[1:])

                if type == 'P':
                    self.slot.append(item)
                    temp_pool = pool
                    temp_pool.append('D'+str(id))
                    temp_pool.remove(item)
                    recursion(p - 1, d + 1, temp_pool)
                    self.slot.pop()
                
                if type == 'D':
                    self.slot.append(item)
                    temp_pool = pool
                    temp_pool.remove(item)
                    recursion(p, d - 1, temp_pool)
                    self.slot.pop()
            return 

class Solution:      
    def displayOrders(self, n: int) -> List[List[str]]:
        self.res = []
        self.slot = []
        pool = ["P" + str(i) for i in range(1, n+1)]
        print(pool)
        def recursion(p , d, pool: List[str]):
            if p == 0 and d == 0:
                self.res.append(self.slot)
                return 
            size = len(pool)
            for i in range(size):
                item = pool[i]
                type = item[:1]
                id = int(item[1:])

                if type == 'P':
                    self.slot.append(item)
                    temp_pool = pool
                    temp_pool.append('D'+str(id))
                    temp_pool.remove(item)
                    #print(self.slot)
                    print(temp_pool)
                    recursion(p - 1, d + 1, temp_pool)
                    self.slot.pop()
                
                if type == 'D':
                    self.slot.append(item)
                    temp_pool = pool
                    temp_pool.remove(item)
                    #print(self.slot)
                    print(temp_pool)
                    recursion(p, d - 1, temp_pool)
                    self.slot.pop()
            return 
        
        recursion(n, 0, pool)

obj = Solution()
obj.displayOrders(4)
print(obj.res)

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        if not prerequisites:
            return [i for i in range(0, numCourses)]
        graph = collections.defaultdict(list)
        inDegree = collections.defaultdict(int)
        for i in range(numCourses):
            inDegree[i] = 0
        res = []
        for item in prerequisites:
            graph[item[1]].append(item[0])
            inDegree[item[0]] += 1
        queue = collections.deque()
        for key, val in inDegree.items():
            if val == 0:
                queue.append(key)
        while len(queue) > 0:
            curCourse = queue.pop()
            res.append(curCourse)
            for c in graph[curCourse]:
                inDegree[c]-=1
                if inDegree[c] == 0:
                    queue.append(c)
        flag = True
        for key, val in inDegree.items():
            if val != 0:
                flag = False
                break
        return res if flag else []

class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        graph = collections.defaultdict(set)
        res = n
        ret = []
        for edge in edges:
            graph[edge[0]].add(edge[1])
            graph[edge[1]].add(edge[0])
        queue = collections.deque()
        for key, val in graph.items():
            if len(val) == 1:
                queue.append(key)
        while n > 2:
            n - len(queue)
            size = len(queue)
            for i in range(size):
                item = queue.popleft()
                for neighbor in graph[item]:
                    graph[neighbor].remove(item)
                    if len(graph[neighbor]) == 1:
                        queue.append(neighbor)
        while len(queue) > 0:
            ret.append(queue.popleft())
        return ret

class Solution:
    def closestStraightCity(city: List[str], x: List[int], y: List[int], queries: List[str]):
        cities = collections.defaultdict(List)
        xMap = collections.defaultdict(List)
        yMap = collections.defaultdict(List)

        for i in range(len(city)):
            x, y, cityName = x[i], y[i], city[i]
            xMap[x].append([y, cityName])
            yMap[y].append([x, cityName])
            cities[cityName] = [x, y]
        
        for key in xMap.keys():
            xMap[key].sort()
        
        for key in yMap.keys():
            yMap[key].sort()

        res = []

        def getCloestCityOnAxis(list: List[List[int]], location: int):
            index = bisect.bisect_left(list, location)
            left_dist, right_dist = float('inf'), float('inf')
            left_city, right_city = [], []
            if index > 0:
                left_city = list[index - 1]
                left_dist = abs(left_city[0] - location)
            if index < len(list) - 1:
                right_city = list[index + 1]
                right_dist = abs(right_city[0] - location)
            if left_dist < right_dist:
                return [left_dist, left_city[1]]
            elif left_dist > right_dist:
                return [right_dist, right_city[1]]
            return []
        
        for q in queries:
            location = cities[q]
            xList = xMap[location[0]] #[[yXis, name]] same x index
            yList = yMap[location[1]] #[[xXis, name]] same y index
            if len(xList) == len(yList) == 1:
                res.append("None")
            elif len(xList) == 1:
                res.append(getCloestCityOnAxis(yList, location[0])[1])
            elif len(yList) == 1:
                res.append(getCloestCityOnAxis(xList, location[1])[1])
            else:
                rowCity = getCloestCityOnAxis(xList, location[1])
                colCity = getCloestCityOnAxis(yList, location[0])
                if rowCity[0] < colCity[0]:
                    res.append(rowCity[1])
                elif rowCity[0] > colCity[0]:
                    res.append(colCity[1])
                else:
                    res.append(rowCity if rowCity[1] < colCity else colCity)

class Solution:
    def numSubarrayBoundedMax(self, nums: List[int], left: int, right: int) -> int:
        def count(bound):
            ans, cur = 0, 0
            for i in nums:
                cur = cur + 1 if i <= bound else 0
                ans += cur
            return ans
        return count(right) - count(left-1)


class TreeNode: 
    def __init__(self, alive = False, val = 0, left = None, right = None) -> None: 
        self.alive = alive 
        self.val = val 
        self.left = left 
        self.right = right 
     
class SolutionMaxPath: 
    def maxPath(self, root: Optional[TreeNode]) -> int: 
        self.res = float('-inf') 
        if root is None: 
            return 0 
        
        def helper(root: Optional[TreeNode]) -> int: 
            if root is None: 
                return float('-inf')
            if not root.left and not root.right and root.alive: 
                return root.val
            
            leftVal = helper(root.left) 
            rightVal = helper(root.right) 

            if leftVal > float('-inf') and rightVal > float('-inf'):  
                if root.alive:
                    curMax = max(leftVal, rightVal) + root.val
                    self.res = max(self.res, curMax) 
                else :
                    curMax = leftVal + rightVal + root.val
                    self.res = max(self.res, curMax)
            elif leftVal > float('-inf') or rightVal > float('-inf'):
                self.res = max(self.res, root.val + leftVal, root.val + rightVal)

            leftMax = float('-inf') if leftVal == float('-inf') else leftVal + root.val
            rightMax = float('-inf') if rightVal == float('-inf') else rightVal + root.val

            if not root.alive:
                return max(leftMax, rightMax)
            else:
                return root.val
        helper(root)
        return self.res

#   0
#   |
#   1
#  / \
# 2   3

#   1
# / | \
# 0 2 3

#  2
#  |
#  1
# / \
# 0  3

#  3
#  |
#  1
# / \
# 0 2


#    3
# / / \ \ 
# 0 1 2  4
#         |
#         5

#    4
#  /    \
#  5     3
#      / | \
#      0 1 2

# 1: [[2:1], [3:2], [5:3]]
# 2: {[1:1], [3:1]}
# 3: {[1:2], [2:1], [4:1], [5:1]}
# 4: {[3:1], [5:1]}
# 5: {[1:3], [3:1], [4:1]}

# graph: 
# [
#     i: [[j, weight_i], [k ,weight_j]]
# ]

# dist: [] dist[i] minimum distance from origin to city i ( direct/undirect) 

# parent: [ [], [] ] parent[i] when is mminimun distance path, the current path is 
#         from which city to city i 

# map: {[edge]: Index}

# heap: [[weight, i]]



# heap{[0:weight, 1:node]}

# visited{1}
# unvisited{2, 3, 4, 5}
# dist[inf, 0, inf, inf, inf, inf] (1 - index)
# choose min weight city from heap 

# 1:
# city 1:
# for i in {2, 3, 5}
#     #relax
#     if dist[2] > dist[1] + distance 1 -> 2:
#             parent[2] clear
#             parent[2] + [1]
#             dist[2] = dist[1] + disance 1-> 2
#             heap + [1:weight, 2:city]

# parent[n] : [ , , ]
# map{[edge, index]}

#       Existing tree                                                   
#         a(1, T)                                                 
#       /         \                                                 
#     b(2, T)   c(3, T)                                   
#   /       \           \                                          
# d(4, T) e(5, T)      g(7, T)                       

#             New tree
#             a(1, T)
#           /        \                                             
#    b(2, T)         c(3, T)  
#    /    |    \           \    
# d(4, T) e(5, T) f(6, T)    g(7, F) 

# Recurse(){

#     N1 key != N2 key 
#     res += all node + children

#     N1 value != N2 value
#         res += 1

#     N2_children_map: {key1:c1, key2:c2, ...}

#     for c in N1_children:
#         c_key
#         N2_child = N2_children_map[c_key].pop()
        
#     }
# }

#            5             Node 5  left2 right 0
#        /      \
#      2           0       
#    /   \        /   
#   100*  50*    14* 

#   dfs(N1) -> MyTreeNode {
#       deal with N1

#       left = dfs(N1 -> left)
#       right = dfs(N1 -> right)

#       left isLeaf AND right isLeaf:
#         global_res = max( , N1 + left + right)
#         return MyTreeNode(True, N1 + left + right)

#     if leftNode.isLeaf: 
#         return MyTreeNode(True, leftNode.val + root.val) 
#     elif rightNode.isLeaf: 
#         return MyTreeNode(True, rightNode.val + root.val) 
#     else: 
#         return MyTreeNode(False, 0)

#       #global_res, N1 + left, N1 + right, N1 + left + right
#       return N1 + left, N1 + right,
#   }

#   MyTreeNode:
#     isLeaf bool
#     val int : accumulate value path 
#     left MyTreeNode
#     right MyTreeNode





