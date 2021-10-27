import collections
from typing import Collection, List, Optional
from collections import *
import sys
import bisect
import heapq

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
        return count

root1 = TreeNode(1, 'A')
root1.children.append(TreeNode(2, 'B'))
root1.children.append(TreeNode(3, 'C'))
root1.children[0].children.append(TreeNode(4, 'D'))
root1.children[0].children.append(TreeNode(5, 'E'))
root1.children[0].children.append(TreeNode(6, 'F'))
root1.children[1].children.append(TreeNode(7, 'G'))
root2 = TreeNode(1, 'A')
root2.children.append(TreeNode(2, 'B'))
root2.children.append(TreeNode(33, 'C'))
root2.children[0].children.append(TreeNode(5, 'E'))
root2.children[0].children.append(TreeNode(4, 'D'))
root2.children[0].children.append(TreeNode(22, 'F'))
root2.children[1].children.append(TreeNode(7, 'G'))

obj = Solution()
print(obj.compute_diff(root1, root2))