
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