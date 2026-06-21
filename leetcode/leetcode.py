import heapq
from collections import Counter
import math
from functools import reduce
from collections import deque

class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        freq = Counter(nums)

        min_heap = []
        for num, count in freq.items():
            if len(min_heap) < k:
                heapq.heappush(min_heap, (count, num))
            elif count > min_heap[0][0]:
                heapq.heapreplace(min_heap, (count, num))

        return [num for count, num in min_heap]
    
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        min = prices[0]
        for num in prices:
            if min < num:
                ans = max(ans, num - min)
            elif min > num:
                min = num
        return ans
    
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0
        
        ans = 0
        cur_end = 0
        cur_Farthest = 0
        for i in range(n-1):
            cur_Farthest = max(cur_Farthest, i + nums[i])
            if i == cur_end:
                ans += 1
                cur_end = cur_Farthest
                if cur_end >= n - 1:
                    break
                
        return ans
    
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        if n <= 1:
            return True
        
        cur_end = 0
        cur_farthest = 0

        for i in range(n-1):
            cur_farthest = max(cur_farthest, i + nums[i])

            if i == cur_end:
                cur_end = cur_farthest
                if cur_end >= n - 1:
                    return True
                
        return False
    

    def partitionLabels(self, s: str) -> List[int]:
        last: dict = {c: i for i,c in enumerate(s)}
        ans = []
        start = end = 0
        for i, c in enumerate(s):
            end = max(end, last[c])

            if i == end:
                ans.append(i -start + 1)
                start = i + 1

        return ans
    
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        if n == 2:
            return 2
        
        pre = {1: 1, 2: 2}
        for i in range(2, n):
            pre[i] = pre[i - 1] +  pre[i - 2]
        
        return pre[n-1]
    
    def generate(self, numRows: int) -> List[List[int]]:
        ans = []
        for i in range(numRows):
            cur = []
            for j in range(i+1):
                if j == 0 or j == i:
                    cur.append(1)
                else:
                    cur.append(ans[i-1][j-1] + ans[i-1][j])
            ans.append(cur)
        return ans
    
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        if len(nums) <= 2:
            return max(nums[0], nums[1])
        nums[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            nums[i] = max(nums[i] + nums[i-2], nums[i-1])
        return nums[len(nums) - 1]
    
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        ans = [-1, -1]
        if len(nums) == 0:
            return ans

        left = 0
        right = len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        if nums[left] != target:
            return ans
        start = left

        left = 0
        right = len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid
        end = left - 1

        return [start, end]
    
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            
            if nums[left] <= nums[mid]:
                if nums[left] <= target and target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target and target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1
    
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] < nums[right]:
                # 右半有序，最小值在左半或 mid 本身
                right = mid
            else:
                # 最小值在右半
                left = mid + 1
        return nums[left]

    def isValid(self, s: str) -> bool:
        dic = {")": "(", "}": "{", "]": "["}
        stack = []
        for c in s:
            if c in dic.values():
                stack.append(c)
            else:
                if not stack or stack.pop() != dic[c]:
                    return False
        return not stack
            

    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap = []
        for num in nums:
            if len(min_heap) <= k:
                heapq.heappush(min_heap, num)
            elif num > min_heap[0]:
                heapq.heapreplace(min_heap, num)

        return heapq.heappop(min_heap)

    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        def buildTreeCore(preorder_left, preorder_right, inorder_left, inorder_right):
            if preorder_left > preorder_right:
                return None
            
            preorder_root = preorder_left
            inorder_root = index[preorder[preorder_root]]

            root = TreeNode(preorder[preorder_root])

            left_subtree_size = inorder_root - inorder_left

            root.left = buildTreeCore(preorder_left + 1, preorder_left + left_subtree_size, inorder_left, inorder_root - 1)
            root.right = buildTreeCore(preorder_left + left_subtree_size + 1, preorder_right, inorder_root + 1, inorder_right)

            return root
        n = len(preorder)
        index = {key: value for value, key in enumerate(inorder)}
        return buildTreeCore(0, n-1, 0, n-1)
    
    def permute(self, nums: List[int]) -> List[List[int]]:
        def permuteCore(cur: List[int]):
            if len(cur) == len(nums):
                ans.append(cur[:])
                return
            for i, num in enumerate(nums):
                    
                if record[i]:
                    continue

                cur.append(num)
                record[i] = True

                permuteCore(cur)

                cur.pop()
                record[i] = False
            return


        record = [False] * len(nums)
        ans = []
        
        permuteCore([])

        return ans
        
    def numSquares(self, n: int) -> int:
        dp = [0] * (n+1)
        for i in range(1, n+1):
            dp[i] = i
            for j in range(1, int(math.sqrt(i)) + 1):
                square = j * j
                dp[i] = min(dp[i], dp[i - square] + 1)
        return dp[n]
    
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1 
    
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dic = set(wordDict)
        dp = [False] * (len(s) + 1)
        dp[0] = True
        n = len(s)

        for i in range(1, n+1):
            for j in range(i):
                if dp[j] and s[j:i] in dic:
                    dp[i] = True
                    break
        return dp[n]
    
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        dp = [1] * (n)
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
    
    def maxProduct(self, nums: List[int]) -> int:
        prev_max = prev_min = ans = nums[0]

        for num in nums[1:]:
            tmp = (num, num*prev_max, num*prev_min)
            prev_max = max(tmp)
            prev_min = min(tmp)

            ans = max(ans, prev_max)

        return ans
    
    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        if n<2:
            return False
        
        total = sum(nums)
        maxNum = max(nums)

        if total % 2 == 1 or maxNum > total / 2:
            return False
        
        target = total // 2

        dp = [[False] * (target + 1) for _ in range(n)]
        for i in range(n):
            dp[i][0] = True

        dp[0][nums[0]] = True
        for i in range(1, n):
            num = nums[i]
            for j in range(1, target+1):
                if j >= num:
                    dp[i][j] = dp[i-1][j] or dp[i][j-num]
                else:
                    dp[i][j] = dp[i-1][j]
        
        return dp[n-1][target]
    
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0]* n for _ in range(m) ]
        for i in range(m):
            dp[i][0] = 1
        for i in range(n):
            dp[0][i] = 1
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]

        return dp[m-1][n-1]
    
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        for i in range(1, m):
            grid[i][0] += grid[i-1][0]
        for i in range(1, n):
            grid[0][i] += grid[0][i-1]

        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] = min(grid[i-1][j], grid[i][j-1]) + grid[i][j]

        return grid[m-1][n-1]
    
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans = []
        tmp = []
        def subsetsCore(index: int):
            ans.append(tmp[:])
            for i in range(index, len(nums)):
                tmp.append(nums[i])
                subsetsCore(i+1)
                tmp.pop()
        
        subsetsCore(0)
        return ans
    
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n < 2:
            return s
        
        max_len = 1
        begin = 0


        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True

        for L in range(2, n+1):
            for i in range(n):
                j = L + i - 1
                if j >= n:
                    break

                if s[i] != s[j]:
                    dp[i][j] = False
                else:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i+1][j-1]

                if dp[i][j] and j - i + 1 > max_len:
                    max_len = j - i + 1
                    begin = i
        return s[begin: begin+max_len]
    
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m,n = len(text1), len(text2)
        dp = [[0] * (n+1) for _ in range(m+1)]

        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def minDistance(self, word1: str, word2: str) -> int:
        n, m = len(word1), len(word2)

        if n*m == 0:
            return n + m
        
        dp = [[0] * (m+1) for _ in range(n+1)]

        for i in range(n+1):
            dp[i][0] = i
        for j in range(m+1):
            dp[0][j] = j

        for i in range(1, n+1):
            for j in range(1, m+1):
                left = dp[i-1][j] + 1
                down = dp[i][j-1] + 1
                left_down = dp[i-1][j-1]
                if word1[i-1] != word2[j-1]:
                    left_down += 1
                dp[i][j] = min(left, down, left_down)

        return dp[n][m]
    
    def singleNumber(self, nums: List[int]) -> int:
        return reduce(lambda x, y : x ^ y, nums)
        
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        res = None

        for num in nums:
            if count == 0:
                res = num
            
            count += (1 if num == res else -1)

        return res
    
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        
        left, right = i + 1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    def numSquares(self, n: int) -> int:
        dp = [0] * (n+1)

        for i in range(1, n+1):
            dp[i] = i
            for j in range(1, int(math.sqrt(i)) + 1):
                dp[i] = min(dp[i], dp[i - j*j] + 1)

        return dp[n]
    
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [math.inf] * (amount + 1)
        dp[0] = 0

        for i in coins:
            for j in range(i, amount + 1):
                dp[i] = min(dp[i], dp[j - i] + 1)
        
        return dp[amount] if dp[amount] != math.inf else -1
    
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dic = set(wordDict)
        dp = [False] * (n+1)
        dp[0] = True
        for i in range(1, n+1):
            for j in range(0, i):
                if s[j:i] in dic:
                    dp[i] = dp[j]

        return dp[n]
    
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * (n+1)

        for i in range(n+1):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[j] + 1, dp[i])

        return max(dp)
    
    def maxProduct(self, nums: List[int]) -> int:
        premin = premax = ans = nums[0]
        for num in nums[1:]:
            premin = min(num, premin*num, premax * num)
            premax = max(num, premin*num, premax * num)

            ans = max(ans, premax)

        return ans

    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        if n<2:
            return False
        
        total = sum(nums)
        maxNum = max(nums)

        if total % 2 == 1 or maxNum > total / 2:
            return False
        
        target = total // 2

        dp = [[False] * (target + 1) for _ in range(n)]

        for i in range(n):
            dp[i][0] = True

        for i in range(n):
            for j in range(1, target+1):
                if j >= nums[i]:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j - nums[i]]
                else:
                    dp[i][j] = dp[i-1][j]
        
        return dp[n-1][target]
    
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        dic = {value: key for key, value in enumerate(inorder)}
        n = len(preorder)
        def buildTreeCore(preorder: List[int], inorder: List[int], preorderLeft:int, preorderRight:int, inorderLeft:int, inorderRight:int) -> Optional[TreeNode]:
            if(preorderLeft > preorderRight):
                return None
            
            inorderRootIndex = dic[preorder[preorderLeft]]

            root = TreeNode(inorder[inorderRootIndex])
            left_tree_len = inorderRootIndex - inorderLeft

            root.left = buildTreeCore(preorder, inorder, preorderLeft + 1, preorderLeft + left_tree_len, inorderLeft, inorderRootIndex - 1)
            root.right = buildTreeCore(preorder, inorder, preorderLeft + left_tree_len + 1, preorderRight, inorderRootIndex + 1, inorderRight)

            return root
        
        return buildTreeCore(preorder, inorder, 0, n-1, 0, n-1)
    
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        dic = {0: 1}
        def pathSumCore(root: Optional[TreeNode], curSum: int) -> int:
            if not root:
                return 0
            
            curSum += root.val

            ret = dic.get(curSum - targetSum, 0)
            dic[curSum] = dic.get(curSum, 0) + 1

            ret += pathSumCore(root.left, curSum)
            ret += pathSumCore(root.right, curSum)

            dic[curSum] -=1

            return ret
        
        return pathSumCore(root, 0)
    
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if not root or root == p or root == q:
            return root
        
        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)

        if l and r:
            return root
        
        return l or r
    
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        
        m = len(grid)
        n = len(grid[0])

        ans = 0

        def dfs(i: int, j: int):
            if(i < 0 or i >= m or j <0 or j >= n or grid[i][j] == '0'):
                return
            
            grid[i][j] = '0'

            dfs(i-1,j)
            dfs(i,j-1)
            dfs(i+1,j)
            dfs(i,j+1)

        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    ans += 1
                    dfs(i,j)

        return ans
    
    def orangesRotting(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return -1

        fresh = 0
        ans = 0
        que = deque()

        m = len(grid)
        n = len(grid[0])

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    fresh += 1

                if grid[i][j] == 2:
                    que.append((i,j))

        if fresh == 0:
            return 0
        
        directions = [(-1,0), (1,0), (0,-1), (0,1)]

        while que:
            size = len(que)
            for _ in range(size):
                x, y = que.popleft()
                for dx, dy in directions:
                    nx, ny = x+dx, y+dy

                    if nx < 0 or nx >=m or ny <0 or ny >=n or grid[nx][ny] == 0 or grid[nx][ny] == 2:
                        continue

                    grid[nx][ny] = 2
                    fresh -= 1
                    que.append((nx,ny))

            if que:
                ans += 1

        return ans if fresh == 0 else -1