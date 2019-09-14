**My Java Solutions for LeetCode problems**  

**If you are interested in [my Python Solutions](https://github.com/Yuol96/leetcode-notes), please visit [this repo](https://github.com/Yuol96/leetcode-notes) of mine.**

[TOC]
# Top Hits

## Top Interview Questions

### 136. Single Number
- [Link](https://leetcode.com/problems/single-number/)
- Tags: Bit Manipulation
- Stars: 1

#### XOR
```java
class Solution {
    public int singleNumber(int[] nums) {
        int temp = 0;
        for(int num: nums){
            temp ^= num;
        }
        return temp;
    }
}
```

### 283. Move Zeroes
- [Link](https://leetcode.com/problems/move-zeroes/)
- Tags: Array, Two pointers
- Stars: 2

#### Insertion Sort
Time: O(n^2)
Space: O(1)
```java
class Solution {
    public void moveZeroes(int[] nums) {
        for(int i=1; i<nums.length; i++){
            int curr = i;
            while(curr > 0 && nums[curr-1] == 0){
                int temp = nums[curr];
                nums[curr] = nums[curr-1];
                nums[curr-1] = temp;
                curr--;
            }
        }
    }
}
```

#### Slow-Fast two pointers
We only need to care about non-zero elements and fill the remaining array with zeros!

Time: O(n)
Space: O(1)
```java
class Solution {
    public void moveZeroes(int[] nums) {
        int i=0, j=0;
        for(;j<nums.length;j++){
            if(nums[j]!=0){
                nums[i] = nums[j];
                i++;
            }
        }
        while(i<nums.length){
            nums[i] = 0;
            i++;
        }
    }
}
```

### 206. Reverse Linked List
- [Link](https://leetcode.com/problems/reverse-linked-list/)
- Tags: Linked List
- Stars: 1

#### Iterative
```java
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode curr = null;
        while(head != null){
            ListNode p = head.next;
            head.next = curr;
            curr = head;
            head = p;
        }
        return curr;
    }
}
```

Another version by [大雪菜]
- time: 100%
- space: 98.92%
```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if (head == null) return null;
        ListNode prev = null, curr = head;
        while(curr != null) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }
}
```

#### Recursive
```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode newHead = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }
}
```

### 371. Sum of Two Integers
- [Link](https://leetcode.com/problems/sum-of-two-integers/)
- Tags: Bit Manipulation
- Stars: 3

#### 行波进位加法器
```java
class Solution {
    public int getSum(int a, int b) {
        int c = 0;
        int result = 0;
        for(int i=0; i<32; i++){
            int pos = (1<<i);
            int m = (a&pos), n = (b&pos);
            int g = (m&n), p = (m|n);
            result |= (m^n^c);
            c = (g | (p & c)) << 1;
        }
        return result;
    }
}
```

#### recursive 行波进位加法器
```java
class Solution {
    public int getSum(int a, int b) {
        if(b==0)
            return a;
        return getSum((a^b), (a&b)<<1);
    }
}
```

### 169. Majority Element
- [Link](https://leetcode.com/problems/majority-element/)
- Tags: Array, Divide and Conquer, Bit Manipulation
- Stars: 2

#### Heavy Guardian (Boyer-Moore Majority Vote Algorithm)
```java
class Solution {
    public int majorityElement(int[] nums) {
        int result = nums[0], count = 0;
        for(int num : nums){
            if(num == result) count++;
            else {
                count--;
                if(count <= 0) {
                    count = 1;
                    result = num;
                }
            }
        }
        return result;
    }
}
```

Updated 2019.8.10

```java
class Solution {
    public int majorityElement(int[] nums) {
        int candidate = nums[0], count = 1;
        for(int i=1; i<nums.length; i++) {
            if (nums[i] == candidate) count++;
            else {
                count--;
                if (count == 0) {
                    count = 1;
                    candidate = nums[i];
                }
            }
        }
        return candidate;
    }
}
```

#### Divide and Conquer
```java
class Solution {
    public int majorityElement(int[] nums) {
        return recurr(nums, 0, nums.length-1);
    }
    
    private int recurr(int[] nums, int l, int r) {
        if(l==r){
            return nums[l];
        }
        int mid = l + ((r-l)>>1);
        int a = recurr(nums, l, mid), b = recurr(nums, mid+1, r);
        if(a==b){
            return a;
        }
        return count(nums, l, r, a) > count(nums, l, r, b) ? a : b;
    }
    
    private int count(int[] nums, int l, int r, int target){
        int n = 0;
        for(int i=l; i<=r; i++){
            if(target == nums[i])
                n++;
        }
        return n;
    }
}
```

#### binary search
- attention: `r-l` might overflow, so you have to use long integer.

```java
class Solution {
    public int majorityElement(int[] nums) {
        // iterate to get max and min element
        long l=nums[0], r=nums[0];
        for(int num : nums){
            if(l > num) l = num;
            if(r < num) r = num;
        }
        // binary search by value
        while(l<r){
            int mid = (int)(l+((r-l)>>1));
            int count = getLTECount(nums, mid);
            if(count > (nums.length>>1)) r = mid;
            else l = mid+1;
        }
        return (int)l;
    }
    private int getLTECount(int[] nums, int target){
        int count = 0;
        for(int num : nums)
            if(num <= target) count++;
        return count;
    }
}
```

#### Bit Manipulation
majority的每一bit都应该是majority！
```java
class Solution {
    public int majorityElement(int[] nums) {
        int result = 0;
        for(int i=0, mask=1; i<32; i++, mask<<=1){
            int bitCount = 0;
            for(int j=0; j<nums.length; j++){
                if((nums[j]&mask)!=0) bitCount++;
                if(bitCount>nums.length/2) {
                    result |= mask;
                    break;
                }
            }
        }
        return result;
    }
}
```

Other Sub-optimal methods: Hash Table, Sorting (must appear at n/2 position), Randomization (random pick one and check if it is majority) 

#### Quick Selection to find the median
```java
class Solution {
    public int majorityElement(int[] nums) {
        int k = ((nums.length-1)>>1);
        int l=0, r=nums.length-1;
        while(l<r){
            int j = partition(nums, l, r);
            if(j == k) return nums[k];
            else if(j>k) r = j-1;
            else l = j+1;
        }
        return nums[k];
    }
    private int partition(int[] nums, int l, int r){
        int i=l, j=r+1;
        while(true){
            while(nums[++i] < nums[l] && i<r);
            while(nums[l] < nums[--j] && j>l);
            if(i>=j) break;
            swap(nums, i, j);
        }
        swap(nums, l, j);
        return j;
    }
    private void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

### 242. Valid Anagram
- [Link](https://leetcode.com/problems/valid-anagram/)
- Tags: Hash Table, Sort
- Stars: 1

#### alphabet counting
1. You don't need 2 alphabet!
2. Arrays automatically got initialized with zero values!
```java
class Solution {
    public boolean isAnagram(String s, String t) {
        int[] alphabet = new int[26];
        for(int i=0; i<s.length(); i++) alphabet[s.charAt(i)-'a']++;
        for(int i=0; i<t.length(); i++) alphabet[t.charAt(i)-'a']--;
        for(int num : alphabet) if(num!=0) return false;
        return true;
    }
}
```

### 268. Missing Number
- [Link](https://leetcode.com/problems/missing-number/)
- Tags: Array, Math, Bit Manipulation
- Stars: 2

#### sum (math)
This method might overflow when we have large amount of large numbers in `nums`!
```java
class Solution {
    public int missingNumber(int[] nums) {
        int n = nums.length;
        int sum = ((n*(n+1))>>1);
        for(int num: nums)
            sum -= num;
        return sum;
    }
}
```

#### XOR with both index and array element
```java
class Solution {
    public int missingNumber(int[] nums) {
        int result = nums.length;
        for(int i=0; i<nums.length; i++){
            result ^= (i ^ nums[i]);
        }
        return result;
    }
}
```

#### swap sort
Given a num in `nums`, one can easily know the postion that this num is supposed to be in. 

O(n) sort:  
```java
class Solution {
    public int missingNumber(int[] nums) {
        int last = -1;
        for(int i=0; i<nums.length; i++){
            while(nums[i] != i){
                if(nums[i] == -1) break;
                if(nums[i] == nums.length){
                    last = nums[i];
                    nums[i] = -1;
                }
                else swap(nums, i, nums[i]);
            }
        }
        for(int i=0; i<nums.length; i++){
            if(i != nums[i]) return i;
        }
        return nums.length;
    }
    private void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

### 121. Best Time to Buy and Sell Stock
- [Link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
- Tags: Array, Dynamic Programming
- Stars: 2

#### my original solution 20190201
不需要数组，保存当前的maxProfit和minPrice
```java
class Solution {
    public int maxProfit(int[] prices) {
        if(prices == null || prices.length == 0)
            return 0;
        int minPrice = prices[0];
        int maxProfit = 0;
        for(int i=1; i<prices.length; i++){
            if(minPrice > prices[i])
                minPrice = prices[i];
            maxProfit = Math.max(maxProfit, prices[i] - minPrice);
        }
        return maxProfit;
    }
}
```

#### DP
max subarray problem, using Kadane's Algorithm.
```java
class Solution {
    public int maxProfit(int[] prices) {
        if(prices.length == 0)
            return 0;
        int[] dp = new int[prices.length];
        // dp[i] means maxProfit we can get in the contiguous subarray ended up with prices[i]
        for(int i=1; i<prices.length; i++){
            dp[i] = Math.max(0, dp[i-1] + prices[i] - prices[i-1]);
        }
        int maxProfit = 0;
        for(int i=0; i<dp.length; i++)
            if(maxProfit < dp[i])
                maxProfit = dp[i];
        return maxProfit;
    }
}
```
The space of the algorithm above can be further optimized:
```java
class Solution {
    public int maxProfit(int[] prices) {
        int result = 0, dp = 0;
        for(int i=1; i<prices.length; i++){
            dp = Math.max(0, dp + prices[i] - prices[i-1]);
            result = Math.max(result, dp);
        }
        return result;
    }
}
```
Notice that we only care about differences of the prices array.

### 21. Merge Two Sorted Lists
- [Link](https://leetcode.com/problems/merge-two-sorted-lists/)
- Tags: Linked List
- Stars: 1

#### iterative (my soluton)
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode root = new ListNode(0), curr = root;
        while(l1 != null && l2 != null){
            if(l1.val < l2.val){
                curr.next = l1;
                l1 = l1.next;
            }
            else {
                curr.next = l2;
                l2 = l2.next;
            }
            curr = curr.next;
        }
        curr.next = l1!=null ? l1 : l2;
        return root.next;
    }
}
```

#### recursive 
```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null) return l2;
        if(l2 == null) return l1;
        ListNode root;
        if(l1.val < l2.val){
            root = l1;
            root.next = mergeTwoLists(l1.next, l2);
        }
        else {
            root = l2;
            root.next = mergeTwoLists(l1, l2.next);
        }
        return root;
    }
}
```

### 202. Happy Number
- [Link](https://leetcode.com/problems/happy-number/)
- Tags: Hash Table, Math
- Stars: 2

#### HashSet
```java
class Solution {
    public boolean isHappy(int n) {
        HashSet<Integer> st = new HashSet<Integer>();
        while(!st.contains(n)){
            if(n == 1)
                return true;
            st.add(n);
            String str = Integer.toString(n);
            n = 0;
            for(int i=0; i<str.length(); i++){
                int a = str.charAt(i) - '0';
                n += a*a;
            }
        }
        return false;
    }
}
```

#### Floyd Cycle detection algorithm
The best video to learn about Floyd Cycle detection : [https://www.youtube.com/watch?v=LUm2ABqAs1w](https://www.youtube.com/watch?v=LUm2ABqAs1w)

```java
class Solution {
    public boolean isHappy(int n) {
        int slow=n, fast=n;
        do {
            slow = digitsSquareSum(slow);
            fast = digitsSquareSum(fast);
            fast = digitsSquareSum(fast);
        }
        while(slow != fast);
        if(slow == 1) return true;
        return false;
    }
    
    private int digitsSquareSum(int n){
        int result = 0;
        while(n>0){
            int digit = (n%10);
            result += digit * digit;
            n /= 10;
        }
        return result;
    }
}
```

### 326. Power of Three
- [Link](https://leetcode.com/problems/power-of-three/)
- Tags: Math
- Stars: 4

#### Math
```java
class Solution {
    public boolean isPowerOfThree(int n) {
        // 1162261467 = 3**19 < 2**31-1 < 3**20
        return (n>0 && 1162261467%n == 0);
    }
}
```

#### binary search
```java
class Solution {
    public boolean isPowerOfThree(int n) {
        int l=0, r=19;
        while(l<=r){
            int mid = l + ((r-l)>>1);
            int power = (int)Math.pow(3, mid);
            if(power == n) return true;
            else if(power > n) r = mid-1;
            else l = mid+1;
        }
        return false;
    }
}
```

### 198. House Robber
- [Link](https://leetcode.com/problems/house-robber/)
- Tags: Dynamic Programming
- Stars: 1

#### DP iterative memo
```java
class Solution {
    public int rob(int[] nums) {
        if(nums.length == 0) return 0;
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        for(int i=1; i<nums.length; i++){
            dp[i] = nums[i];
            if(i-2>=0) dp[i] += dp[i-2];
            if(i-3>=0) dp[i] = Math.max(dp[i], dp[i-3]+nums[i]);
        }
        int result = dp[nums.length-1];
        if(nums.length-2 >=0 && dp[nums.length-2] > result) result = dp[nums.length-2];
        return result;
        // dp[i] = Math.max(dp[i-2]+nums[i], dp[i-3]+nums[i]);
    }
}
```

#### DP iterative + 2 variables
```java
class Solution {
    public int rob(int[] nums) {
        if(nums.length == 0)
            return 0;
        if(nums.length == 1)
            return nums[0];
        int a = nums[0], b = nums[1];
        if(nums.length == 2)
            return Math.max(a, b);
        int c = a + nums[2];
        for(int i=3; i<nums.length; i++){
            int temp = Math.max(a+nums[i], b+nums[i]);
            a = b;
            b = c;
            c = temp;
        }
        return Math.max(b, c);
    }
}
```

#### DP recursive
```java
class Solution {
    private HashMap<Integer, Integer> map;
    public Solution() {
        map = new HashMap<Integer, Integer>();
    }
    public int rob(int[] nums) {
        return rob(nums, nums.length-1);
    }
    public int rob(int[] nums, int i){
        if(i < 0)
            return 0;
        if(i == 0)
            return nums[0];
        if(i == 1)
            return Math.max(nums[0], nums[1]);
        if(map.containsKey(i))
            return map.get(i);
        map.put(i, Math.max(rob(nums, i-1), rob(nums, i-2) + nums[i]));
        return map.get(i);
    }
}
```

### 213. House Robber II
- [Link](https://leetcode.com/problems/house-robber-ii/)
- Tags: Dynamic Programming
- Stars: 1

#### 2 pass House Robber I
- time: 100%
- space: 100%
- interviewLevel
- attention: It needs to be taken care of when `nums` only has one element.

```java
class Solution {
    public int rob(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        return Math.max(rob(nums, 0, nums.length-2), rob(nums, 1, nums.length-1));
    }
    public int rob(int[] nums, int l, int r) {
        if (l > r) return 0;
        int[] dp = new int[nums.length];
        int result = 0;
        for(int i=l; i<=r; i++) {
            if (i < l+2) dp[i] = nums[i];
            else if (i == l+2) dp[i] = nums[i-2] + nums[i];
            else dp[i] = Math.max(dp[i-2], dp[i-3]) + nums[i];
            result = Math.max(result, dp[i]);
        }
        return result;
    }
}
```

### 337. House Robber III
- [Link](https://leetcode.com/problems/house-robber-iii/)
- Tags: Tree, DFS
- Stars: 4

#### DFS
```java
class Solution {
    public int rob(TreeNode root) {
        return rob(root, false);
    }
    public int rob(TreeNode root, boolean parentRobbed){
        if(root == null) return 0;
        if(parentRobbed) return rob(root.left, false) + rob(root.right, false);
        return Math.max(rob(root.left, false) + rob(root.right, false), 
                        rob(root.left, true) + rob(root.right, true) + root.val);
    }
}
```

Updated 2019.8.28
- time: 34.22%
- space: 22%
- language: `System.identityHashCode(Object o)`
- attention: This method does not differentiate return values from child tree under different circumstances. Thus, it is much slower.
```java
class Solution {
    Map<Pair, Integer> map = new HashMap<>();
    public int rob(TreeNode root) {
        return rob(root, 2);
    }
    public int rob(TreeNode root, int distance) {
        if (root == null) return 0;
        Pair p = new Pair(root, distance);
        if (map.containsKey(p)) return map.get(p);
        int ret = 0;
        if (distance == 0) ret = root.val + rob(root.left, 1) + rob(root.right, 1);
        else if (distance == 1) ret = rob(root.left, 2) + rob(root.right, 2);
        else {
            ret = rob(root, 0);
            ret = Math.max(ret, rob(root.left, 0) + rob(root.right, 0));
            ret = Math.max(ret, rob(root.left, 0) + rob(root.right, 1));
            ret = Math.max(ret, rob(root.left, 1) + rob(root.right, 0));
        }
        map.put(p, ret);
        return ret;
    }
    private class Pair {
        TreeNode node;
        int distance;
        public Pair(TreeNode n, int d) {node = n; distance = d;}
        public int hashCode() {
            return System.identityHashCode(node) + Integer.hashCode(distance);
        }
        public boolean equals(Object o) {
            Pair p = (Pair)o;
            return this.node == p.node && this.distance == p.distance;
        }
    }
}
```

#### DFS optimized (memo)
```java
class Solution {
    public int rob(TreeNode root) {
        Tuple tup = DFS(root);
        return Math.max(tup.robRoot, tup.notRobRoot);
    }
    private Tuple DFS(TreeNode root) {
        if(root == null) return new Tuple(0,0);
        Tuple l = DFS(root.left);
        Tuple r = DFS(root.right);
        int robRoot = root.val + l.notRobRoot + r.notRobRoot;
        int notRobRoot = Math.max(l.notRobRoot + r.notRobRoot, 
                                  Math.max(l.notRobRoot + r.robRoot, 
                                           Math.max(l.robRoot + r.notRobRoot, 
                                                    l.robRoot + r.robRoot)));
        return new Tuple(robRoot, notRobRoot);
    }
}
public class Tuple {
    int robRoot, notRobRoot;
    Tuple(int a, int b){
        robRoot = a;
        notRobRoot = b;
    }
}
```

Updated 2019.8.28
- time: 94.79%
- space: 86.11%
- language: init an array with values. `new int[]{0,0,0}`
- attention: during dfs, you must not ignore `l[1]+r[1]` (i.e. `l.notRobRoot + r.notRobRoot`)
- reference: https://leetcode.com/problems/house-robber-iii/discuss/79330/Step-by-step-tackling-of-the-problem
```java
class Solution {
    public int rob(TreeNode root) {
        int[] temp = dfs(root);
        return Math.max(temp[0], temp[1]);
    }
    public int[] dfs(TreeNode root) {
        if (root == null) return new int[]{0, 0};
        int[] l = dfs(root.left);
        int[] r = dfs(root.right);
        int robRoot = root.val+l[1]+r[1],
            notRobRoot = Math.max(Math.max(l[0]+r[0], l[1]+r[1]), 
                                  Math.max(l[0]+r[1], l[1]+r[0]));
        return new int[]{robRoot, notRobRoot};
    }
}
```

### 66. Plus One
- [Link](https://leetcode.com/problems/plus-one/)
- Tags: Array, Math
- Stars: 1

#### 数组初始化
注意：默认初始化，数组元素相当于对象的成员变量，默认值跟成员变量的规则一样。**数字0**，布尔false，char\u0000，引用：null

本题不适合把`Arrays.asList()`转化为List, `.asList`方法不适用于基本数据类型（byte, short, int, long, float, double, boolean）
```java
class Solution {
    public int[] plusOne(int[] digits) {
        for(int i=digits.length-1; i>=0; i--){
            if(digits[i]<9){
                digits[i]++;
                return digits;
            }
            digits[i] = 0;
        }
        int[] ret = new int[digits.length+1];
        ret[0] = 1;
        return ret;
    }
}
```

### 172. Factorial Trailing Zeroes
- [Link](https://leetcode.com/problems/factorial-trailing-zeroes/)
- Tags: Math
- Stars: 3

#### Increment (Time Limit Exceeded)
Time: O(n)
```java
class Solution {
    public int trailingZeroes(int n) {
        int count = 0;
        for(int i=1; i<=n; i++){
            int temp = i;
            while(temp%5 == 0 && temp>0){
                count++;
                temp /= 5;
            }
        }
        return count;
    }
}
```

#### Recursive
1\*2\*3 --multiply by three 5-> 1\*2\*3\*4\***5**\*6\*7\*8\*9\***10**\*11\*12\*13\*14\***15**

Time: O(logn)
```java
class Solution {
    public int trailingZeroes(int n) {
        if(n<5)
            return 0;
        return trailingZeroes(n/5) + n/5;
    }
}
```

#### Iterative
Similar to the Recursive method
```java
class Solution {
    public int trailingZeroes(int n) {
        int count = 0;
        while(n>4){
            n /= 5;
            count += n;
        }
        return count;
    }
}
```

### 155. Min Stack
- [Link](https://leetcode.com/problems/min-stack/)
- Tags: Stack, Design
- Stars: 2

#### Use two stacks
Store series of minValue into another stack to obtain O(1) time!
```java
class MinStack {
    Stack<Integer> minst, numst;

    public MinStack() {
        minst = new Stack<Integer>();
        numst = new Stack<Integer>();
    }
    
    public void push(int x) {
        numst.push(x);
        if(minst.empty()) minst.push(x);
        else{
            minst.push(Math.min(minst.peek(), x));
        }
    }
    
    public void pop() {
        minst.pop();
        numst.pop();
    }
    
    public int top() {
        return numst.peek();
    }
    
    public int getMin() {
        return minst.peek();
    }
}
```

#### only use one Stack
1. Use only one stack by storing the gap between min value and current value in it. 
2. Since we store differences of integers, we need to convert it into `Long`. 
```java
class MinStack {
    long min;
    Stack<Long> st;

    public MinStack() {
        st = new Stack<Long>();
        min = Integer.MAX_VALUE;
    }
    
    public void push(int x) {
        st.push(x-min);
        if(x<min)
            min = x;
    }
    
    public void pop() {
        long temp = st.pop();
        if(temp<0)
            min -= temp;
    }
    
    public int top() {
        long temp = st.peek();
        if(temp<0)
            return (int)min;
        return (int)(temp + min);
    }
    
    public int getMin() {
        return (int)min;
    }
}
```


### 234. Palindrome Linked List
- [Link](https://leetcode.com/problems/palindrome-linked-list/)
- Tags: Linked List, Two Pointers
- Stars: 1

#### halve and reverse
```java
class Solution {
    public boolean isPalindrome(ListNode head) {
        if(head == null) return true;
        
        int count = countListNode(head);
        ListNode mid = moveToMid(head, count);
        if(count%2==0){
            ListNode temp = mid.next;
            mid.next = null;
            mid = temp;
        }
        ListNode reverse = getReversedList(mid);
        while(reverse!=null && head!=null){
            if(reverse.val != head.val)
                return false;
            reverse = reverse.next;
            head = head.next;
        }
        return true;
    }
    private int countListNode(ListNode head){
        int count = 0;
        while(head!=null){
            count++;
            head = head.next;
        }
        return count;
    }
    private ListNode moveToMid(ListNode head, int count){
        ListNode p = head;
        for(int i=0; i<count/2-1; i++){
            p = p.next;
        }
        if(count%2==1) p = p.next;
        return p;
    }
    private ListNode getReversedList(ListNode head){
        ListNode newhead = null;
        while(head!=null){
            ListNode temp = head.next;
            head.next = newhead;
            newhead = head;
            head = temp;
        }
        return newhead;
    }
}
```

### 14. Longest Common Prefix
- [Link](https://leetcode.com/problems/longest-common-prefix/)
- Tags: String
- Stars: 1

#### compare chars in each position
```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        if(strs.length==0) return "";
        int count = 0, minLen = Integer.MAX_VALUE;
        for(String s: strs)
            if(minLen>s.length())
                minLen = s.length();
        while(count<minLen){
            char c = strs[0].charAt(count);
            for(int i=1; i<strs.length; i++){
                if(strs[i].charAt(count)!=c)
                    return strs[0].substring(0, count);
            }
            count++;
        }
        return strs[0].substring(0, count);
    }
}
```
#### String.indexOf
```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        if(strs == null || strs.length == 0)    return "";
        String pre = strs[0];
        int i = 1;
        while(i < strs.length){
            while(strs[i].indexOf(pre) != 0)
                pre = pre.substring(0,pre.length()-1);
            i++;
        }
        return pre;
    }
}
```
#### sort and compare the first and last String
```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        if(strs == null || strs.length == 0)    return "";
        Arrays.sort(strs);
        int count = 0;
        String a=strs[0], b=strs[strs.length-1];
        for(int i=0; i<a.length(); i++){
            if(b.length()>i && b.charAt(i)==a.charAt(i))
                count++;
            else
                return a.substring(0, count);
        }
        return a.substring(0, count);
    }
}
```

### 160. Intersection of Two Linked Lists
- [Link](https://leetcode.com/problems/intersection-of-two-linked-lists/)
- Tags: Linked List
- Stars: 1

#### turning into a loop
We don't need to know the length of each lists. We just want to ensure that two pointers reach the intersection point at the same time. 

Notice that `a` and `b` will eventually be `null` if the two linked lists have no intersection. Therefore, we have no need to worry about infinite loop problem. 
```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode a = headA, b = headB;
        while(a != b) {
            a = a == null ? headB : a.next;
            b = b == null ? headA : b.next;
        }
        return a;
    }
}
```

#### get lengths and eliminate differences
```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int lenA = getLen(headA), lenB = getLen(headB);
        while(lenA>lenB){
            headA = headA.next;
            lenA--;
        }
        while(lenB>lenA){
            headB = headB.next;
            lenB--;
        }
        while(headA!=headB){
            headA = headA.next;
            headB = headB.next;
        }
        return headA;
    }
    private int getLen(ListNode head){
        int count = 0;
        ListNode p = head;
        while(p!=null){
            p = p.next;
            count++;
        }
        return count;
    }
}
```

### 28. Implement strStr()
- [Link](https://leetcode.com/problems/implement-strstr/)
- Tags: Two Pointers, String
- Stars: 2

#### KMP
[原理讲解(忽略代码部分)](https://m.toutiaocdn.com/group/6578243698759303688/?iid=59744622620&app=news_article&timestamp=1549524948&group_id=6578243698759303688&tt_from=copy_link&utm_source=copy_link&utm_medium=toutiao_ios&utm_campaign=client_share)
```java
class Solution {
    public int strStr(String haystack, String needle) {
        int[] next = getNextArray(needle);
        int i=0, j=0;
        while(j<needle.length() && i<haystack.length()){
            while(haystack.charAt(i)!=needle.charAt(j) && j>0){
                j = next[j];
            }
            if(haystack.charAt(i)==needle.charAt(j))
                j++;
            i++;
        }
        if(j==needle.length())
            return i-needle.length();
        return -1;
    }
    private int[] getNextArray(String s){
        int[] next = new int[s.length()];
        for(int i=2; i<s.length(); i++){
            int maxCommonLen = next[i-1];
            while(maxCommonLen>0 && s.charAt(i-1) != s.charAt(maxCommonLen)){
                maxCommonLen = next[maxCommonLen];
            }
            if(s.charAt(i-1) == s.charAt(maxCommonLen))
                next[i] = maxCommonLen+1;
        }
        return next;
    }
}
```

### 190. Reverse Bits
- [Link](https://leetcode.com/problems/reverse-bits/)
- Tags: Bit Manipulation
- Stars: 1

#### move bit one by one
```java
public class Solution {
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        int result = 0;
        for(int i=0; i<32; i++){
            result |= ((n&1)<<(31-i));
            n = n>>>1;
        }
        return result;
    }
}
```

#### divide and conquer
```java
public class Solution {
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        n = (n>>>16) | (n<<16);
        n = ((n&0xFF00FF00)>>>8) | ((n&0x00FF00FF)<<8);
        n = ((n&0xF0F0F0F0)>>>4) | ((n&0x0F0F0F0F)<<4);
        n = ((n&0xCCCCCCCC)>>>2) | ((n&0x33333333)<<2);
        n = ((n&0xAAAAAAAA)>>>1) | ((n&0x55555555)<<1);
        return n;
    }
}
```

### 189. Rotate Array
- [Link](https://leetcode.com/problems/rotate-array/)
- Tags: Array
- Stars: 1

#### rotate partially
- attention: `k` needs to be reduced to [0, nums.length).

```java
class Solution {
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        rotate(nums, 0, nums.length-k-1);
        rotate(nums, nums.length-k, nums.length-1);
        rotate(nums, 0, nums.length-1);
    }
    private void rotate(int[] nums, int l, int r){
        while(l<r)
            swap(nums, l++, r--);
    }
    private void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

### 238. Product of Array Except Self
- [Link](https://leetcode.com/problems/product-of-array-except-self/)
- Tags: Array
- Stars: 1

#### Use only one array
```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int[] left = new int[nums.length];
        // int[] right = new int[nums.length];
        left[0] = 1;
        for(int i=1; i<nums.length; i++)
            left[i] = left[i-1]*nums[i-1];
        // right[nums.length-1] = 1;
        int right = 1;
        for(int i=nums.length-2; i>=0; i--){
            // right[i] = right[i+1]*nums[i+1];
            right *= nums[i+1];
            left[i] *= right;
        }
        // for(int i=0; i<nums.length; i++)
        //     left[i] *= right[i];
        return left;
    }
}
```

### 347. Top K Frequent Elements
- [Link](https://leetcode.com/problems/top-k-frequent-elements/)
- Tags: Hash Table, Heap
- Stars: 3

#### HashMap
```java
class Solution {
    public List<Integer> topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        for(int num: nums)
            map.put(num, map.getOrDefault(num, 0) + 1);
        HashMap<Integer, List<Integer>> freq2list = new HashMap<Integer, List<Integer>>();
        for(int num: map.keySet()){
            int freq = map.get(num);
            if(freq2list.get(freq)==null)
                freq2list.put(freq, new ArrayList<Integer>());
            freq2list.get(freq).add(num);
        }
        List<Integer> result = new ArrayList<Integer>();
        for(int i=nums.length; i>=1 && k>0; i--){
            if(freq2list.containsKey(i)){
                result.addAll(freq2list.get(i));
                k -= freq2list.get(i).size();
            }
        }
        return result;
    }
}
```

#### maxHeap and Map.Entry
```java
class Solution {
    public List<Integer> topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        for(int num: nums)
            map.put(num, map.getOrDefault(num, 0) + 1);
        PriorityQueue<Map.Entry<Integer, Integer>> maxHeap = new PriorityQueue<>((a,b)->(b.getValue()-a.getValue()));
        for(Map.Entry<Integer, Integer> entry: map.entrySet()){
            maxHeap.add(entry);
        }
        List<Integer> result = new ArrayList<Integer>();
        while(k>0){
            result.add(maxHeap.poll().getKey());
            k--;
        }
        return result;
    }
}
```

#### TreeMap
```java
class Solution {
    public List<Integer> topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        for(int num: nums)
            map.put(num, map.getOrDefault(num, 0) + 1);
        TreeMap<Integer, List<Integer>> freq2list = new TreeMap<>();
        for(int num: map.keySet()){
            int freq = map.get(num);
            if(freq2list.get(freq) == null)
                freq2list.put(freq, new ArrayList<Integer>());
            freq2list.get(freq).add(num);
        }
        List<Integer> result = new ArrayList<Integer>();
        while(k>0){
            Map.Entry<Integer, List<Integer>> entry = freq2list.pollLastEntry();
            result.addAll(entry.getValue());
            k -= entry.getValue().size();
        }
        return result;
    }
}
```

### 384. Shuffle an Array
- [Link](https://leetcode.com/problems/shuffle-an-array/)
- Tags: Design
- Stars: 4

#### swap step by step
```java
class Solution {
    private int[] arr;
    public Solution(int[] nums) {
        arr = nums;
    }
    public int[] reset() {
        return arr;
    }
    public int[] shuffle() {
        if(arr==null) return null;
        int[] newArr = arr.clone();
        Random rand = new Random();
        for(int i=newArr.length-1; i>=1; i--){
            int randpos = rand.nextInt(i+1);
            swap(newArr, i, randpos);
        }
        return newArr;
    }
    private static void swap(int[] nums,int i,int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

Or in the reverse order:

```java
public class Solution {
    private int[] nums;
    private Random random;
    public Solution(int[] nums) {
        this.nums = nums;
        random = new Random();
    }
    public int[] reset() {
        return nums;
    }
    public int[] shuffle() {
        if(nums == null) return null;
        int[] a = nums.clone();
        for(int j = 1; j < a.length; j++) {
            int i = random.nextInt(j + 1);
            swap(a, i, j);
        }
        return a;
    }
    private void swap(int[] a, int i, int j) {
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}
```

### 378. Kth Smallest Element in a Sorted Matrix
- [Link](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)
- Tags: Binary Search, Heap
- Stars: 3

#### Max Heap
```java
public class Solution {
    public int kthSmallest(int[][] matrix, int k) {
        int n = matrix.length;
        PriorityQueue<Tuple> qu = new PriorityQueue<>();
        for(int i=0; i<n; i++){
            qu.add(new Tuple(0, i, matrix[0][i]));
        }
        Tuple temp = qu.peek();
        for(int i=0; i<k; i++){
            temp = qu.poll();
            if(temp.row < n-1)
                qu.add(new Tuple(temp.row+1, temp.col, matrix[temp.row+1][temp.col]));
        }
        return temp.val;
    }
}
class Tuple implements Comparable<Tuple> {
    int row, col, val;
    public Tuple(int x, int y, int v){
        row = x; col = y; val = v;
    }
    @Override
    public int compareTo(Tuple o){
        return this.val - o.val;
    }
}
```


#### Binary Search
- attention: when `count == k`, `mid` might not exists in `matrix`, so we need to get the largest element that is less than or equal to `mid` in `matrix`. Therefore, we have `getMaxlte`.

<span id="378-binary-search"></span>
1. There's a situation that might break the while loop, i.e., there are more than one elements that have the same value as the kth smallest. When this happens, r will goes below l, and it breaks the while loop. Therefore, we need to return `l` instead of an arbitrary number outside the while loop. 
2. The whole picture of this algorithm:
> The key point for any binary search is to figure out the "Search Space". For me, I think there are two kind of "Search Space" -- index and range(the range from the smallest number to the biggest number). Most usually, when the array is sorted in one direction, we can use index as "search space", when the array is unsorted and we are going to find a specific number, we can use "range". 

Similar to [287. Find the Duplicate Number](#287-binary-search).

```java
class Solution {
    public int kthSmallest(int[][] matrix, int k) {
        int n = matrix.length;
        int l = matrix[0][0], r = matrix[n-1][n-1];
        while(l<=r){
            int mid = l + ((r-l)>>1);
            int count = countlte(matrix, mid);
            if(count == k)
                return getMaxlte(matrix, mid);
            else if(count > k)
                r = mid - 1;
            else 
                l = mid + 1;
        }
        return l;
    }
    private int countlte(int[][] matrix, int target){
        int n = matrix.length, count = 0;
        for(int[] row: matrix){
            int j = n;
            while(j>0 && row[j-1] > target)
                j--;
            count += j;
        }
        return count;
    }
    private int getMaxlte(int[][] matrix, int target){
        int maxVal = Integer.MIN_VALUE;
        int n = matrix.length;
        for(int[] row: matrix)
            for(int ele: row)
                if(ele <= target && maxVal < ele)
                    maxVal = ele;
        return maxVal;
    }
    
}
```

Updated 2019.8.26
- time: 67.32%
- space: 78.38%
```java
class Solution {
    public int kthSmallest(int[][] matrix, int k) {
        int n = matrix.length, l = matrix[0][0], r = matrix[n-1][n-1];
        while(l<r) {
            int mid = l + ((r-l)>>1), count = countLTE(matrix, mid);
            if (count >= k) r = mid;
            else l = mid + 1;
        }
        return l;
    }
    public int countLTE(int[][] matrix, int target) {
        int count = 0;
        for(int[] row: matrix) 
            for(int num: row) {
                if (num > target) break;
                count++;
            }
        return count;
    }
}
```

### 287. Find the Duplicate Number
- [Link](https://leetcode.com/problems/find-the-duplicate-number/)
- Tags: Array, Two Pointers, Binary Search
- Stars: 3

#### Binary Search
<span id="287-binary-search"></span>
Similar to [378. Kth Smallest Element in a Sorted Matrix](#378-binary-search)
```java
class Solution {
    public int findDuplicate(int[] nums) {
        int n = nums.length - 1;
        int l = 1, r = n;
        while(l<r){
            int mid = l + ((r-l)>>1);
            int count = countLTE(nums, mid);
            if(count > mid)
                r = mid;
            else
                l = mid + 1;
        }
        return l;
    }
    private int countLTE(int[] nums, int target){
        int count = 0;
        for(int num: nums)
            if(num <= target)
                count++;
        return count;
    }
}
```


#### slow-fast two pointers
- cheatFlag
<span id="287-two-pointers"></span>
Similar to [142. Linked List Cycle II](#142-two-pointers)
```java
class Solution {
    public int findDuplicate(int[] nums) {
        int slow = 0, fast = 0;
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        while(slow!=fast);
        fast = 0;
        while(slow != fast){
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
}
```

### 142. Linked List Cycle II
- [Link](https://leetcode.com/problems/linked-list-cycle-ii/)
- Tags: Linked List, Two Pointers
- Stars: 3


#### slow-fast two pointers
<span id="142-two-pointers"></span>
Similar to [287. Find the Duplicate Number](#287-two-pointers)
```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        if(head == null) return null;
        ListNode slow = head, fast = head;
        do{
            if(fast.next == null || fast.next.next == null)
                return null;
            slow = slow.next;
            fast = fast.next.next;
        }
        while(slow != fast);
        fast = head;
        while(slow != fast){
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }
}
```

Updated 2019.8.10

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) return null;
        ListNode slow = head, fast = head.next;
        while(fast != null && fast.next != null && slow != fast) {
            slow = slow.next;
            fast = fast.next.next;
        }
        if (fast == null || fast.next == null) return null;
        slow = slow.next;
        fast = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }
}
```

### 328. Odd Even Linked List
- [Link](https://leetcode.com/problems/odd-even-linked-list/)
- Tags: Linked List
- Stars: 1

#### two heads
```java
class Solution {
    public ListNode oddEvenList(ListNode head) {
        if(head == null) return null;
        ListNode curr = head, odd = head, even = head.next;
        while(curr.next != null){
            ListNode temp = curr.next;
            curr.next = temp.next;
            curr = temp;
        }
        curr = odd;
        while(curr.next != null)
            curr = curr.next;
        curr.next = even;
        return odd;
    }
}
```

Updated 2019.8.26
- time: 100%
- space: 100%

```java
class Solution {
    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode odd = head, even = head.next, oddTail = odd, evenTail = even, curr = even.next;
        while(curr != null) {
            oddTail.next = curr;
            curr = curr.next;
            oddTail = oddTail.next;
            if (curr == null) break;
            evenTail.next = curr;
            curr = curr.next;
            evenTail = evenTail.next;
        }
        oddTail.next = even;
        evenTail.next = null;
        return odd;
    }
}
```

### 102. Binary Tree Level Order Traversal
- [Link](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- Tags: Tree, BFS
- Stars: 1


#### BFS
<span id="102-BFS"></span>
Similar to [103. Binary Tree Zigzag Level Order Traversal](#103-BFS)
```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null) return result;
        Queue<TreeNode> qu = new LinkedList<>();
        qu.add(root);
        List<Integer> row = new ArrayList<>();
        int count = 1;
        while(!qu.isEmpty()){
            TreeNode temp = qu.poll();
            if(temp.left != null)
                qu.add(temp.left);
            if(temp.right != null)
                qu.add(temp.right);
            row.add(temp.val);
            count--;
            if(count == 0){
                count = qu.size();
                result.add(row);
                row = new ArrayList<>();
            }
        }
        return result;
    }
}
```

Optimized 2019.9.13 [大雪菜]
- time: 92.48%
- space: 100%
```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ret = new ArrayList<>();
        if (root == null) return ret;
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        while(!q.isEmpty()) {
            int len = q.size();
            List<Integer> level = new ArrayList<>();
            for(int i=0; i<len; i++) {
                TreeNode node = q.poll();
                level.add(node.val);
                if (node.left != null) q.add(node.left);
                if (node.right != null) q.add(node.right);
            }
            ret.add(level);
        }
        return ret;
    }
}
```

### 341. Flatten Nested List Iterator
- [Link](https://leetcode.com/problems/flatten-nested-list-iterator/)
- Tags: Stack, Design
- Stars: 2

#### not real iterator
```java
public class NestedIterator implements Iterator<Integer> {
    Stack<NestedInteger> st;
    public NestedIterator(List<NestedInteger> nestedList) {
        st = new Stack<>();
        for(int i=nestedList.size()-1; i>=0; i--)
            st.push(nestedList.get(i));
    }
    @Override
    public Integer next() {
        if(!hasNext()) return null;
        return st.pop().getInteger();
    }
    @Override
    public boolean hasNext() {
        while(!st.empty()){
            NestedInteger curr = st.peek();
            if(curr.isInteger()){
                return true;
            }
            else{
                st.pop();
                List<NestedInteger> list = curr.getList();
                for(int i=list.size()-1; i>=0; i--)
                    st.push(list.get(i));
            }
        }
        return false;
    }
}
```

#### real iterator
```java
public class NestedIterator implements Iterator<Integer> {
    Stack<ListIterator<NestedInteger>> st;
    public NestedIterator(List<NestedInteger> nestedList) {
        st = new Stack<>();
        st.push(nestedList.listIterator());
    }
    @Override
    public Integer next() {
        if(!hasNext()) return null;
        return st.peek().next().getInteger();
    }
    @Override
    public boolean hasNext() {
        while(!st.empty()){
            if(!st.peek().hasNext()){
                st.pop();
            }
            else{
                NestedInteger curr = st.peek().next();
                if(curr.isInteger()) {
                    st.peek().previous();
                    return true;
                }
                st.push(curr.getList().listIterator());
            }
        }
        return false;
    }
}
```

#### 2019.8.26 stack
- time: 53.03%
- space: 100%

```java
public class NestedIterator implements Iterator<Integer> {
    Stack<NestedInteger> stack = new Stack<>();
    public NestedIterator(List<NestedInteger> nestedList) {
        for(int i=nestedList.size()-1; i>=0; i--) stack.add(nestedList.get(i));
    }
    @Override
    public Integer next() {
        NestedInteger top = stack.pop();
        if (top.isInteger()) return top.getInteger();
        List<NestedInteger> list = top.getList();
        for(int i=list.size()-1; i>=0; i--) stack.add(list.get(i));
        return this.next();
    }
    @Override
    public boolean hasNext() {
        if (stack.isEmpty()) return false;
        NestedInteger top = stack.peek();
        if (top.isInteger()) return true;
        stack.pop();
        List<NestedInteger> list = top.getList();
        for(int i=list.size()-1; i>=0; i--) stack.add(list.get(i));
        return this.hasNext();
    }
}
```

### 48. Rotate Image
- [Link](https://leetcode.com/problems/rotate-image/)
- Tags: Array
- Stars: 1

#### Onion
```java
class Solution {
    public void rotate(int[][] matrix) {
        // rotate(matrix, 0, matrix.length-1);
        for(int i=0, j=matrix.length-1; i<j; i++, j--){
            rotate(matrix, i, j);
        }
    }
    private void rotate(int[][] matrix, int min, int max){
        if(min >= max) return ;
        int len = max-min;
        for(int i=0; i<len; i++){
            int temp = matrix[min][min+i];
            matrix[min][min+i] = matrix[max-i][min];
            matrix[max-i][min] = matrix[max][max-i];
            matrix[max][max-i] = matrix[min+i][max];
            matrix[min+i][max] = temp;
        }
        // rotate(matrix, min+1, max-1);
    }
}
```

#### swap
```java
class Solution {
    public void rotate(int[][] matrix) {
        reverse(matrix);
        int n = matrix.length;
        for(int i=0; i<n; i++)
            for(int j=i+1; j<n; j++)
                diagSwap(matrix, i, j);
    }
    private void reverse(int[][] matrix){
        int l=0, r=matrix.length-1;
        while(l<r)
            swap(matrix, l++, r--);
    }
    private void swap(int[][] matrix, int i, int j){
        int[] temp = matrix[i];
        matrix[i] = matrix[j];
        matrix[j] = temp;
    }
    private void diagSwap(int[][] matrix, int i, int j){
        int temp = matrix[i][j];
        matrix[i][j] = matrix[j][i];
        matrix[j][i] = temp;
    }
}
```

### 62. Unique Paths
- [Link](https://leetcode.com/problems/unique-paths/)
- Tags: Array, Dynamic Programming
- Stars: 2

#### DP
This is a space-optimized DP solution. `dp[i][j] = dp[i-1][j] + dp[i][j-1]`
```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[] dp = new int[m];
        Arrays.fill(dp, 1);
        for(int i=1; i<n; i++)
            for(int j=1; j<m; j++) 
                dp[j] += dp[j-1];
        return dp[dp.length-1];
    }
}
```

#### Math
This is a tricky solution. By observing the DP matrix,
```
1   1   1   1
1   2   3   4
1   3   6   10
1   4   10  20
1   5   15  35
1   6   21  56
```
we can see a Pascal's triangle in the diagonal direction.
Therefore, we have formula `$C_{m+n-2}^{m-1}$` for the final result.

```java
class Solution {
    public int uniquePaths(int m, int n) {
        m--; n--;
        int min = Math.min(m, n);
        if(min == 0) return 1;
        long a = factorial(m+n, m+n-min+1);
        long b = factorial(min, 1);
        return (int)(a/b);
    }
    private long factorial(int max, int min){
        long result = (long)min;
        for(int i=min+1; i<=max; i++)
            result *= i;
        return result;
    }
}
```

### 215. Kth Largest Element in an Array
- [Link](https://leetcode.com/problems/kth-largest-element-in-an-array/)
- Tags: Divide and Conquer, Heap
- Stars: 4

#### Quick Selection
```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        int l = 0, r = nums.length-1;
        while(l<r){
            int idx = partition(nums, l, r);
            if(idx+1 == k) return nums[idx];
            else if(idx+1 > k) r = idx-1; //
            else l = idx+1; // how can this line deal with duplicates??
        }
        return nums[k-1];
    }
    private int partition(int[] nums, int l, int r){
        int i=l, j=r+1;
        while(true){
            while(nums[++i] > nums[l] && i<r);
            while(nums[l] > nums[--j] && j>l);
            if(i>=j) break;
            swap(nums, i, j);
        }
        swap(nums, l, j); //  It's j!! not i!!
        return j;  //  It's j!! not i!!
    }
    private void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

### 49. Group Anagrams
- [Link](https://leetcode.com/problems/group-anagrams/)
- Tags: Hash Table, Sting
- Stars: 1

#### Encoding String into Integer by primes
- time: 11.32%
- space: 98.39%

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<Integer, List<String>> map = new HashMap<>();
        // int[] primes = getPrimes();
        int[] primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101};
        for(String s: strs){
            int num = getNum(primes, s);
            if(!map.containsKey(num))
                map.put(num, new ArrayList<>());
            map.get(num).add(s);
        }
        List<List<String>> result = new ArrayList<>();
        for(List<String> list : map.values()){
            result.add(list);
        }
        return result;
    }
    // private int[] getPrimes(){
    //     int[] primes = new int[26];
    //     int k=0;
    //     int N = 102;
    //     boolean[] isPrime = new boolean[N];
    //     for(int i=2; i<N; i++)
    //         isPrime[i] = true;
    //     for(int i=2; i<N; i++){
    //         if(isPrime[i]){
    //             primes[k++] = i;
    //             if(k==26)  return primes;
    //             for(int j=i+i; j<N; j+=i)
    //                 isPrime[j] = false;
    //         }
    //     }
    //     return null;
    // }
    private int getNum(int[] primes, String s){
        int result = 1;
        for(int i=0; i<s.length(); i++)
            result *= primes[s.charAt(i)-'a'];
        return result;
    }
}
```

#### Hashable Array
- attention: the usage of `map.computeIfAbsent` and its return value. 

When implementing `HashArray.equals()`, the parameter `o` must be of type `Object`!!

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<HashArray, List<String>> map = new HashMap<>();
        for(String s: strs){
            HashArray ha = new HashArray(s);
            map.computeIfAbsent(ha, key->new ArrayList<>()).add(s);
        }
        List<List<String>> result = new ArrayList<>();
        for(List<String> list: map.values())
            result.add(list);
        return result;
    }
}
class HashArray {
    int[] arr = new int[26];
    public HashArray(String s){
        for(char c: s.toCharArray())
            arr[c-'a']++;
    }
    public boolean equals(Object o) {
        return Arrays.equals(this.arr, ((HashArray)o).arr);
    }
    public int hashCode(){
        return Arrays.hashCode(arr);
    }
}
```

#### Encode Anagrams into a String
```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> map = new HashMap<>();
        for(String str: strs) {
            String code = encode(str);
            map.computeIfAbsent(code, key -> new ArrayList<>()).add(str);
        }
        List<List<String>> result = new ArrayList<>();
        for(List<String> list: map.values()) {
            result.add(list);
        }
        return result;
    }
    
    private String encode(String str) {
        StringBuilder sb = new StringBuilder();
        int[] arr = countChars(str);
        for(int i=0; i<arr.length; i++){
            if(arr[i] == 0) continue;
            char c = (char)('a' + i);
            sb.append(c);
            sb.append(arr[i]);
        }
        return sb.toString();
    }
    
    private int[] countChars(String str) {
        int[] result = new int[26];
        for(char c: str.toCharArray()) {
            result[c-'a']++;
        }
        return result;
    }
}
```

#### 2019.9.14 sort [大雪菜]
- time: 80.38%
- space: 94.74%
```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> ret = new ArrayList<>();
        Map<String, List<String>> map = new HashMap<>();
        for(String s: strs) {
            char[] chrs = s.toCharArray();
            Arrays.sort(chrs);
            String key = new String(chrs);
            map.putIfAbsent(key, new ArrayList<>());
            map.get(key).add(s);
        }
        for(List<String> list: map.values()) {
            ret.add(list);
        }
        return ret;
    }
}
```

### 289. Game of Life
- [Link](https://leetcode.com/problems/game-of-life/)
- Tags: Array
- Stars: 1

#### Encoding all possible states
The key idea is to encode all 4 possible transitions:  
    live -> live,  1  
    live -> dead,  -1  
    dead -> live,  2  
    dead -> dead.  0  

In this way, we can calculate `num = (Math.abs(board[i][j])&1)`
```java
class Solution {
    public void gameOfLife(int[][] board) {
        for(int i=0; i<board.length; i++)
            for(int j=0; j<board[0].length; j++){
                int num = countCells(board, i, j);
                if(board[i][j] == 1)
                    if(num<2 || num>3) 
                        board[i][j] = -1;
                else
                    if(num == 3)
                        board[i][j] = 2;
            }
        for(int i=0; i<board.length; i++)
            for(int j=0; j<board[0].length; j++){
                if(board[i][j] == -1) board[i][j] = 0;
                if(board[i][j] == 2) board[i][j] = 1;
            }
    }
    private int countCells(int[][] board,int x,int y){
        int count = 0;
        for(int i=x-1; i<=x+1; i++)
            for(int j=y-1; j<=y+1; j++){
                if(i==x && j==y) continue;
                if(i>=0 && j>=0 && i<board.length && j<board[0].length){
                    count += (Math.abs(board[i][j])&1);
                }
            }
        return count;
    }
}
```

#### Encoding by the right most two bits
- time: 100%
- space: 100%

```java
class Solution {
    public void gameOfLife(int[][] board) {
        if (board.length == 0 || board[0].length == 0) return ;
        for(int i=0; i<board.length; i++)
            for(int j=0; j<board[0].length; j++) {
                int numNb = countLiveNeighbors(board, i, j);
                if((board[i][j]&1) == 0) {
                    // dead cell
                    if(numNb == 3) board[i][j] |= 2;
                    else board[i][j] &= ~2;
                }
                else {
                    // live cell
                    if(numNb < 2 || numNb > 3) board[i][j] &= ~2;
                    else board[i][j] |= 2;
                }
            }
        for(int i=0; i<board.length; i++)
            for(int j=0; j<board[0].length; j++) {
                board[i][j] &= 2;
                board[i][j] >>= 1;
            }
    }
    
    private int countLiveNeighbors(int[][] board, int x, int y) {
        int count = 0;
        for(int i=x-1; i<=x+1; i++) {
            if (i<0 || i>board.length-1) continue;
            for(int j=y-1; j<=y+1; j++) {
                if (j<0 || j>board[0].length - 1) continue;
                if(i==x && j==y) continue;
                if((board[i][j] & 1) == 1) count++;
            }
        }
        return count;
    }
}
```

### 11. Container With Most Water
- [Link](https://leetcode.com/problems/container-with-most-water/)
- Tags: Array, Two Pointers
- Stars: 2

#### Not My Solution
[Here is an awesome explanation!](https://leetcode.com/problems/container-with-most-water/discuss/6099/Yet-another-way-to-see-what-happens-in-the-O(n)-algorithm)
```java
class Solution {
    public int maxArea(int[] height) {
        int l = 0, r = height.length-1;
        int area = Integer.MIN_VALUE;
        while(l<r){
            area = Math.max(area, Math.min(height[l], height[r])*(r-l));
            if(height[l] < height[r]) l++;
            else r--;
        }
        return area;
    }
}
```

#### two pointers
- time: 94.18%
- space: 97.90%

```java
class Solution {
    public int maxArea(int[] height) {
        if(height.length < 2) return 0;
        int maxVolume = 0;
        int i=0, j=height.length - 1;
        while(i < j) {
            maxVolume = Math.max(maxVolume, (j-i) * Math.min(height[i], height[j]));
            if(height[i] > height[j]) j--;
            else if (height[i] < height[j]) i++;
            else {
                i++; j--;
            }
        }
        return maxVolume;
    }
}
```

### 380. Insert Delete GetRandom O(1)
- [Link](https://leetcode.com/problems/insert-delete-getrandom-o1/)
- Tags: Array, Hash Table, Design
- Stars: 3

#### Tomb
```java
class RandomizedSet {
    HashMap<Integer, Integer> map;
    List<Integer> list;
    Random rand;
    /** Initialize your data structure here. */
    public RandomizedSet() {
        rand = new Random();
        map = new HashMap<>();
        list = new ArrayList<>();
    }
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if(map.containsKey(val)) 
            return false;
        map.put(val, list.size());
        list.add(val);
        return true;
    }
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        if(!map.containsKey(val)) 
            return false;
        int idx = map.get(val);
        list.set(idx, null);
        map.remove(val);
        if(map.size() <= list.size()/2){
            map = new HashMap<>();
            List<Integer> newList = new ArrayList<>();
            for(int i=0; i<list.size(); i++){
                Integer num = list.get(i);
                if(num != null){
                    map.put(num, newList.size());
                    newList.add(num);
                }
            }
            list = newList;
        }
        return true;
    }
    /** Get a random element from the set. */
    public int getRandom() {
        while(true){
            int idx = rand.nextInt(list.size());
            Integer num = list.get(idx);
            if(num!=null) return num;
        }
    }
}
```

#### 2019.8.30 swap to the last one
- time: 75.87%
- space: 76%
```java
class RandomizedSet {
    Map<Integer, Integer> map = new HashMap<>();
    List<Integer> list = new ArrayList<>();
    Random rand = new Random();
    public RandomizedSet() {}
    public boolean insert(int val) {
        if (map.containsKey(val)) return false;
        map.put(val, list.size());
        list.add(val);
        return true;
    }
    public boolean remove(int val) {
        if (!map.containsKey(val)) return false;
        int i = map.get(val), j = list.size()-1;
        map.remove(val);
        if (i != j) {
            map.put(list.get(j), i);
            list.set(i, list.get(j));
        }
        list.remove(j);
        return true;
    }
    public int getRandom() {
        return list.get(rand.nextInt(list.size()));
    }
}
```

### 36. Valid Sudoku
- [Link](https://leetcode.com/problems/valid-sudoku/)
- Tags: Hash Table
- Stars: 3

#### Encoding by self-defined Class
type == 0 --> row  
type == 1 --> col  
type == 2 --> 3x3 block  
```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        HashSet<Tuple> st = new HashSet<>(3*81);
        for(int i=0; i<9; i++)
            for(int j=0; j<9; j++)
                if(board[i][j] != '.'){
                    char c = board[i][j];
                    if(!st.add(new Tuple(0, i, c)) ||
                       !st.add(new Tuple(1, j, c)) ||
                       !st.add(new Tuple(2, i/3, j/3 ,c)))
                        return false;
                }
        return true;
    }
}
class Tuple {
    int type, i, j;
    char c;
    public Tuple(int t, int k, char ch){
        this(t, k, k, ch);
    }
    public Tuple(int t, int x, int y, char ch){
        c = ch;
        type = t;
        i = x;
        j = y;
    }
    public boolean equals(Object o){
        Tuple obj = (Tuple) o;
        if(this.type != obj.type || this.c!=obj.c) return false;
        if(type == 0) return this.i==obj.i;
        if(type == 1) return this.j==obj.j;
        else return this.i==obj.i && this.j==obj.j;
    }
    public int hashCode(){
        return (Integer.hashCode(type) +
            Integer.hashCode(i) +
            Integer.hashCode(j) +
            Integer.hashCode(c));
    }
}
```

#### Encoding by native String
- time: 49.15%
- space: 89.64%

"r%d%c" --> row  
"c%d%c" --> col  
"b%d%d%c" --> 3x3 block  

```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        HashSet<String> st = new HashSet<>();
        for(int i=0; i<9; i++)
            for(int j=0; j<9; j++)
                if(board[i][j] != '.'){
                    char ch = board[i][j];
                    if(!st.add("r"+i+ch) || 
                       !st.add("c"+j+ch) || 
                       !st.add("b"+i/3+j/3+ch))
                        return false;
                }
        return true;
    }
}
```

### 75. Sort Colors
- [Link](https://leetcode.com/problems/sort-colors/)
- Tags: Array, Two Pointers, Sort
- Stars: 1

#### one pass solution
two pointers
```java
class Solution {
    public void sortColors(int[] nums) {
        int curr=0, i=0, j=nums.length-1;
        while(curr<=j){
            if(nums[curr] == 2) swap(nums, curr, j--);
            else if(nums[curr] == 1) curr++;
            else swap(nums, curr++, i++);
        }
    }
    private void swap(int[] nums, int i, int j){
        if(nums[i] != nums[j]){
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
        }
    }
}
```

#### two pass solution
Count
```java
class Solution {
    public void sortColors(int[] nums) {
        int zeros=0, ones=0, twos=0;
        for(int num: nums){
            if(num==0) zeros++;
            else if(num==1) ones++;
            else twos++;
        }
        int i=0;
        while(zeros-->0) nums[i++]=0;
        while(ones-->0) nums[i++]=1;
        while(twos-->0) nums[i++]=2;
    }
}
```

### 162. Find Peak Element
- [Link](https://leetcode.com/problems/find-peak-element/)
- Tags: Array, Binary Search
- Stars: 1

#### Binary Search differences of adjacent elements
```java
class Solution {
    public int findPeakElement(int[] nums) {
        if(nums.length == 0) return 0;
        if(goesUp(nums, nums.length-1)) return nums.length-1;
        int l=0, r=nums.length-1;
        while(true){
            int mid = l + ((r-l)>>1);
            if(goesUp(nums, mid)) l = mid;
            else r = mid;
            if(l + 1 >= r) break;
        }
        return l;
    }
    private boolean goesUp(int[] nums, int idx){
        if(idx == 0) return true;
        return nums[idx]>nums[idx-1];
    }
}
```

#### binary search too
```java
class Solution {
    public int findPeakElement(int[] nums) {
        if(nums.length == 1) return 0;
        // if(nums[0]>nums[1]) return 0;
        if(nums[nums.length-1] > nums[nums.length-2]) return nums.length-1;
        int l=0, r=nums.length-1;
        while(l<r){
            int mid = l+((r-l)>>1);
            // if(nums[mid]>nums[mid-1]) l = mid;
            // else r = mid-1;
            if(nums[mid]>nums[mid+1]) r = mid;
            else l = mid+1;
        }
        return l;
    }
}
```

#### simple binary search
- time: 100%
- space: 99.09%
- interviewLevel

```java
class Solution {
    public int findPeakElement(int[] nums) {
        int l = 0, r = nums.length - 1;
        while(l<r) {
            int mid = l + ((r-l)>>1);
            if(nums[mid] < nums[mid+1]) l = mid + 1;
            else r = mid;
        }
        return l;
    }
}
```

### 103. Binary Tree Zigzag Level Order Traversal
- [Link](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)
- Tags: Stack, Tree, BFS
- Stars: 1

#### BFS
<span id="103-BFS"></span>
Similar to [102. Binary Tree Level Order Traversal](#102-BFS)
```java
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null) return result;
        Queue<TreeNode> qu = new LinkedList<>();
        qu.add(root);
        int count = 1;
        List<Integer> list = new ArrayList<>();
        while(!qu.isEmpty()){
            TreeNode temp = qu.poll();
            count--;
            if(temp.left != null) qu.add(temp.left);
            if(temp.right != null) qu.add(temp.right);
            list.add(temp.val);
            if(count == 0){
                count = qu.size();
                result.add(list);
                list = new ArrayList<>();
            }
        }
        for(int i=1; i<result.size(); i+=2)
            reverse(result.get(i));
        return result;
    }
    private void reverse(List<Integer> list){
        int i=0, j=list.size()-1;
        while(i<j)
            Collections.swap(list, i++, j--);
    }
}
```

### 279. Perfect Squares
- [Link](https://leetcode.com/problems/perfect-squares/)
- Tags: Math, Dynamic Programming, BFS
- Stars: 2

#### static DP 298ms
```java
class Solution {
    int maxSq;
    HashMap<Integer,Integer> map;
    public int numSquares(int n) {
        maxSq = getSquares(n);
        map = new HashMap<>();
        return getNumSquares(n);
    }
    private int getNumSquares(int n){
        if(n<=0) return 0;
        if(map.containsKey(n)) return map.get(n);
        int min = Integer.MAX_VALUE;
        for(int i=1; i<=maxSq; i++){
            int sq = i*i;
            if(sq>n) break;
            min = Math.min(min, 1 + getNumSquares(n-sq));
        }
        map.put(n, min);
        return min;
    }
    private int getSquares(int n){
        int l=1, r=46340;
        while(l<r){
            int mid = l + ((r-l)>>1);
            int sq = mid*mid;
            if(sq == n) return mid;
            else if(sq > n) r = mid;
            else l = mid+1;
        }
        return l;
    }
}
```

#### DP 25ms
<span id="279-DP" />

Similar to [322. Coin Change](#322-DP)
```java
class Solution {
    public int numSquares(int n) {
        int[] dp = new int[n+1];
        for(int i=2; i<n+1; i++) dp[i] = Integer.MAX_VALUE;
        dp[1] = 1;
        for(int i=2; i<=n; i++){
            for(int j=1; j*j<=i; j++){
                dp[i] = Math.min(dp[i], 1+dp[i-j*j]);
            }
        }
        return dp[n];
    }
}
```

#### BFS 87ms
```java
class Solution {
    public int numSquares(int n) {
        Queue<Integer> qu = new LinkedList<>();
        qu.add(n);
        int count = 1, level = 0;
        while(!qu.isEmpty()){
            int curr = qu.poll();
            count--;
            if(curr == 0) return level;
            for(int i=1; i*i<=curr; i++){
                int val = curr-i*i;
                if(val == 0) return level+1;
                qu.add(val);
            }
            if(count == 0){
                count = qu.size();
                level++;
            }
        }
        return 0;
    }
}
```

### 322. Coin Change
- [Link](https://leetcode.com/problems/coin-change/)
- Tags: Dynamic Programming
- Stars: 2

#### DP
<span id="322-DP" />

Similar to [279. Perfect Squares](#279-DP)
```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        if(amount == 0) return 0;
        if(coins.length == 0) return -1;
        int[] dp = new int[amount+1];
        Arrays.fill(dp, -1);
        Arrays.sort(coins);
        dp[0] = 0;
        for(int i=0; i<=amount; i++){
            for(int coin : coins){
                if(coin > i) break;
                if(dp[i-coin] == -1) continue;
                dp[i] = dp[i] == -1 ? dp[i-coin]+1 : Math.min(dp[i], dp[i-coin]+1);
            }
        }
        return dp[amount];
    }
}
```

#### still DP, but init with a self-defined maxVal instead of -1
```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        if(amount == 0) return 0;
        if(coins.length == 0) return -1;
        int[] dp = new int[amount+1];
        int maxVal = amount+1;
        Arrays.fill(dp, maxVal);
        Arrays.sort(coins);
        dp[0] = 0;
        for(int i=0; i<=amount; i++){
            for(int coin : coins){
                if(coin > i) break;
                dp[i] = Math.min(dp[i], dp[i-coin]+1);
            }
        }
        return dp[amount] == maxVal ? -1 : dp[amount];
    }
}
```

### 240. Search a 2D Matrix II
- [Link](https://leetcode.com/problems/search-a-2d-matrix-ii/)
- Tags: Binary Search, Divide and Conquer
- Stars: 1

#### BST
```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        if(matrix.length == 0 || matrix[0].length==0) return false;
        int m = matrix.length, n = matrix[0].length;
        int i=0, j=n-1;
        while(i<m && j>=0){
            if(matrix[i][j] == target) return true;
            else if(matrix[i][j] < target) i++;
            else j--;
        }
        return false;
    }
}
```

### 300. Longest Increasing Subsequence
- [Link](https://leetcode.com/problems/longest-increasing-subsequence/)
- Tags: Binary Search, Dynamic Programming
- Stars: 3

#### DP O(n^2)
```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        if(nums.length == 0) return 0;
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        for(int i=1; i<nums.length; i++){
            for(int j=0; j<i; j++){
                if(nums[j] < nums[i]){
                    dp[i] = Math.max(dp[i], 1+dp[j]);
                }
            }
        }
        int result = 0;
        for(int i=0; i<dp.length; i++){
            if(result < dp[i]) result = dp[i];
        }
        return result;
    }
}
```

#### binary search O(nlogn)
<span id="300-binary-search"></span>
`tails[i]` = the min value of the last elements of all subsequences with length of i+1. 

Similar to [334. Increasing Triplet Subsequence](#334-binary-search)
```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] tails = new int[nums.length];
        int maxLen = 0;
        for(int num: nums){
            int idx = Arrays.binarySearch(tails, 0, maxLen, num);
            if(idx<0) idx = -(idx+1);
            tails[idx] = num;
            if(maxLen == idx) maxLen++;
        }
        return maxLen;
    }
}
```

### 334. Increasing Triplet Subsequence
- [Link](https://leetcode.com/problems/increasing-triplet-subsequence/)
- Tags: 
- Stars: 2

#### binary search 
<span id="334-binary-search"></span>
Similar to [300. Longest Increasing Subsequence](#300-binary-search)
```java
class Solution {
    public boolean increasingTriplet(int[] nums) {
        int[] tails = new int[3];
        int maxLen = 0;
        for(int num: nums){
            int idx = getIdx(tails, maxLen, num);
            tails[idx] = num;
            if(idx == maxLen) maxLen++;
            if(maxLen == 3) return true;
        }
        return false;
    }
    private int getIdx(int[] tails, int maxLen, int num){
        for(int i=0; i<maxLen; i++){
            if(num <= tails[i]) return i;
        }
        return maxLen;
    }
}
```

#### another binary search
```java
class Solution {
    public boolean increasingTriplet(int[] nums) {
        int a=Integer.MAX_VALUE, b=Integer.MAX_VALUE;
        for(int num: nums){
            if(num<=a) a = num;
            else if(num<=b) b = num;
            else return true;
        }
        return false;
    }
}
```

### 105. Construct Binary Tree from Preorder and Inorder Traversal
- [Link](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
- Tags: Array, Tree, DFS
- Stars: 3

#### DFS
```java
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return buildTree(preorder, 0, inorder, 0, inorder.length);
    }
    private TreeNode buildTree(int[] preorder, int m, 
                               int[] inorder, int n, int len){
        if(len == 0) return null;
        TreeNode root = new TreeNode(preorder[m]);
        int mid = indexOf(inorder, n, len, root.val);
        root.left = buildTree(preorder, m+1, inorder, n, mid-n);
        root.right = buildTree(preorder, m+(mid-n+1), inorder, mid+1, len-(mid-n+1));
        return root;
    }
    private int indexOf(int[] nums, int start, int len, int target){
        for(int i=start; i<start+len; i++)
            if(nums[i]==target)
                return i;
        return -1;
    }
}
```

#### 2019.9.13 DFS with memoization by [大雪菜]
- time: 97.50%
- space: 17.76%
- reviewFlag
```java
class Solution {
    Map<Integer, Integer> pos = new HashMap<>();
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        for(int i=0; i<inorder.length; i++) {
            pos.put(inorder[i], i);
        }
        return buildTree(preorder, inorder, 0, 0, preorder.length);
    }
    public TreeNode buildTree(int[] preorder, int[] inorder, int pl, int il, int len) {
        if (len == 0) return null;
        TreeNode root = new TreeNode(preorder[pl]);
        int iroot = pos.get(preorder[pl]), lLen = iroot-il, rLen = len-1-lLen;
        root.left = buildTree(preorder, inorder, pl+1, il, lLen);
        root.right = buildTree(preorder, inorder, pl+1+lLen, iroot+1, rLen);
        return root;
    }
}
```

### 73. Set Matrix Zeroes
- [Link](https://leetcode.com/problems/set-matrix-zeroes/)
- Tags: Array
- Stars: 3

#### O(mn) time and O(1) space
```java
class Solution {
    public void setZeroes(int[][] matrix) {
        if(matrix.length==0 || matrix[0].length==0) return ;
        // check whether the first row has zero;
        boolean firstRowZero = false;
        for(int j=0; j<matrix[0].length; j++)
            if(matrix[0][j]==0){
                firstRowZero = true;
                break;
            }
        // Set all rows that have zero to zeros, and mark the zero column in the first row
        for(int i=1; i<matrix.length; i++){
            boolean rowZero = false;
            for(int j=0; j<matrix[0].length; j++){
                if(matrix[i][j]==0){
                    rowZero = true;
                    matrix[0][j] = 0;
                }
            }
            if(rowZero) Arrays.fill(matrix[i], 0);
        }
        // Set all the zero columns to zeros
        for(int j=0; j<matrix[0].length; j++)
            if(matrix[0][j]==0)
                for(int i=1; i<matrix.length; i++)
                    matrix[i][j] = 0;
        // deal with the first row
        if(firstRowZero) Arrays.fill(matrix[0], 0);
    }
}
```

### 395. Longest Substring with At Least K Repeating Characters
- [Link](https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/)
- Tags: 
- Stars: 5
- reviewFlag

#### two pointers
```java
class Solution {
    public int longestSubstring(String s, int k) {
        return longestSubstring(s, 0, s.length()-1, k);
    }
    public int longestSubstring(String s, int l, int r, int k){
        if(r-l+1 < k) return 0;
        int[] stat = new int[26];
        for(int i=l; i<=r; i++) stat[s.charAt(i)-'a']++;
        int charIdx = getCharIdx(stat, k);
        while(l<=r && charIdx != -1){
            if(s.charAt(r)-'a' == charIdx){
                stat[charIdx]--;
                r--;
            }
            else if (s.charAt(l)-'a' == charIdx){
                stat[charIdx]--;
                l++;
            }
            else {
                for(int i=l; i<=r; i++)
                    if(s.charAt(i)-'a' == charIdx){
                        return Math.max(longestSubstring(s, l, i-1, k), 
                                        longestSubstring(s, i+1, r, k));
                    }
            }
            if(stat[charIdx] == 0) charIdx = getCharIdx(stat, k);
        }
        if(l>r) return 0;
        return r-l+1;
    }
    private int getCharIdx(int[] stat, int k){
        for(int j=0; j<26; j++)
            if(stat[j]>0 && stat[j]<k)
                return j;
        return -1;
    }
}
```

#### 2019.9.6 DFS
- time: 100%
- space: 100%
- attention: We use all the infrequent elements as splits
- cheatFlag
```java
class Solution {
    public int longestSubstring(String s, int k) {
        return longestSubstring(s, k, 0, s.length()-1);
    }
    public int longestSubstring(String s, int k, int l, int r) {
        if (l > r) return 0;
        int[] stat = new int[26];
        for(int i=l; i<=r; i++) {
            stat[s.charAt(i)-'a']++;
        }
        boolean flag = true;
        for(int count: stat)
            if (count > 0 && count < k) flag = false;
        if (flag) return r-l+1;
        
        int ret = 0, start = l;
        for(int i=l; i<=r; i++) {
            if(stat[s.charAt(i)-'a'] < k) {
                ret = Math.max(ret, longestSubstring(s, k, start, i-1));
                start = i+1;
            }
        }
        ret = Math.max(ret, longestSubstring(s, k, start, r));
        return ret;
    }
}
```

### 207. Course Schedule
- [Link](https://leetcode.com/problems/course-schedule/)
- Tags: BFS, DFS, Graph, Topological Sort
- Stars: 3
- reviewFlag

#### Topological Sort
<span id="207-topo-sort"></span>
Similar to [210. Course Schedule II](#210-topo-sort)
```java
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] degrees = new int[numCourses];
        HashMap<Integer, List<Integer>> graph = new HashMap<>();
        for(int[] edge: prerequisites)
            graph.computeIfAbsent(edge[1], key->new ArrayList<>()).add(edge[0]);
        for(int[] edge: prerequisites) 
            degrees[edge[0]]++;
        Queue<Integer> qu = new LinkedList<>();
        for(int i=0; i<numCourses; i++)
            if(degrees[i] == 0) qu.add(i);
        while(!qu.isEmpty()){
            int course = qu.poll();
            if(graph.containsKey(course))
                for(int subCourse: graph.get(course))
                    if(--degrees[subCourse] == 0) qu.add(subCourse);
        }
        for(int degree: degrees)
            if(degree>0) return false;
        return true;
    }
}
```

Optimized 2019.9.6
- time: 89.87%
- space: 97.69%
- interviewLevel
```java
class Solution {
    public boolean canFinish(int n, int[][] edges) {
        List<Integer>[] graph = new List[n];
        for(int i=0; i<n; i++) {
            graph[i] = new ArrayList<>();
        }
        int[] degrees = new int[n];
        for(int[] e: edges) {
            graph[e[1]].add(e[0]);
            degrees[e[0]]++;
        }
        Queue<Integer> qu = new LinkedList<>();
        for(int i=0; i<n; i++) {
            if (degrees[i] == 0) qu.add(i);
        }
        int count = 0;
        while(!qu.isEmpty()) {
            int leaf = qu.poll();
            count++;
            for(int inner: graph[leaf]) {
                degrees[inner]--;
                if (degrees[inner] == 0) qu.add(inner);
            }
        }
        return count == n;
    }
}
```

#### DFS
1. Use `visiting` array to trace along the DFS path. Just like what backtracking usually does, you set visiting[i] to true before DFS, and reset it to false after DFS. 
2. The `visited` array is just used to remember the DFS result of each node to avoid recomputation. 
```java
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        HashMap<Integer, List<Integer>> graph = new HashMap<>();
        for(int[] edge: prerequisites)
            graph.computeIfAbsent(edge[1], key->new ArrayList<>()).add(edge[0]);
        for(int i=0; i<numCourses; i++)
            if(!DFS(graph, new boolean[numCourses], new boolean[numCourses], i)) return false;
        return true;
        
    }
    private boolean DFS(HashMap<Integer, List<Integer>> graph, boolean[] visited, boolean[] visiting, int course){
        if(visited[course]) return true;
        
        if(visiting[course]) return false;
        visiting[course] = true;
        if(graph.containsKey(course))
            for(Integer subCourse: graph.get(course)){
                if(!DFS(graph, visited, visiting, subCourse)) return false;
            }
        visiting[course] = false;
        
        visited[course] = true;
        return true;
    }
}
```

Optimized 2019.9.6
- time: 99.87%
- space: 96.92%
- attention: When initiating dfs, those whose degree are not initially zero may turn to zero. Thus, we need a `visited` to only init dfs on the initially-zero nodes
```java
class Solution {
    public boolean canFinish(int n, int[][] edges) {
        List<Integer>[] graph = new List[n];
        for(int i=0; i<n; i++) {
            graph[i] = new ArrayList<>();
        }
        int[] degrees = new int[n];
        boolean[] visited = new boolean[n];
        for(int[] e: edges) {
            graph[e[1]].add(e[0]);
            degrees[e[0]]++;
        }
        for(int i=0; i<n; i++) {
            if (degrees[i] == 0 && !visited[i]) {
                dfs(graph, degrees, visited, i);
            }
        }
        for(int d: degrees)
            if (d > 0) return false;
        return true;
    }
    private void dfs(List<Integer>[] graph, int[] degrees, boolean[] visited, int leaf) {
        visited[leaf] = true;
        for(int inner: graph[leaf]) {
            degrees[inner]--;
            if (degrees[inner] == 0) dfs(graph, degrees, visited, inner);
        }
    }
}
```

### 210. Course Schedule II
- [Link](https://leetcode.com/problems/course-schedule-ii/)
- Tags: BFS, DFS, Graph, Topological Sort
- Stars: 1

#### Topological Sort
<span id="210-topo-sort"></span>
Similar to [207. Course Schedule](#210-topo-sort)
```java
class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        HashMap<Integer, List<Integer>> graph = new HashMap<>();
        int[] degrees = new int[numCourses];
        for(int[] edge: prerequisites){
            graph.computeIfAbsent(edge[1], key-> new ArrayList<>()).add(edge[0]);
            degrees[edge[0]]++;
        }
        Queue<Integer> qu = new LinkedList<>();
        for(int i=0; i<numCourses; i++) 
            if(degrees[i] == 0) qu.add(i);
        int[] result = new int[numCourses];
        int i=0;
        while(!qu.isEmpty()){
            int course = qu.poll();
            if(graph.containsKey(course))
                for(int subcourse : graph.get(course)) 
                    if(--degrees[subcourse] == 0) qu.add(subcourse);
            result[i++] = course;
        }
        if(i != numCourses) return new int[0];
        return result;
    }
    
}
```

Optimized 2019.9.6
- time: 8 ms
- space: 45.4 MB
```java
class Solution {
    public int[] findOrder(int n, int[][] edges) {
        List<Integer>[] graph = new List[n];
        for(int i=0; i<n; i++) {
            graph[i] = new ArrayList<>();
        }
        int[] degrees = new int[n];
        for(int[] e: edges) {
            graph[e[1]].add(e[0]);
            degrees[e[0]]++;
        }
        List<Integer> leaves = new LinkedList<>();
        int p=0;
        for(int i=0; i<n; i++) {
            if (degrees[i] == 0) leaves.add(i);
        }
        while(p < leaves.size()) {
            int leaf = leaves.get(p++);
            for(int inner: graph[leaf]) {
                degrees[inner]--;
                if (degrees[inner] == 0) leaves.add(inner);
            }
        }
        if (leaves.size() != n) return new int[0];
        int[] ret = new int[n];
        for(int i=0; i<n; i++)
            ret[i] = leaves.get(i);
        return ret;
    }
}
```

### 34. Find First and Last Position of Element in Sorted Array
- [Link](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
- Tags: Array, Binary Search
- Stars: 3

#### Binary Search Lower Bound
This question needs careful consideration of the boundaries!

1. To ensure the output of lowerBound is still in the range of [0, nums.length-1], `int b = lowerBound(nums, a, nums.length-1, target+1);` cannot be `int b = lowerBound(nums, a+1, nums.length-1, target+1);` in order to adapt to the case where `a==b`
2. We still need `arr[1] = nums[b] == target ? b : b-1;` because b might be the last element and thus nums[b] might be smaller than `target+1`
```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int[] arr = {-1,-1};
        if(nums.length == 0) return arr;
        int a = lowerBound(nums, 0, nums.length-1, target);
        if(nums[a]!=target) return arr;
        int b = lowerBound(nums, a, nums.length-1, target+1);
        arr[0] = a; 
        arr[1] = nums[b] == target ? b : b-1;
        return arr;
    }
    private int lowerBound(int[] nums, int i, int j, int target){
        int l=i, r=j;
        while(l<r){
            int mid = l + ((r-l)>>1);
            if(nums[mid] == target) r = mid;
            else if(nums[mid] > target) r = mid-1;
            else l = mid+1;
        }
        return l;
    }
}
```

Updated 2019.7.31

Here we define `lowerBound` function to find the minimal element in `nums` that is greater than or equal to the target.

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int[] result = new int[2];
        result[0] = result[1] = -1;
        // Binary search boundary check
        if (nums.length == 0 || target < nums[0] || target > nums[nums.length - 1]) return result;
        int l = 0, r = nums.length - 1;
        int start = lowerBound(nums, target);
        if (nums[start] != target) return result;
        result[0] = start;
        // Binary search boundary check
        if (target + 1 > nums[nums.length - 1]) {
            result[1] = nums.length - 1;
            return result;
        }
        int end = lowerBound(nums, target + 1);
        result[1] = end - 1;
        return result;
    }
    public int lowerBound(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int mid = l + ((r-l)>>1);
            if (nums[mid] < target) l = mid + 1;
            else if (nums[mid] >= target) r = mid;
        }
        return l;
    }
}
```

Updated 2019.9.12
- time: 100%
- space: 100%
- attention: `r` is initiated with `nums.length` instead of `nums.length-1`. `if (target == Integer.MAX_VALUE) return new int[]{i, nums.length-1};` is used to avoid overflow of `target+1`.
```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        if (nums.length == 0 || target > nums[nums.length-1]) return new int[]{-1, -1};
        int i = binarySearch(nums, target);
        if (nums[i] != target) return new int[]{-1, -1};
        if (target == Integer.MAX_VALUE) return new int[]{i, nums.length-1};
        int j = binarySearch(nums, target+1);
        return new int[]{i, j-1};
    }
    public int binarySearch(int[] nums, int target) {
        int l = 0, r = nums.length;
        while(l<r) {
            int mid = l+r >> 1;
            if (nums[mid] >= target) r = mid;
            else l = mid + 1;
        }
        return l;
    }
}
```

### 236. Lowest Common Ancestor of a Binary Tree
- [Link](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
- Tags: Tree
- Stars: 1

#### DFS
```java
class Solution {
    TreeNode node;
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        helper(root, p, q);
        return node;
    }
    private int helper(TreeNode root, TreeNode p, TreeNode q){
        if(root == null) return 0;
        int a = helper(root.left, p, q);
        if(a == 2) return 2;
        int b = helper(root.right, p, q);
        if(b == 2) return 2;
        int result = a + b;
        if(root == p || root == q) result++;
        if(result == 2) node = root;
        return result;
    }
}
```

#### DFS without helper counting function
```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null) return null;
        if(root == p || root == q) return root;
        TreeNode l = lowestCommonAncestor(root.left, p, q);
        TreeNode r = lowestCommonAncestor(root.right, p, q);
        if(l == null) return r;
        if(r == null) return l;
        return root;
    }
}
```

### 235. Lowest Common Ancestor of a Binary Search Tree
- [Link](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)
- Tags: Tree
- Stars: 1

#### BST, beats 100%
Please refer to **236**.
```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null) return null;
        if(p.val > q.val) return lowestCommonAncestor(root, q, p);
        if(root.val < p.val) return lowestCommonAncestor(root.right, p, q);
        if(root.val > q.val) return lowestCommonAncestor(root.left, p, q);
        return root;
    }
}
```

### 139. Word Break
- [Link](https://leetcode.com/problems/word-break/)
- Tags: Dynamic Programming
- Stars: 3

#### DP
```java
class Solution {
    HashMap<Character, List<String>> map = new HashMap<>();
    public boolean wordBreak(String s, List<String> wordDict) {
        for(String str: wordDict)
            map.computeIfAbsent(str.charAt(str.length()-1), key->new ArrayList<>()).add(str);
        boolean[] dp = new boolean[s.length()];
        for(int i=0; i<s.length(); i++){
            char c = s.charAt(i);
            String subs = s.substring(0, i+1);
            if(map.containsKey(c))
               for(String word: map.get(c)){
                   if(subs.endsWith(word) && (i<word.length() || dp[i-word.length()])) dp[i] = true;
               }
        }
        return dp[s.length()-1];
    }
}
```

#### DP with memorization
- time: 100%
- space: 99.93%
- interviewLevel

```java
class Solution {
    boolean[] marked;
    public boolean wordBreak(String s, List<String> wordDict) {
        marked = new boolean[s.length()];
        return wordBreak(s, wordDict, 0);
    }
    public boolean wordBreak(String s, List<String> wordDict, int start) {
        if (start >= s.length()) return true;
        if (marked[start]) return false;
        for (String word: wordDict) {
            if (s.startsWith(word, start)) {
                if (wordBreak(s, wordDict, start + word.length())) return true;
                marked[start] = true;
            }
        }
        return false;
    }
}
```


### 19. Remove Nth Node From End of List
- [Link](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)
- Tags: Linked List, Two Pointers
- Stars: 3

#### two pass solution
- time: 100%
- space: 100%

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null) return null;
        ListNode curr = head;
        int count = 0;
        while(curr != null) {
            count++;
            curr = curr.next;
        }
        if (count == n) return head.next;
        curr = moveForward(head, count - n - 1);
        curr.next = curr.next.next;
        return head;
    }
    public ListNode moveForward(ListNode head, int n) {
        for(int i=0; i<n; i++)
            head = head.next;
        return head;
    }
}
```

#### one pass solution
- time: 100%
- space: 100%
- interviewLevel

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if(head == null) return null;
        ListNode fast = head, slow = head;
        for(int i=0; i<n; i++) fast = fast.next;
        if(fast == null) return head.next;
        while(fast.next != null){
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return head;
    }
}
```

### 56. Merge Intervals
- [Link](https://leetcode.com/problems/merge-intervals/)
- Tags: Array, Sort
- Stars: 2

#### sort
- time: 52.24%
- space: 73.13%

The way of writting a sort function can be simplified to `intervals.sort((o1, o2)->o1.start-o2.start);`.

Take a look at [435. Non-overlapping Intervals](#435. Non-overlapping Intervals) 
```java
class Solution {
    public List<Interval> merge(List<Interval> intervals) {
        if(intervals.size() == 0) return intervals;
        Collections.sort(intervals, new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2){
                return o1.start - o2.start;
            }
        });
        int start = intervals.get(0).start, end = intervals.get(0).end;
        int i=1;
        List<Interval> result = new ArrayList<>();
        for(; i<intervals.size(); i++){
            if(intervals.get(i).start > end){
                result.add(new Interval(start, end));
                start = intervals.get(i).start; end = intervals.get(i).end;
            }
            else {
                end = Math.max(end, intervals.get(i).end);
            }
        }
        result.add(new Interval(start, end));
        return result;
    }
}
```

Updated 2019.7.30
```java
class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals.length < 2) return intervals;
        List<int[]> list = new ArrayList<>();
        Arrays.sort(intervals, (int[] r1, int[] r2) -> r1[0] - r2[0]);
        int start = intervals[0][0], end = intervals[0][1];
        for(int i=1; i<intervals.length; i++) {
            int[] range = intervals[i];
            if (range[0] <= end) {
                end = Math.max(end, range[1]);
            }
            else {
                insert(list, start, end);
                start = range[0];
                end = range[1];
            }
        }
        insert(list, start, end);
        int[][] result = new int[list.size()][2];
        for(int i=0; i<list.size(); i++) {
            int[] range = list.get(i);
            result[i] = range;
        }
        return result;
    }
    public void insert(List<int[]> list, int start, int end) {
        int[] range = new int[2];
        range[0] = start;
        range[1] = end;
        list.add(range);
    }
}
```

### 435. Non-overlapping Intervals
- [Link](https://leetcode.com/problems/non-overlapping-intervals/description/)
- Tags: Greedy
- Stars: 2

### 134. Gas Station
- [Link](https://leetcode.com/problems/gas-station/)
- Tags: Greedy
- Stars: 3

#### pseudo two pointers
1. `while(slow < len && gas[slow]<0) slow++;` cannot be written as `while(slow < len && gas[slow]<=0) slow++;`. Consider cases like `gas[i] == cost[i]` for all possible i. 
2. Let's assume `remain[i] = gas[i]-cost[i]`. When `sum(remain, slow, fast) < 0`, all the stations between slow and fast cannot satisfy the demand.  
This idea comes from the fact that the array is an non-decreasing array.  
We know that when we have `slow` fixed and start considering `fast`, any station i between slow and fast should hold `sum(remain, slow, i) >= 0`. 
Let's assume there is an station k between slow and fast, which satisfy `sum(remain, k, fast) > 0`.
From this, we can easily get `sum(remain, slow, k-1) < 0`, which is contradicted to the assumption. 
Therefore, as long as `sum(remain, slow, fast) < 0`, we can skip all the stations between slow and fast, and set `slow = fast + 1`. 
3. If `sum(gas) >= sum(cost)`, there must be a solution. Otherwise, no solution. 
```java
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        for(int i=0; i<gas.length; i++) gas[i] -= cost[i];
        int slow = 0, len = gas.length;
        while(slow < len && gas[slow]<0) slow++;
        if(slow == len) return -1;
        int fast = slow, result = 0;
        while(true){
            result += gas[fast];
            fast = (fast+1)%len;
            if(result < 0){
                if(fast > slow) {
                    slow = fast;
                    while(slow < len && gas[slow]<0) slow++;
                    if(slow == len) return -1;
                    result = gas[slow];
                    fast = (slow+1)%len;
                }
                else return -1;
            }
            if(fast == slow) return slow;
        }
    }
}
```

Updated 2019.7.31
```java
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int[] delta = new int[gas.length];
        int n = delta.length;
        for(int i=0; i<n; i++) delta[i] = gas[i] - cost[i];
        int start = 0, balance = 0, curr = 0;
        while(start < n) {
            balance += delta[curr];
            if (balance < 0) {
                if (curr < start) return -1;
                start = ++curr;
                balance = 0;
                continue;
            }
            curr = (curr + 1) % n;
            if (curr == start) return start;
        }
        return -1;
    }
}
```

### 33. Search in Rotated Sorted Array
- [Link](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- Tags: Array, Binary Search
- Stars: 2

#### Method 1: use binary search for 3 times
The most direct method is to find the pivot, and then separate `nums` into two subarrays according to the position of the pivot, and then apply binary search to each subarray. 
```java
class Solution {
    public int search(int[] nums, int target) {
        if(nums.length == 0) return -1;
        int l = 0, r = nums.length-1;
        while(l+1 < r){
            int mid = l + ((r-l)>>1);
            if(nums[mid] < nums[0]) r = mid;
            else l = mid;
        }
        // System.out.println(l + " " + r);
        int a = Arrays.binarySearch(nums, 0, r, target);
        if(a<0){
            int b = Arrays.binarySearch(nums, r, nums.length, target);
            if(b<0) return -1;
            return b;
        }
        return a;
    }
}
```

#### Method 2: use binary search for 2 times
Just like the above solution, we find the pivot first. Once we have pivot, we can establish a mapping of indices before and after the rotation. 

Be careful to a special case where `shift == 0` (i.e. the position of the smallest element)

Time: 6 ms
```java
class Solution {
    public int search(int[] nums, int target) {
        // compute `shift`
        if(nums.length == 0) return -1;
        int l = 0, r = nums.length-1;
        int shift = -1;
        if(nums[l] < nums[r]) shift = 0;
        else {
            while(l+1 < r){
                int mid = l + ((r-l)>>1);
                if(nums[mid] < nums[0]) r = mid;
                else l = mid;
            }
            shift = r;
        }
        // use `shift` to map index
        l = 0;
        r = nums.length-1;
        while(l<=r) {
            int mid = l + ((r-l)>>1);
            int midAfterRotate = (mid+shift)%nums.length; //idxAfterRotate(mid, shift, nums.length);
            if(nums[midAfterRotate] == target) return midAfterRotate;
            else if(nums[midAfterRotate] > target) r = mid - 1;
            else l = mid + 1;
        }
        return -1;
    }
    // private int idxAfterRotate(int i, int k, int len){
    //     return (i+k)%len;
    // }
}
```

#### Method 3: direct binary search
`nums` is almost in a sorted order, and we can take advantage of it! 

we can divide this problem into two cases:  
1. `target >= nums[0]`: target is in the left part
2. `target < nums[0]`: target is in the right part

For each case, we only need to deal with a situation when `nums[mid]` is in the wrong part.

Time: 6 ms
```java
class Solution {
    public int search(int[] nums, int target) {
        if(nums.length == 0) return -1;
        int l = 0, r = nums.length - 1;
        // take care of the special case
        if(nums[l] < nums[r]){
            int idx = Arrays.binarySearch(nums, l, r+1, target);
            return idx<0 ? -1 : idx;
        }
        if(target >= nums[0]){
            // target is in the left part
            while(l<=r){
                int mid = l + ((r-l)>>1);
                if(nums[mid] < nums[0]) r = mid-1;
                else if(nums[mid] == target) return mid;
                else if(nums[mid] > target) r = mid-1;
                else l = mid + 1;
            }
            return -1;
        }
        else {
            // target is in the right part
            while(l<=r){
                int mid = l + ((r-l)>>1);
                if(nums[mid] >= nums[0]) l = mid+1;
                else if(nums[mid] == target) return mid;
                else if(nums[mid] > target) r = mid-1;
                else l = mid+1;
            }
            return -1;
        }
    }
}
```

Updated 2019.7.31
```java
class Solution {
    public int search(int[] nums, int target) {
        if (nums.length == 0) return -1;
        if (nums[0] < nums[nums.length - 1]) 
            return Math.max(-1, Arrays.binarySearch(nums, 0, nums.length, target));
        int l = 0, r = nums.length - 1;
        while (l+1 < r) {
            int mid = l + ((r-l)>>1);
            if(nums[mid] > target) {
                if (target < nums[0] && nums[mid] >= nums[0]) l = mid+1;
                else r = mid;
            } else if (nums[mid] < target) {
                if (target >= nums[0] && nums[mid] < nums[0]) r = mid-1;
                else l = mid;
            } else return mid;
        }
        if (nums[l] == target) return l;
        else if (nums[r] == target) return r;
        return -1;
    }
}
```

#### 2019.9.13 standard method by [大雪菜]
- time: 100%
- space: 15.73%
- interviewLevel
- attention: `if (nums[l] == target) return l;` only. `if (nums[r] == target) return l;` is wrong. 
```java
class Solution {
    public int search(int[] nums, int target) {
        if (nums.length == 0) return -1;
        int l = 0, r = nums.length - 1;
        while(l<r) {
            int mid = l+r >> 1;
            if (nums[mid] <= nums[nums.length-1]) r = mid;
            else l = mid + 1;
        }
        if (target <= nums[nums.length - 1]) {
            r = nums.length - 1;
        } else {
            l = 0;
            r--;
        }
        while(l<r) {
            int mid = l+r >> 1;
            if (nums[mid] >= target) r = mid;
            else l = mid + 1;
        }
        if (nums[l] == target) return l;
        return -1;
    }
}
```

### 150. Evaluate Reverse Polish Notation
- [Link](https://leetcode.com/problems/evaluate-reverse-polish-notation/)
- Tags: Stack
- Stars: 1

#### stack
- attention: the order of parameters in `compute` function

`token.length()>1` is used to deal with negative numbers.

```java
class Solution {
    public int evalRPN(String[] tokens) {
        Stack<Integer> st = new Stack<>();
        for(String token: tokens){
            // System.out.println(st.toString());
            if(Character.isDigit(token.charAt(0)) || token.length()>1){ 
                int num = Integer.parseInt(token);
                st.push(num);
            }
            else {
                st.push(compute(st.pop(), st.pop(), token.charAt(0)));
            }
        }
        // System.out.println(st.toString());
        return st.pop();
    }
    private int compute(int b, int a, char c){
        if(c == '+') return a+b;
        else if(c=='-') return a-b;
        else if(c=='*') return a*b;
        else return a/b;
    }
}
```

Updated 2019.8.4

```java
class Solution {
    public int evalRPN(String[] tokens) {
        Stack<Integer> st = new Stack<>();
        for (String token: tokens) {
            if (Character.isDigit(token.charAt(token.length()-1))) st.add(Integer.parseInt(token));
            else {
                int b = st.pop(), a = st.pop();
                char c = token.charAt(0);
                if (c == '+') st.add(a+b);
                else if (c == '-') st.add(a-b);
                else if (c == '*') st.add(a*b);
                else if (c == '/') st.add(a/b);
            }
        }
        return st.pop();
    }
}
```

### 55. Jump Game
- [Link](https://leetcode.com/problems/jump-game/)
- Tags: Array, Greedy
- Stars: 1

#### DP
```java
class Solution {
    public boolean canJump(int[] nums) {
        int dp = nums[0];
        for(int i=1; i<nums.length; i++){
            if(dp < i) return false;
            dp = Math.max(dp, i+nums[i]);
        }
        return dp>=nums.length-1;
    }
}
```

### 2. Add Two Numbers
- [Link](https://leetcode.com/problems/add-two-numbers/)
- Tags: Linked List, Math
- Stars: 2

#### simple solution beats 91.83% in time and 96.99% in space
- attention: there is one more step/iteration after the while loop.

```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int cin = 0;
        ListNode head = new ListNode(-1);
        ListNode curr = head;
        while(l1 != null || l2 != null){
            int temp = cin;
            if(l1 != null) {
                temp += l1.val;
                l1 = l1.next;
            }
            if(l2 != null){
                temp += l2.val;
                l2 = l2.next;
            }
            curr.next = new ListNode(temp%10);
            curr = curr.next;
            cin = temp/10;
        }
        if(cin > 0)
            curr.next = new ListNode(cin);
        return head.next;
    }
}
```

### 54. Spiral Matrix
- [Link](https://leetcode.com/problems/spiral-matrix/)
- Tags: Array
- Stars: 1

#### recursive
The key of this problem lies in the boundaries.

`if(m != 2*k+1)` and `if(n != 2*k+1)` are used to deal with rectangular matrices to prevent duplicates. 
```java
class Solution {
    List<Integer> result = new ArrayList<>();
    int m,n;
    public List<Integer> spiralOrder(int[][] matrix) {
        if(matrix.length == 0 || matrix[0].length == 0) return result;
        this.m = matrix.length;
        this.n = matrix[0].length;
        spiralOrder(matrix, 0);
        return result;
    }
    private void spiralOrder(int[][] matrix, int k){
        if(k>(m-1)/2 || k>(n-1)/2) return ;
        result.add(matrix[k][k]);
        for(int j=k+1; j<n-k; j++) result.add(matrix[k][j]);
        for(int i=k+1; i<m-k; i++) result.add(matrix[i][n-k-1]);
        if(m != 2*k+1)
            for(int j=n-k-2; j>=k; j--) result.add(matrix[m-k-1][j]);
        if(n != 2*k+1)
        for(int i=m-k-2; i>=k+1; i--) result.add(matrix[i][k]);
        spiralOrder(matrix, k+1);
    }
}
```

#### 2019.8.11 
- time: 100%
- space: 100%
- interviewLevel

```java
class Solution {
    List<Integer> result = new ArrayList<>();
    public List<Integer> spiralOrder(int[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) return result;
        DFS(matrix, 0, 0, matrix.length, matrix[0].length);
        return result;
    }
    public void DFS(int[][] matrix, int a, int b, int nRows, int nCols) {
        if (nRows <= 0 || nCols <= 0) return;
        if (nRows == 1) for(int j=b; j<b+nCols; j++) result.add(matrix[a][j]);
        else if (nCols == 1) for(int i=a; i<a+nRows; i++) result.add(matrix[i][b]);
        else {
            for(int j=b; j<b+nCols-1; j++) result.add(matrix[a][j]);
            for(int i=a; i<a+nRows-1; i++) result.add(matrix[i][b+nCols-1]);
            for(int j=b+nCols-1; j>b; j--) result.add(matrix[a+nRows-1][j]);
            for(int i=a+nRows-1; i>a; i--) result.add(matrix[i][b]);
        }
        DFS(matrix, a+1, b+1, nRows-2, nCols-2);
    }
}
```

### 152. Maximum Product Subarray
- [Link](https://leetcode.com/problems/maximum-product-subarray/)
- Tags: Array, Dynamic Programming
- Stars: 2

#### DP, space-optimized
`dp[i]` means the largest product of the subarray ended up with nums[i]  
Here we use `maxVal` and `minVal` to record the local state of an iteration. 
```java
class Solution {
    public int maxProduct(int[] nums) {
        // int[] dp = new int[nums.length];
        int dp;
        int maxVal, minVal;
        // maxVal = minVal = dp[0] = nums[0];
        maxVal = minVal = dp = nums[0];
        for(int i=1; i<nums.length; i++){
            int num = nums[i];
            if(num > 0){
                maxVal = Math.max(num, maxVal * num);
                minVal = Math.min(num, minVal * num);
            }
            else if(num < 0){
                int nextMinVal = Math.min(num, maxVal * num);
                maxVal = Math.max(num, minVal * num);
                minVal = nextMinVal;
            }
            else {
                maxVal = minVal = 0;
            }
            // dp[i] = maxVal;
            if(dp < maxVal) dp = maxVal;
        }
        // int result = Integer.MIN_VALUE;
        // for(int res: dp)
        //     if(result < res) result = res;
        // return result;
        return dp;
    }
}
```

Updated 2019.8.12
- time: 99.15%
- space: 8.54%
- attention: a temporary variable `temp` is needed for the case of `nums[i]<0`.

```java
class Solution {
    public int maxProduct(int[] nums) {
        int max = nums[0], min = nums[0], result = nums[0];
        for(int i=1; i<nums.length; i++) {
            if (nums[i]>0) {
                max = Math.max(max*nums[i], nums[i]);
                min = Math.min(min*nums[i], nums[i]);
            } else if (nums[i] < 0) {
                int temp = Math.max(min*nums[i], nums[i]);
                min = Math.min(max*nums[i], nums[i]);
                max = temp;
            } else max = min = 0;
            result = Math.max(result, max);
        }
        return result;
    }
}
```

### 50. Pow(x, n)
- [Link](https://leetcode.com/problems/powx-n/)
- Tags: Math, Binary Search
- Stars: 2

#### iterative
```java
class Solution {
    public double myPow(double x, int n) {
        if(n == 0) return 1;
        if(x == 0) return 0;
        if(n == Integer.MIN_VALUE) {
            if(x > 1 || x<-1) return 0;
            return 1;
        }
        if(n<0) {
            n = -n;
            x = 1/x;
        }
        HashMap<Integer, Double> map = new HashMap<>();
        map.put(1, x);
        int currN = 1;
        while((currN<<1) > 0 && (currN<<1) < n){
            double temp = map.get(currN);
            currN <<= 1;
            map.put(currN, temp * temp);
        }
        double result = 1;
        while(n>0){
            while(n < currN) currN >>= 1;
            result *= map.get(currN);
            n -= currN;
        }
        return result;
    }
}
```

#### recursive
- time: 100%
- space: 5.88%
- interviewLevel

```java
class Solution {
    public double myPow(double x, int n) {
        if(n == 0) return 1;
        if(n == Integer.MIN_VALUE){
            return myPow(x*x, n>>1);
        } 
        if(n<0) {
            n = -n;
            x = 1/x;
        }
        return (n%2) == 0 ? myPow(x*x, n>>1) : x * myPow(x*x, n>>1);
    }
}
```

2019.8.12 Similar idea
- time: 100%
- space: 5.88%

```java
class Solution {
    public double myPow(double x, int n) {
        if (x == 0) return 0;
        if (n == Integer.MIN_VALUE) return myPow(x, n+1) / x;
        if (n == 0) return 1;
        if (n%2 != 0) return n < 0 ? myPow(x, n+1) / x : myPow(x, n-1) * x;
        double temp = myPow(x, n/2);
        return temp * temp;
    }
}
```

### 3. Longest Substring Without Repeating Characters
- [Link](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- Tags: Hash Table, Two Pointers, String
- Stars: 2

#### two pointers, thinking in a DP way
Notice that `s` may contain any ASCII character. 
```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if(s == null || s.length() == 0) return 0;
        boolean[] alphabet = new boolean[256];
        int head = 0, tail = 1;
        alphabet[s.charAt(0)] = true;
        int result = 1;
        while(tail < s.length()){
            if(alphabet[s.charAt(tail)]){
                do {
                    alphabet[s.charAt(head++)] = false;
                } while(s.charAt(head-1) != s.charAt(tail));
            }
            alphabet[s.charAt(tail++)] = true;
            if(result < tail-head) result = tail-head;
        }
        return result;
    }
}
```

Updated 2019.8.12
- time: 91.66%
- space: 99.76%
- interviewLevel
```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s.length() == 0) return 0;
        int i=0, j=1, result = 1;
        boolean[] stat = new boolean[256];
        stat[s.charAt(0)] = true;
        while(j<s.length()) {
            while (stat[s.charAt(j)]) stat[s.charAt(i++)] = false;
            stat[s.charAt(j++)] = true;
            result = Math.max(result, j-i);
        }
        return result;
    }
}
```

### 138. Copy List with Random Pointer
- [Link](https://leetcode.com/problems/copy-list-with-random-pointer/)
- Tags: Hash Table, Linked List
- Stars: 1

#### 2 pass copy with Hash Mapping, O(n) space and O(n) time
```java
class Solution {
    public Node copyRandomList(Node head) {
        if(head == null) return null;
        Node newHead = new Node(head.val, head.next, head.random);
        HashMap<Node, Node> map = new HashMap<>();
        map.put(head, newHead);
        Node curr = newHead;
        while(curr.next != null){
            Node temp = curr.next;
            curr.next = new Node(temp.val, temp.next, temp.random);
            map.put(temp, curr.next);
            curr = curr.next;
        }
        curr = newHead;
        while(curr != null){
            if(curr.random != null)
                curr.random = map.get(curr.random);
            curr = curr.next;
        }
        return newHead;
    }
}
```

#### 3 pass O(1) space and O(n) time
Since the original Linkedlist has to remain unchanged, we need to restore next pointer of the original nodes.  
Notice that you cannot setup the random pointers while extracting the new Head at the same time. 
```java
class Solution {
    public Node copyRandomList(Node head) {
        if(head == null) return null;
        // interleaving copy
        Node curr = head;
        while(curr != null){
            curr.next = new Node(curr.val, curr.next, curr.random);
            curr = curr.next.next;
        }
        Node newHead = head.next;
        // setup the random pointers
        curr = head.next;
        while(true){
            if(curr.random != null)
                curr.random = curr.random.next;
            if(curr.next == null) break;
            curr = curr.next.next;
        }
        // extract the newHead
        curr = head;
        Node copy = newHead;
        while(true) {
            curr.next = copy.next;
            curr = curr.next;
            if(copy.next == null) break;
            copy.next = curr.next;
            copy = copy.next;
        }
        return newHead;
    }
}
```

Updated 2019.8.12
- time: 100%
- space: 99.07%
- interviewLevel
- attention: step2 and step3 cannot be combined with each other.
- attention: when returning the new linked list in step3, we cannot return `head.next`.

```java
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        // step1: copy all the nodes, and inserts the cloned nodes just after the corresponding node. 
        Node curr = head;
        while(curr != null) {
            curr.next = new Node(curr.val, curr.next, curr.random);
            curr = curr.next.next;
        }
        // step2: ajust random pointers for the copied nodes.
        curr = head;
        while(curr != null) {
            Node newNode = curr.next, next = curr.next.next;
            if (newNode.random != null) newNode.random = newNode.random.next;
            curr = next;
        }
        // step3: extract the cloned nodes from the linked list.
        curr = head;
        Node newHead = head.next;
        while(curr != null) {
            Node newNode = curr.next, next = curr.next.next;
            curr.next = next;
            newNode.next = next == null ? null : next.next;
            curr = next;
        }
        return newHead;
    }
}
```

### 179. Largest Number
- [Link](https://leetcode.com/problems/largest-number/)
- Tags: Sort
- Stars: 4

#### Arrays.sort Comparator
1. be careful about these cases: comparing 3456 & 345, 3451 & 345
2. remember to remove the leading zeroes
3. `Arrays.sort(xxx, new Comparator<xxx>() {})` can only be applied to object arrays
```java
class Solution {
    public String largestNumber(int[] nums) {
        // convert nums to a String Array
        String[] strs = new String[nums.length];
        for(int i=0; i<nums.length; i++)
            strs[i] = Integer.toString(nums[i]);
        // Self-defined sorting
        Arrays.sort(strs, new Comparator<String>() {
            @Override
            public int compare(String a, String b) {
                int i=0; 
                while(i<a.length() && i<b.length()){
                    if(a.charAt(i) != b.charAt(i))
                        return b.charAt(i) - a.charAt(i);
                    i++;
                }
                if(i == a.length() && i == b.length()) return 0;
                else if(i == b.length())
                    return compare(a.substring(i, a.length()) + b, a);
                else 
                    return compare(b, b.substring(i, b.length()) + a);
            }
        });
        // join the strings
        String result = String.join("", Arrays.asList(strs));
        int k = 0;
        // remove the leading zeros
        while(k<result.length() && result.charAt(k) == '0') k++;
        if(k == result.length()) return "0";
        return result.substring(k, result.length());
    }
}
```

#### smarter but much slower idea using concatenation
- time: 10.51%
- space: 82.22%
- attention: You have to remove the leading zeros
- attention: The comparing function `-(a+b).compareTo(b+a)` is the best solution to understand.

```java
class Solution {
    public String largestNumber(int[] nums) {
        String[] strs = new String[nums.length];
        for(int i=0; i<nums.length; i++)
            strs[i] = Integer.toString(nums[i]);
        Arrays.sort(strs, (String a, String b)->(b+a).compareTo(a+b));
        String result = String.join("", Arrays.asList(strs));
        int k = 0;
        while(k<result.length() && result.charAt(k) == '0') k++;
        if(k == result.length()) return "0";
        return result.substring(k, result.length());
    }
}
```

### 98. Validate Binary Search Tree
- [Link](https://leetcode.com/problems/validate-binary-search-tree/)
- Tags: Tree, BFS
- Stars: 4

#### recursive
- attention: you need to take care of cases like `root.val == Integer.MIN_VALUE` and `root.val == Integer.MAX_VALUE`, because under these circumstances, the boundaries might overflow.

```java
class Solution {
    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
    public boolean isValidBST(TreeNode root, int l, int r){
        if(root == null) return true;
        if(root.val > r || root.val < l) return false;
        if(root.val == Integer.MIN_VALUE && root.left != null) return false;
        if(root.val == Integer.MAX_VALUE && root.right != null) return false;
        return isValidBST(root.left, l, root.val-1) && isValidBST(root.right, root.val+1, r);
    }
}
```

### 127. Word Ladder
- [Link](https://leetcode.com/problems/word-ladder/)
- Tags: BFS
- Stars: 1

#### My BFS
1. You need to mark all the words that has been visited. 
2. You can initiate a boolean array for marking, but removing elements from wordList directly seems faster. 
```java
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Queue<String> qu = new LinkedList<>();
        qu.add(beginWord);
        // boolean[] mark = new boolean[wordList.size()];
        int count = 1, level = 0;
        while(!qu.isEmpty()){
            String curr = qu.poll();
            count--;
            if(curr.equals(endWord)) return level+1;
            for(int i=0; i<wordList.size(); i++){
                String word = wordList.get(i);
                // if(!mark[i] && isSimilar(curr, word)) {qu.add(word); mark[i] = true;}
                if(isSimilar(curr, word)) {
                    qu.add(word); 
                    wordList.remove(i--);
                }
            }
            if(count == 0){
                count = qu.size();
                level++;
            }
        }
        return 0;
    }
    private boolean isSimilar(String a, String b){
        // if(a.length() != b.length()) return false;
        int count = 0;
        for(int i=0; i<a.length(); i++){
            if(a.charAt(i) != b.charAt(i)) count++;
            if(count == 2) return false;
        }
        return true;
    }
}
```

#### A faster BFS
```java
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        HashSet<String> set = new HashSet<>(wordList);
        Queue<String> qu = new LinkedList<>();
        qu.add(beginWord);
        int count = 1, level = 0;
        while(!qu.isEmpty()){
            String curr = qu.poll();
            count--;
            if(curr.equals(endWord)) return level+1;
            char[] chrs = curr.toCharArray();
            for(int i=0; i<chrs.length; i++){
                char temp = chrs[i];
                for(char c='a'; c<='z'; c++){
                    if(c == temp) continue;
                    chrs[i] = c;
                    String word = new String(chrs);
                    if(set.contains(word)) {
                        qu.add(word);
                        set.remove(word);
                    }
                }
                chrs[i] = temp;
            }
            if(count == 0){
                count = qu.size();
                level++;
            }
        }
        return 0;
    }
}
```

### 130. Surrounded Regions
- [Link](https://leetcode.com/problems/surrounded-regions/)
- Tags: DFS, BFS, Union Find
- Stars: 2

#### My BFS
```java
class Solution {
    int[][] map;
    public void solve(char[][] board) {
        if(board.length == 0 || board[0].length == 0) return ;
        map = new int[board.length][board[0].length];
        for(int i=0; i<board.length; i++)
            for(int j=0; j<board[0].length; j++){
                if(isSurrounded(board, i, j)) flip(board, i, j, 'M', 'X');
                else flip(board, i, j, 'M', 'O');
            }
    }
    private boolean isSurrounded(char[][] board, int i, int j){
        if(board[i][j] == 'X' || board[i][j] == 'M') return true;
        if(i==0 || i==board.length-1 || j==0 || j==board[0].length-1) return false;
        if(map[i][j] == 1) return true;
        else if(map[i][j] == -1) return false;
        board[i][j] = 'M';
        boolean temp = isSurrounded(board, i-1, j) && isSurrounded(board, i+1, j) && isSurrounded(board, i, j-1) && isSurrounded(board, i, j+1);
        map[i][j] = temp ? 1 : -1;
        return temp;
    }
    private void flip(char[][] board, int i, int j, char src, char dst){
        if(board[i][j] == src) {
            board[i][j] = dst;
            flip(board, i+1, j, src, dst);
            flip(board, i-1, j, src, dst);
            flip(board, i, j+1, src, dst);
            flip(board, i, j-1, src, dst);
        }
    }
}
```

#### My faster BFS, in an outside-in way
Flip all the un-surrounded 'O's into 'M' in an outside-in way. Then, iterate the board and flip the remaining 'O's into 'X'. Finally, flip all the 'M's into 'O'. 
```java
class Solution {
    int m, n;
    public void solve(char[][] board) {
        if(board.length == 0 || board[0].length == 0) return ;
        m = board.length;
        n = board[0].length;
        // Flip all the un-surrounded 'O's into 'M' in an outside-in way.
        for(int i=0; i<m; i++){
            if(board[i][0] == 'O') flip(board, i, 0, 'O', 'M');
            if(board[i][n-1] == 'O') flip(board, i, n-1, 'O', 'M');
        }
        for(int j=1; j<n-1; j++){
            if(board[0][j] == 'O') flip(board, 0, j, 'O', 'M');
            if(board[m-1][j] == 'O') flip(board, m-1, j, 'O', 'M');
        }
        // flip the remaining 'O's into 'X'.
        for(int i=0; i<m; i++)
            for(int j=0; j<n; j++)
                if(board[i][j] == 'O') flip(board, i, j, 'O', 'X');
        // flip all the 'M's into 'O'. 
        for(int i=0; i<m; i++)
            for(int j=0; j<n; j++)
                if(board[i][j] == 'M') flip(board, i, j, 'M', 'O');
    }
    private void flip(char[][] board, int i, int j, char src, char dst){
        if(i>=0 && j>=0 && i<m && j<n && board[i][j] == src){
            board[i][j] = dst;
            flip(board, i+1, j, src, dst);
            flip(board, i-1, j, src, dst);
            flip(board, i, j+1, src, dst);
            flip(board, i, j-1, src, dst);
        }
    }
}
```

Updated 2019.8.12
- time: 27.14%
- space: 50%

```java
class Solution {
    public void solve(char[][] board) {
        if(board.length == 0 || board[0].length == 0) return;
        int m = board.length, n = board[0].length;
        for(int j=0; j<n-1; j++) convertTo(board, 0, j, 'O', 'B');
        for(int i=0; i<m-1; i++) convertTo(board, i, n-1, 'O', 'B');
        for(int j=n-1; j>0; j--) convertTo(board, m-1, j, 'O', 'B');
        for(int i=m-1; i>0; i--) convertTo(board, i, 0, 'O', 'B');
        for(int i=1; i<m-1; i++)
            for(int j=1; j<n-1; j++)
                convertTo(board, i, j, 'O', 'X');
        for(int i=0; i<m; i++)
            for(int j=0; j<n; j++)
                convertTo(board, i, j, 'B', 'O');
    }
    public void convertTo(char[][] board, int i, int j, char src, char dst) {
        int m = board.length, n = board[0].length;
        if (i<0 || i>=m || j<0 || j>=n || board[i][j] != src) return;
        board[i][j] = dst;
        convertTo(board, i-1, j, src, dst);
        convertTo(board, i+1, j, src, dst);
        convertTo(board, i, j-1, src, dst);
        convertTo(board, i, j+1, src, dst);
    }
}
```

### 91. Decode Ways
- [Link](https://leetcode.com/problems/decode-ways/)
- Tags: String, Dynamic Programming
- Stars: 3

#### DP, space optimized
`s.charAt(i) == 0` is a special case. 

```java
class Solution {
    public int numDecodings(String s) {
        if(s.length() == 0 || s.charAt(0) == '0') return 0;
        // int[] dp = new int[s.length()];
        // dp[0] = 1;
        int dp1 = 1, dp2 = 1;
        for(int i=1; i<s.length(); i++){
            int dp=0;
            if(s.charAt(i) == '0'){
                if(s.charAt(i-1) == '1' || s.charAt(i-1) == '2') dp = dp2;
                else return 0;
            }
            else {
                dp = dp1;
                if(s.charAt(i-1) == '1' || 
                   (s.charAt(i-1) == '2' && s.charAt(i) < '7'))
                    dp += dp2;
            }
            dp2 = dp1;
            dp1 = dp;
        }
        return dp1;
    }
}
```

### 29. Divide Two Integers
- [Link](https://leetcode.com/problems/divide-two-integers/)
- Tags: Math, Binary Search
- Stars: 2

#### subtract and double
```java
class Solution {
    public int divide(int dividend, int divisor) {
        if(dividend == 0) return 0;
        int sign = (dividend<0 && divisor<0) || (dividend>0 && divisor>0) ? 1 : -1;
        //convert to positive and avoid overflow
        if(divisor == Integer.MIN_VALUE) return dividend == Integer.MIN_VALUE ? 1 : 0;
        if(divisor < 0) divisor = -divisor;
        int result = 0;
        if(dividend == Integer.MIN_VALUE) {
            if(divisor == 1 && sign == 1) return Integer.MAX_VALUE;
            dividend += divisor;
            result++;
        }
        if(dividend < 0) dividend = -dividend;
        //use map to record
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(1, divisor);
        int curr = 1;
        while(dividend >= map.get(curr)){
            // experiment shows that adding the following two lines will makes it faster, though not necessary. 
            dividend -= map.get(curr);
            result += curr;
            // avoid overflow
            if((map.get(curr)<<1) < 0) break;
            map.put(curr<<1, map.get(curr)<<1);
            curr <<= 1;
        }
        while(dividend >= map.get(1)){
            while(dividend < map.get(curr)) curr>>=1;
            dividend -= map.get(curr);
            result += curr;
        }
        return sign * result;
    }
}
```

#### Same algorithm from the computer architecture course
- time: 100%
- space: 5.28%

```java
class Solution {
    public int divide(int dividend, int divisor) {
        long quot = 0L, div = 0L, sign = 1L,
            dividendL = (long)dividend, divisorL = (long)divisor;
        if (dividendL < 0) {
            sign *= -1L;
            dividendL *= -1L;
        }
        if (divisorL < 0) {
            sign *= -1L;
            divisorL *= -1L;
        }
        for (int i=31; i>=0; i--) {
            div <<= 1;
            div |= (1L&(dividendL>>i));
            quot <<= 1;
            if (div >= divisorL) {
                div -= divisorL;
                quot |= 1L;
            }
        }
        quot *= sign;
        if (quot < Integer.MIN_VALUE || quot > Integer.MAX_VALUE)
            quot = Integer.MAX_VALUE;
        return (int)quot;
    }
}
```

### 8. String to Integer (atoi)
- [Link](https://leetcode.com/problems/string-to-integer-atoi/)
- Tags: Math, String
- Stars: 1

#### boundary check
```java
class Solution {
    public int myAtoi(String str) {
        if(str.length() == 0) return 0;
        int i=0;
        // remove leading white space
        while(i<str.length() && str.charAt(i) == ' ') i++;
        // detect plus/minus sign
        int sign = 1;
        if(i<str.length() && str.charAt(i) == '-') {sign = -1; i++;}
        else if(i<str.length() && str.charAt(i) == '+') {sign = 1; i++;}
        // remove leading zeros
        while(i<str.length() && str.charAt(i) == '0') i++;
        // parse Integer
        long result = 0;
        int start = i;
        for(; i<str.length() && i-start < 11; i++){
            if(!Character.isDigit(str.charAt(i))) break;
            result *= 10;
            result += str.charAt(i) - '0';
        }
        result *= sign;
        if(result < Integer.MIN_VALUE) return Integer.MIN_VALUE;
        else if (result > Integer.MAX_VALUE) return Integer.MAX_VALUE;
        return (int)result;
    }
}
```

Updated 2019.7.31
```java
class Solution {
    public int myAtoi(String str) {
        int i = 0;
        long result = 0;
        // prefix whitespaces
        while (i < str.length() && str.charAt(i) == ' ') i++;
        if (i >= str.length()) return 0;
        // optional plus and minus sign
        int sign = 1;
        if (str.charAt(i) == '-') {
            sign = -1; i++;
        } else if (str.charAt(i) == '+') i++;
        else if (!Character.isDigit(str.charAt(i))) return 0;
        // numerical digits
        if (i >= str.length()) return 0;
        while (i<str.length() && Character.isDigit(str.charAt(i))) {
            result *= 10;
            result += str.charAt(i++) - '0';
            // deal with the overflow problem
            if (result < Integer.MIN_VALUE || result > Integer.MAX_VALUE) break;
        }
        // clip the result
        return (int)(Math.max(Math.min(sign*result, Integer.MAX_VALUE), Integer.MIN_VALUE));
    }
}
```

### 222. Count Complete Tree Nodes
- [Link](https://leetcode.com/problems/count-complete-tree-nodes/)
- Tags: Binary Search, Tree
- Stars: 2

#### Binary Search in a tree
Compute the left height `h` in each iteration. 
if #nodes of right subtree > #perfect complete tree with height `h-2`, we can conclude that the left subtree must be a perfect complete tree with height `h-1`.
```java
class Solution {
    public int countNodes(TreeNode root) {
        if(root == null) return 0;
        int height = getLeftHeight(root);
        if(height == 0) return 1;
        if(height == 1) return root.right == null ? 2 : 3;
        int rightNum = countNodes(root.right);
        if(rightNum > computeTotalNodes(height-2))
            return computeTotalNodes(height-1) + rightNum + 1;
        return countNodes(root.left) + rightNum + 1;
    }
    private int computeTotalNodes(int height){
        return (1<<(height+1))-1;
    }
    private int getLeftHeight(TreeNode root){
        int count = 0;
        while(root.left != null){
            count++;
            root = root.left;
        }
        return count;
    }
}
```

#### concise version
```java
class Solution {
    public int countNodes(TreeNode root) {
        int height = getHeight(root);
        if(height == 0) return 0;
        if(getHeight(root.right) == height-1)
            return (1<<height-1) + countNodes(root.right);
        return countNodes(root.left) + (1<<(height-2));
    }
    private int getHeight(TreeNode root){
        int count = 0;
        while(root != null){
            count++;
            root = root.left;
        }
        return count;
    }
}
```

#### 2019.8.3 recursive
- time: 100%
- space: 100%
- interviewLevel

```java
class Solution {
    public int countNodes(TreeNode root) {
        if (root == null) return 0;
        int leftHeight = getLeftHeight(root);
        int rightHeight = getRightHeight(root);
        if (leftHeight == rightHeight) return (int)(Math.pow(2, leftHeight)) - 1;
        return countNodes(root.left) + 1 + countNodes(root.right);
    }
    public int getLeftHeight(TreeNode root) {
        if (root == null) return 0;
        int count = 0;
        while (root != null) {
            count++;
            root = root.left;
        }
        return count;
    }
    public int getRightHeight(TreeNode root) {
        if (root == null) return 0;
        int count = 0;
        while (root != null) {
            count++;
            root = root.right;
        }
        return count;
    }
}
```

Updated

```java
class Solution {
    public int countNodes(TreeNode root) {
        return countNodes(root, -1, -1);
    }
    public int countNodes(TreeNode root, int leftHeight, int rightHeight) {
        if (root == null) return 0;
        if (leftHeight == -1) leftHeight = getLeftHeight(root);
        if (rightHeight == -1) rightHeight = getRightHeight(root);
        if (leftHeight == rightHeight) return (int)(Math.pow(2, leftHeight)) - 1;
        return countNodes(root.left, leftHeight-1, -1) + 1 + countNodes(root.right, -1, rightHeight-1);
    }
    public int getLeftHeight(TreeNode root) {
        if (root == null) return 0;
        int count = 0;
        while (root != null) {
            count++;
            root = root.left;
        }
        return count;
    }
    public int getRightHeight(TreeNode root) {
        if (root == null) return 0;
        int count = 0;
        while (root != null) {
            count++;
            root = root.right;
        }
        return count;
    }
}
```


### 53. Maximum Subarray
- [Link](https://leetcode.com/problems/maximum-subarray/)
- Tags: Array, Divide and Conquer, Dynamic Programming
- Stars: 1

#### DP
```java
class Solution {
    public int maxSubArray(int[] nums) {
        if(nums.length == 0) return 0;
        int dp = nums[0], result = nums[0];
        for(int i=1; i<nums.length; i++){
            if(dp < 0) dp = nums[i];
            else dp += nums[i];
            if(result < dp) result = dp;
        }
        return result;
    }
}
```

#### Divide and Conquer
```java
// not implemented yet
```

### 343. Integer Break
- [Link](https://leetcode.com/problems/integer-break/)
- Tags: Math, Dynamic Programming
- Stars: 3

#### Math solution
- reference: https://leetcode.com/problems/integer-break/discuss/80721/Why-factor-2-or-3-The-math-behind-this-problem./85299
```java
class Solution {
    public int integerBreak(int n) {
        if(n == 2) return 1;
        if(n == 3) return 2;
        int result = 1;
        while(n>4){
            result *= 3;
            n -= 3;
        }
        result *= n;
        return result;
    }
}
```

Updated 2019.9.1
```java
class Solution {
    public int integerBreak(int n) {
        if (n < 4) return n-1;
        int ret = 1;
        while(n >= 3) {
            n -= 3;
            ret *= 3;
        }
        if (n == 2) ret *= 2;
        else if (n == 1) ret = ret/3*4;
        return ret;
    }
}
```

### 415. Add Strings
- [Link](https://leetcode.com/problems/add-strings/)
- Tags: Math
- Stars: 1

#### while loop with StringBuilder.insert(0, xxx)
```java
class Solution {
    public String addStrings(String num1, String num2) {
        StringBuilder sb = new StringBuilder();
        int i=num1.length()-1, j=num2.length()-1;
        int cin = 0;
        while(i>=0 || j>=0){
            int temp = cin;
            if(i>=0) temp += num1.charAt(i--)-'0';
            if(j>=0) temp += num2.charAt(j--)-'0';
            if(temp > 9){
                cin = 1;
                temp %= 10;
            }
            else cin = 0;
            sb.insert(0, temp);
        }
        if(cin > 0) sb.insert(0, cin);
        return sb.toString();
    }
}
```

### 43. Multiply Strings
- [Link](https://leetcode.com/problems/multiply-strings/)
- Tags: Math, String
- Stars: 2

#### StringBuilder.insert
```java
class Solution {
    public String multiply(String num1, String num2) {
        StringBuilder sb = new StringBuilder();
        int base = 0, carry = 0;
        while(++base <= num1.length() + num2.length()){
            for(int i=0; i<base && i<num1.length(); i++){
                int j = base-1-i;
                if(j >= num2.length()) continue;
                carry += (num1.charAt(num1.length()-1-i)-'0') * (num2.charAt(num2.length()-1-j)-'0');
            }
            sb.insert(0, carry%10);
            carry /= 10;
        }
        if(carry > 0) sb.insert(0, carry);
        for(int i=0; i<sb.length(); i++){
            if(sb.charAt(i) != '0') return sb.substring(i, sb.length());
        }
        return "0";
    }
}
```

#### 2019.8.11 int array
- time: 87.39%
- space: 100%
- interviewLevel

```java
class Solution {
    public String multiply(String num1, String num2) {
        int[] nums = new int[num1.length() + num2.length() + 5];
        for(int i=0; i<num1.length(); i++) {
            int a = num1.charAt(num1.length()-i-1) - '0';
            for(int j=0; j<num2.length(); j++) {
                int b = num2.charAt(num2.length()-j-1) - '0';
                nums[i+j] += a*b;
            }
        }
        for(int i=0; i<nums.length-1; i++) {
            nums[i+1] += nums[i]/10;
            nums[i] %= 10;
        }
        int idx = nums.length - 1;
        while(idx>=0 && nums[idx] == 0) idx--;
        if (idx < 0) return "0";
        StringBuilder sb = new StringBuilder();
        while(idx >= 0) sb.append(nums[idx--]);
        return sb.toString();
    }
}
```

### 349. Intersection of Two Arrays
- [Link](https://leetcode.com/problems/intersection-of-two-arrays/)
- Tags: Hash Table, Two Pointers, Binary Search, Sort
- Stars: 1

#### Hash Set 2ms
```java
class Solution {
    public int[] intersection(int[] nums1, int[] nums2) {
        List<Integer> list = new ArrayList<>();
        HashSet<Integer> set = new HashSet<>();
        for(int num: nums1) set.add(num);
        for(int num: nums2) {
            if(set.contains(num)){
                list.add(num);
                set.remove(num);
            }
        }
        int[] result = new int[list.size()];
        for(int i=0; i<list.size(); i++)
            result[i] = list.get(i);
        return result;
    }
}
```

#### Two pointers 2ms
```java
class Solution {
    public int[] intersection(int[] nums1, int[] nums2) {
        List<Integer> list = new ArrayList<>();
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int i=0, j=0;
        while(i<nums1.length && j<nums2.length){
            if(nums1[i] == nums2[j]){
                list.add(nums1[i]);
                i++; j++;
                while(i<nums1.length && nums1[i] == nums1[i-1]) i++;
                while(j<nums2.length && nums2[j] == nums2[j-1]) j++;
            }
            else if(nums1[i] > nums2[j]) {
                j++;
                while(j<nums2.length && nums2[j] == nums2[j-1]) j++;
            }
            else {
                i++;
                while(i<nums1.length && nums1[i] == nums1[i-1]) i++;
            }
        }
        int[] result = new int[list.size()];
        for(int k=0; k<list.size(); k++)
            result[k] = list.get(k);
        return result;
    }
}
```

### 350. Intersection of Two Arrays II
- [Link](https://leetcode.com/problems/intersection-of-two-arrays-ii/)
- Tags: Hash Table, Two Pointers, Binary Search, Sort
- Stars: 2

#### Hash Table O(n)time, O(n)space  2ms
```java
class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        List<Integer> list = new ArrayList<>();
        HashMap<Integer, Integer> map = new HashMap<>();
        for(int num: nums1)
            map.put(num, map.getOrDefault(num, 0)+1);
        for(int num : nums2){
            int temp = map.getOrDefault(num, 0);
            if(temp > 0){
                map.put(num, temp-1);
                list.add(num);
            }
        }
        int[] result = new int[list.size()];
        for(int i=0; i<list.size(); i++)
            result[i] = list.get(i);
        return result;
    }
}
```

#### Two pointers O(nlogn)time O(1) space, 1ms
```java
class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        List<Integer> list = new ArrayList<>();
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int i=0, j=0;
        while(i<nums1.length && j<nums2.length){
            if(nums1[i] < nums2[j]) i++;
            else if(nums1[i] > nums2[j]) j++;
            else {
                list.add(nums1[i]);
                i++; j++;
            }
        }
        int[] result = new int[list.size()];
        for(int k=0; k<list.size(); k++)
            result[k] = list.get(k);
        return result;
    }
}
```

### 101. Symmetric Tree
- [Link](https://leetcode.com/problems/symmetric-tree/)
- Tags: Tree, DFS, BFS
- Stars: 1

#### DFS
```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if(root == null) return true;
        return isSymmetric(root.left, root.right);
    }
    private boolean isSymmetric(TreeNode a, TreeNode b){
        if(a==null) return b==null;
        if(b==null) return false;
        if(a.val != b.val) return false;
        return isSymmetric(a.right, b.left) && isSymmetric(a.left, b.right);
    }
}
```

#### 2019.9.13 iterative
- time: 44.53%
- space: 63.26%
- cheatFlag
```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        Stack<TreeNode> t1 = new Stack<>(), t2 = new Stack<>();
        TreeNode p1 = root.left, p2 = root.right;
        while(p1 != null || !t1.isEmpty() || p2 != null || !t2.isEmpty()) {
            while(p1 != null && p2 != null) {
                t1.add(p1);
                p1 = p1.left;
                t2.add(p2);
                p2 = p2.right;
            }
            if (p1 != null || p2 != null) return false;
            p1 = t1.pop();
            p2 = t2.pop();
            if (p1.val != p2.val) return false;
            p1 = p1.right;
            p2 = p2.left;
        }
        return true;
    }
}
```

### 69. Sqrt(x)
- [Link](https://leetcode.com/problems/sqrtx/)
- Tags: Math, Binary Search
- Stars: 1

#### 2019.9.12
- time: 100%
- space: 5%
- attention: `Math.min(x, (int)Math.sqrt(Integer.MAX_VALUE))` can avoid overflow problems. Otherwise, we need to use `int mid = l + (long)r + 1 >> 1` and `mid > x/mid` to avoid overflows.
```java
class Solution {
    public int mySqrt(int x) {
        int l = 0, r = Math.min(x, (int)Math.sqrt(Integer.MAX_VALUE));
        while(l<r) {
            int mid = l+r+1 >> 1;
            if (mid*mid > x) r = mid-1;
            else if (mid*mid < x) l = mid;
            else return mid;
        }
        return l;
    }
}
```

Another version:
- time: 100%
- space: 5%
```java
class Solution {
    public int mySqrt(int x) {
        int l = 0, r = x;
        while(l<r) {
            int mid = (int)((long)l+r+1 >> 1);
            if (mid > x/mid) r = mid-1;
            else if (mid < x/mid) l = mid;
            else return mid;
        }
        return l;
    }
}
```

### 674. Longest Continuous Increasing Subsequence
- [Link](https://leetcode.com/problems/longest-continuous-increasing-subsequence/)
- Tags: Array
- Stars: 1

#### DP
```java
class Solution {
    public int findLengthOfLCIS(int[] nums) {
        if(nums.length == 0) return 0;
        int maxLen = 1, result = 1;
        for(int i=1; i<nums.length; i++){
            if(nums[i] > nums[i-1]) maxLen++;
            else {
                if(result < maxLen) result = maxLen;
                maxLen = 1;
            }
        }
        if(result < maxLen) result = maxLen;
        return result;
    }
}
```

### 673. Number of Longest Increasing Subsequence
- [Link](https://leetcode.com/problems/number-of-longest-increasing-subsequence/)
- Tags: Dynamic Programming
- Stars: 3

#### DP O(n^2)time
```java
class Solution {
    public int findNumberOfLIS(int[] nums) {
        if(nums.length == 0) return 0;
        int[] len = new int[nums.length]; // the longest length of Increasing Subsequence that ends with nums[i]
        int[] count = new int[nums.length];// the number of longest Increasing Subsequence that ends with nums[i]
        len[0] = 1;
        count[0] = 1;
        int maxLen = 1, result = 1;
        for(int i=1; i<nums.length; i++){
            len[i] = count[i] = 1;
            for(int j=0; j<i; j++){
                if(nums[i] > nums[j]){
                    if(len[i] == len[j]+1) count[i] += count[j];
                    else if(len[i] < len[j]+1) {
                        len[i] = len[j]+1;
                        count[i] = count[j];
                    }
                }
            }
            if(maxLen == len[i]) result += count[i];
            else if(maxLen < len[i]) {
                maxLen = len[i];
                result = count[i];
            }
        }
        return result;
    }
}
```

### 324. Wiggle Sort II
- [Link](https://leetcode.com/problems/wiggle-sort-ii/)
- Tags: Sort
- Stars: 5
- exploreFlag

#### 2019.9.1 O(nlogn) time sort 
- time: 100%
- space: 100%
- reference: https://leetcode.com/problems/wiggle-sort-ii/discuss/77678/3-lines-Python-with-Explanation-Proof
```java
class Solution {
    public void wiggleSort(int[] nums) {
        if(nums.length == 0) return ;
        int[] copy = nums.clone();
        Arrays.sort(copy);
        int l = (nums.length-1)/2, r = nums.length-1;
        int i=0;
        while(i<nums.length){
            nums[i++] = copy[l--];
            if(r==(nums.length-1)/2) break;
            nums[i++] = copy[r--];
        }
    }
}
```

#### 2019.9.1 virtual sort O(1) space
- time: 51.95%
- space: 100%
virtual indexing
```java
class Solution {
    int len, mid;
    public void wiggleSort(int[] nums) {
        len = nums.length;
        mid = (nums.length-1)/2;
        virtualSort(nums, 0, nums.length-1);
    }
    public void virtualSort(int[] nums, int vl, int vr) {
        if (vl >= vr) return;
        int vmid = partition(nums, vl, vr);
        virtualSort(nums, vl, vmid-1);
        virtualSort(nums, vmid+1, vr);
    }
    public int partition(int[] nums, int vl, int vr) {
        int vi = vl, vj = vr+1, pivot = get(nums, vl);
        while(true) {
            while(get(nums, ++vi) < pivot && vi < vr);
            while(pivot < get(nums, --vj) && vl < vj);
            if (vi >= vj) break;
            swap(nums, vi, vj);
        }
        swap(nums, vl, vj);
        return vj;
    }
    public void swap(int[] nums, int vi, int vj) {
        int i = virtual2real(vi), j = virtual2real(vj), temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    public int get(int[] nums, int vi) {
        return nums[virtual2real(vi)];
    }
    public int virtual2real(int i) {
        if (i <= mid) return 2*(mid-i);
        return 2*(len-1-i) + 1;
    }
    public int real2virtual(int i) {
        if (i%2 == 0) return mid - (i>>1);
        return len-1-((i-1)>>1);
    }
}
```

#### 2019.9.1 O(n) time + O(1) space after find median
- time: 51.95%
- space: 100%
- attention: Similar to the solution above, but notice that you don't need a complete sort. Actually, you only need to use "Quick Search partition" to put the median in the middle of virtual indices. Then, make sure all the elements that are equal to the median are placed adjacent to the median in the virtual world. With this, you get an array where elements with virtual index smaller than `mid` are smaller or equal to the median, and vise versa.
- reference: https://leetcode.com/problems/wiggle-sort-ii/discuss/77682/Step-by-step-explanation-of-index-mapping-in-Java
```java
class Solution {
    int len, mid;
    public void wiggleSort(int[] nums) {
        len = nums.length;
        mid = (nums.length-1)/2;
        findNthEle(nums, 0, len-1, mid);
        int median = get(nums, mid);
        for(int vi=mid-1, vl=mid-1; vi>=0; vi--) 
            if (get(nums, vi) == median) swap(nums, vi, vl--);
        for(int vi=mid+1, vr=mid+1; vi<len; vi++)
            if (get(nums, vi) == median) swap(nums, vi, vr++);
    }
    public void findNthEle(int[] nums, int vl, int vr, int vmid) {
        if (vl >= vr) return;
        while(true) {
            int vj = partition(nums, vl, vr);
            if (vj > vmid) vr = vj-1;
            else if (vj < vmid) vl = vj+1;
            else return;
        }
    }
    public int partition(int[] nums, int vl, int vr) {
        int vi = vl, vj = vr+1, pivot = get(nums, vl);
        while(true) {
            while(get(nums, ++vi) < pivot && vi < vr);
            while(pivot < get(nums, --vj) && vl < vj);
            if (vi >= vj) break;
            swap(nums, vi, vj);
        }
        swap(nums, vl, vj);
        return vj;
    }
    public void swap(int[] nums, int vi, int vj) {
        int i = virtual2real(vi), j = virtual2real(vj), temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    public int get(int[] nums, int vi) {
        return nums[virtual2real(vi)];
    }
    public int virtual2real(int i) {
        if (i <= mid) return 2*(mid-i);
        return 2*(len-1-i) + 1;
    }
    public int real2virtual(int i) {
        if (i%2 == 0) return mid - (i>>1);
        return len-1-((i-1)>>1);
    }
}
```

### 42. Trapping Rain Water
- [Link](https://leetcode.com/problems/trapping-rain-water/)
- Tags: Array, Two Pointers, Stack
- Stars: 3

#### My Solution, O(n) time, O(1) space
1. Iterate `height` from left to right: each iteration, check if height[i] is the highest height (higher than `currHeight`). If true, count the volume of water between `currIdx`(i.e. the index of the currHeight position) and `i`. 
2. The iterations above fail to account for water after the last highest height. Therefore, do the same thing from right to left again. 
```java
class Solution {
    public int trap(int[] height) {
        if(height.length < 2) return 0;
        int currHeight = 0, currIdx = 0, result = 0;
        // Iterate from left to right
        for(int i=0; i<height.length; i++){
            if(height[i] >= currHeight){
                for(int j=currIdx+1; j<i; j++)
                    result += currHeight-height[j];
                currHeight = height[i];
                currIdx = i;
            }
        }
        int midIdx = currIdx, midHeight=currHeight;
        currHeight = 0; currIdx = height.length-1;
        // Iterate from right to left
        for(int i=height.length-1; i>=midIdx; i--){
            if(height[i] >= currHeight){
                for(int j=i+1; j<currIdx; j++)
                    result += currHeight-height[j];
                currHeight = height[i];
                currIdx = i;
            }
            if(height[i] == midHeight) break;
        }
        return result;
    }
}
```

#### one-pass solution, O(n) time, O(1) space
Two pointers!  
If height[i] <= height[j], there are only two situations:  
1) height[i] is the highest in subarray from 0 to i. For this situation, the `result` remains unchanged.  
2) height[i] is not the highest from left, there must be water above it, and the height of water above height[i] should be `maxLeftHeight - height[i]`. 
```java
class Solution {
    public int trap(int[] height) {
        if(height.length < 2) return 0;
        int i=0, j=height.length-1;
        int result = 0, maxLeftHeight = 0, maxRightHeight = 0;
        while(i<=j){
            if(height[i] <= height[j]){
                if(maxLeftHeight < height[i]) maxLeftHeight = height[i];
                result += maxLeftHeight-height[i];
                i++;
            }
            else {
                if(maxRightHeight < height[j]) maxRightHeight = height[j];
                result += maxRightHeight-height[j];
                j--;
            }
        }
        return result;
    }
}
```

Updated 2019.8.17
- time: 98.33%
- space: 97.26%

```java
class Solution {
    public int trap(int[] height) {
        if (height.length == 0) return 0;
        int result = 0;
        int i=0, j=height.length - 1, left = height[i], right = height[j];
        while(i<j) {
            if (left <= right) {
                result += left - height[i++];
                left = Math.max(left, height[i]);
            } else {
                result += right - height[j--];
                right = Math.max(right, height[j]);
            }
        }
        return result;
    }
}
```

### 128. Longest Consecutive Sequence
- [Link](https://leetcode.com/problems/longest-consecutive-sequence/)
- Tags: Array, Union Find
- Stars: 4

#### union find based on HashMap, only beats 27.49% in time and 34.14% in space
```java
class Solution {
    public int longestConsecutive(int[] nums) {
        if(nums.length == 0) return 0;
        HashMap<Integer, Integer> uf = new HashMap<>();
        HashMap<Integer, Integer> lens = new HashMap<>();
        for(int num: nums){
            if(uf.containsKey(num)) continue;
            uf.put(num, num);
            lens.put(num, 1);
            if(uf.get(num-1)!=null) union(uf, lens, num, num-1);
            if(uf.get(num+1)!=null) union(uf, lens, num, num+1);
        }
        int result = 0;
        for(int num: nums)
            if(uf.get(num) == num && result < lens.get(num)) result = lens.get(num);
        return result;
    }
    private void union(HashMap<Integer, Integer> uf, HashMap<Integer, Integer> lens,
                       int a, int b){
        int roota = find(uf, a), rootb = find(uf, b);
        if(roota==rootb) return ;
        if(lens.get(roota) <= lens.get(rootb)){
            uf.put(roota, rootb);
            lens.put(rootb, lens.get(roota)+lens.get(rootb));
        }
        else {
            uf.put(rootb, roota);
            lens.put(roota, lens.get(roota)+lens.get(rootb));
        }
    }
    private int find(HashMap<Integer, Integer> uf, int a){
        while(uf.get(a) != a){
            int b = uf.get(a);
            uf.put(a, uf.get(b));
            a = b;
        }
        return a;
    }
}
```

#### only the boundaries, real O(n) time, beats 90.75% in time
- attention: `if(map.containsKey(num)) continue;` is necessary to avoid duplicate counting. For the same reason, we also need to update `map.put(num, leftLen+rightLen+1);`.

The tricky part was to understand why only the boundaries need to be updated and not the entire sequence with the new sum.
```java
class Solution {
    public int longestConsecutive(int[] nums) {
        if(nums.length == 0) return 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        int result = 0;
        for(int num: nums){
            if(map.containsKey(num)) continue;
            // get left and right sequence length
            int leftLen = map.getOrDefault(num-1, 0);
            int rightLen = map.getOrDefault(num+1, 0);
            map.put(num, leftLen+rightLen+1);
            // update the return value `result`
            result = Math.max(result, leftLen+rightLen+1);
            // we only need to udpate the ends of a sequence
            if(leftLen>0) map.put(num-leftLen, leftLen+rightLen+1);
            if(rightLen>0) map.put(num+rightLen, leftLen+rightLen+1);
        }
        return result;
    }
}
```

#### GENIUS!!
```java
class Solution {
    public int longestConsecutive(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for(int num : nums) set.add(num);
        int result = 0;
        for(int num: set){
            if(set.contains(num-1)) continue;
            int idx = num+1, count = 1;
            while(set.contains(idx)){
                count++;
                idx++;
            }
            if(result < count) result = count;
        }
        return result;
    }
}
```

### 329. Longest Increasing Path in a Matrix
- [Link](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/)
- Tags: DFS, Topological Sort, Memoization
- Stars: 3

#### My solution, DFS+Memoization, only beats 18% in time
1. borrowed idea from 128#GENIUS!! where only starts from the smaller side to avoid bi-direction checks. That's why I use `isMin` instead of `isMinMax`. `isMin` is a function to check whether an element in `matrix` is the minimum compared to its surrounding neighbors. 
2. Use DFS to get the longest increasing path starting from a local minimum. 
3. Need to record all the intermediate outcomes to avoid waste of computation. (Memoization) 
```java
class Solution {
    int[][] record;
    public int longestIncreasingPath(int[][] matrix) {
        if(matrix.length == 0 || matrix[0].length == 0) return 0;
        record = new int[matrix.length][matrix[0].length];
        boolean[][] mark = new boolean[matrix.length][matrix[0].length];
        int result = 0;
        for(int i=0; i<matrix.length; i++)
            for(int j=0; j<matrix[0].length; j++){
                if(!isMin(matrix, i, j)) continue;
                int temp = DFS(matrix, mark, i, j);
                if(result < temp) result = temp;
            }
        return result;
    }
    private int DFS(int[][] matrix, boolean[][] mark, int i, int j){
        if(i<0 || j<0 || i>= matrix.length || j>= matrix[0].length) return 0;
        if(record[i][j] > 0) return record[i][j];
        mark[i][j] = true;
        int result = 1;
        if(i-1>=0 && matrix[i-1][j] > matrix[i][j]) 
            result = Math.max(result, 1+DFS(matrix, mark, i-1, j));
        if(i+1<matrix.length && matrix[i+1][j] > matrix[i][j]) 
            result = Math.max(result, 1+DFS(matrix, mark, i+1, j));
        if(j-1>=0 && matrix[i][j-1] > matrix[i][j]) 
            result = Math.max(result, 1+DFS(matrix, mark, i, j-1));
        if(j+1<matrix[0].length && matrix[i][j+1] > matrix[i][j]) 
            result = Math.max(result, 1+DFS(matrix, mark, i, j+1));
        mark[i][j] = false;
        record[i][j] = result;
        return result;
    }
    private boolean isMin(int[][] matrix, int i, int j){
        boolean result = true;
        if(i-1>=0 && matrix[i-1][j] < matrix[i][j]) return false;
        if(i+1<matrix.length && matrix[i+1][j] < matrix[i][j]) return false;
        if(j-1>=0 && matrix[i][j-1] < matrix[i][j]) return false;
        if(j+1<matrix[0].length && matrix[i][j+1] < matrix[i][j]) return false;
        return true;
    }
}
```

#### optimized DFS, beats 98% in time
1. compared with the first solution, I find that don't actually need `isMin` method. 
2. I can use a static final `directions` to indicate all the possible four directions, instead of hard coding the four directions like the solution above. 
3. the `mark` boolean array can be discarded, as the `record` has already contains the information. 
```java
class Solution {
    public int[][] record;
    public final static int[][] directions = {{0,1},{0,-1},{1,0},{-1,0}};
    public int longestIncreasingPath(int[][] matrix) {
        if(matrix.length == 0 || matrix[0].length == 0) return 0;
        record = new int[matrix.length][matrix[0].length];
        int result = 0;
        for(int i=0; i<matrix.length; i++)
            for(int j=0; j<matrix[0].length; j++){
                int temp = DFS(matrix, i, j);
                if(result < temp) result = temp;
            }
        return result;
    }
    private int DFS(final int[][] matrix, final int i, final int j){
        // if(i<0 || j<0 || i>= matrix.length || j>= matrix[0].length) return 0;
        if(record[i][j] != 0) return record[i][j];
        int result = 1;
        for(int[] direction : directions){
            int x = i+direction[0], y = j+direction[1];
            if(x>=0 && y>=0 && x<matrix.length && y<matrix[0].length 
               && matrix[x][y] > matrix[i][j])
                result = Math.max(result, 1+DFS(matrix, x, y));
        }
        record[i][j] = result;
        return result;
    }
}
```

#### Topological Sort
https://leetcode.com/problems/longest-increasing-path-in-a-matrix/discuss/78336/Graph-theory-Java-solution-O(v2)-no-DFS

### 315. Count of Smaller Numbers After Self
- [Link](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)
- Tags: Divide and Conquer, Binary Indexed Tree, Segment Tree, Binary Search Tree
- Stars: 3

#### lower bound binary search insertion, O(nlogn) time and O(n) space. 
insert the elements of `nums` into `order` one by one from tail to head. 
```java
class Solution {
    public List<Integer> countSmaller(int[] nums) {
        List<Integer> result = new ArrayList<>();
        if(nums.length == 0) return result;
        List<Integer> order = new ArrayList<>();
        result.add(0);
        order.add(nums[nums.length-1]);
        for(int i=nums.length-2; i>=0; i--){
            int idx = binarySearch(order, nums[i]);
            if(idx<0) idx = -(idx+1);
            result.add(0, idx);
            order.add(idx, nums[i]);
        }
        return result;
    }
    private int binarySearch(List<Integer> list, int target){
        int l=0, r=list.size()-1;
        while(l+1 < r){
            int mid = l+((r-l)>>1);
            if(list.get(mid) >= target) r = mid;
            else l = mid;
        }
        if(list.get(l) >= target) return l;
        if(list.get(r) >= target) return r;
        return r+1;
    }
}
```

### 239. Sliding Window Maximum
- [Link](https://leetcode.com/problems/sliding-window-maximum/)
- Tags: Heap, Sliding Window
- Stars: 4

#### My solution, MaxQueue
MaxQueue is implemented by two MaxStack.
```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums.length == 0) return new int[0];
        MaxQueue qu = new MaxQueue();
        int[] result = new int[nums.length-k+1];
        int p = 0;
        for(int i=0; i<nums.length; i++){
            qu.add(nums[i]);
            if(qu.size() == k){
                result[p++] = qu.getMax();
                qu.pop();
            }
        }
        return result;
    }
}
class MaxQueue {
    public MaxStack st1 = new MaxStack(), st2 = new MaxStack();
    public void add(int x){
        st1.add(x);
    }
    public int size(){
        return st1.size() + st2.size();
    }
    public int pop(){
        if(st2.isEmpty()){
            assert !st1.isEmpty();
            while(!st1.isEmpty()){
                st2.add(st1.pop());
            }
        }
        return st2.pop();
    }
    public int getMax(){
        return Math.max(st1.getMax(), st2.getMax());
    }
}
class MaxStack {
    public Stack<Integer> st = new Stack(), maxSt = new Stack();
    public int maxVal = Integer.MIN_VALUE;
    public void add(int x){
        if(maxVal < x) maxVal = x;
        maxSt.add(maxVal);
        st.add(x);
    }
    public int size(){
        return st.size();
    }
    public boolean isEmpty(){
        return st.isEmpty();
    }
    public int pop(){
        maxSt.pop();
        maxVal = maxSt.size() == 0 ? Integer.MIN_VALUE : maxSt.peek();
        return st.pop();
    }
    public int getMax(){
        return maxVal;
    }
}
```

#### 2019.8.18 TreeMap O(nlogn)
- time: 12.40%
- space: 90.63%

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums.length == 0) return new int[0];
        int[] result = new int[nums.length - k + 1];
        TreeMap<Integer, Integer> map = new TreeMap<>();
        for(int i=0, j=0; i<nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
            if (i >= k-1) {
                result[j++] = map.lastKey();
                map.put(nums[i-k+1], map.get(nums[i-k+1]) - 1);
                if (map.get(nums[i-k+1]) == 0) map.remove(nums[i-k+1]);
            }
        }
        return result;
    }
}
```

### 295. Find Median from Data Stream
- [Link](https://leetcode.com/problems/find-median-from-data-stream/)
- Tags: Heap, Design
- Stars: 2

#### lower bound binary search insertion
- language: `list.add(idx, num)`, `list.addFirst(num)`

```java
class MedianFinder {
    List<Integer> list;
    public MedianFinder() {
        list = new ArrayList<>();
    }
    public void addNum(int num) {
        int idx = binarySearch(list, num);
        list.add(idx, num);
    }
    public double findMedian() {
        if(list.size()%2 == 1) return (double)list.get(list.size()/2);
        return (list.get(list.size()/2)+list.get(list.size()/2-1))/2.0;
    }
    private int binarySearch(List<Integer> list, int target){
        if(list.size() == 0) return 0;
        int l=0, r=list.size()-1;
        while(l<r){
            int mid = l + ((r-l)>>1);
            if(list.get(mid) >= target) r = mid-1;
            else l = mid + 1;
        }
        if(list.get(l) >=target) return l;
        return l+1;
    }
}
```

#### minHeap and maxHeap
- language: `Comparator.reverseOrder()`

```java
class MedianFinder {
    PriorityQueue<Integer> maxHeap;
    PriorityQueue<Integer> minHeap;
    public MedianFinder() {
        maxHeap = new PriorityQueue<>(Comparator.reverseOrder());
        minHeap = new PriorityQueue<>();
    }
    public void addNum(int num) {
        if(maxHeap.isEmpty() || maxHeap.peek() >= num) maxHeap.add(num);
        else minHeap.add(num);
        while(maxHeap.size() < minHeap.size())
            maxHeap.add(minHeap.poll());
        while(maxHeap.size()-1 > minHeap.size())
            minHeap.add(maxHeap.poll());
    }
    public double findMedian() {
        if(maxHeap.size() == minHeap.size()) return (maxHeap.peek()+minHeap.peek())/2.0;
        return (double)maxHeap.peek();
    }
}
```

### 23. Merge k Sorted Lists
- [Link](https://leetcode.com/problems/merge-k-sorted-lists/)
- Tags: Linked List, Divide and Conquer, Heap
- Stars: 3

#### minHeap, 41ms
```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists.length == 0) return null;
        ListNode head = new ListNode(0), curr = head;
        PriorityQueue<Tuple> qu = new PriorityQueue<>(
            (tup1, tup2) -> (tup1.node.val-tup2.node.val));
        for(int i=0; i<lists.length; i++){
            if(lists[i] != null){
                qu.add(new Tuple(i, lists[i]));
                lists[i] = lists[i].next;
            }
        }
        while(!qu.isEmpty()){
            Tuple tup = qu.poll();
            curr.next = tup.node;
            curr = curr.next;
            if(lists[tup.idx] != null){
                qu.add(new Tuple(tup.idx, lists[tup.idx]));
                lists[tup.idx] = lists[tup.idx].next;
            }
        }
        return head.next;
    }
}
class Tuple {
    int idx;
    ListNode node;
    public Tuple(int i, ListNode n){
        idx = i;
        node = n;
    }
}
```

#### minHeap, simplified version, 9ms
```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists.length == 0) return null;
        ListNode head = new ListNode(0), curr = head;
        PriorityQueue<ListNode> qu = new PriorityQueue<>(new Comparator<ListNode>(){
            public int compare(ListNode a, ListNode b){
                return a.val - b.val;
            }
        });
        for(ListNode node : lists)
            if(node != null) qu.add(node);
        while(!qu.isEmpty()){
            curr.next = qu.poll();
            curr = curr.next;
            if(curr.next != null) qu.add(curr.next);
        }
        return head.next;
    }
}
```

#### divide and conquer, use merge 2 linkedlist, 5ms beats 100% in time
- interviewLevel

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists.length == 0) return null;
        return mergeKLists(lists, 0, lists.length-1);
    }
    public ListNode mergeKLists(ListNode[] lists, int start, int end){
        if(start == end) return lists[start];
        int mid = start + ((end-start)>>1);
        ListNode l1 = mergeKLists(lists, start, mid);
        ListNode l2 = mergeKLists(lists, mid+1, end);
        return merge(l1, l2);
    }
    private ListNode merge(ListNode l1, ListNode l2){
        if(l1 == null) return l2;
        if(l2 == null) return l1;
        if(l1.val < l2.val){
            l1.next = merge(l1.next, l2);
            return l1;
        }
        else {
            l2.next = merge(l1, l2.next);
            return l2;
        }
    }
}
```

### 124. Binary Tree Maximum Path Sum
- [Link](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
- Tags: Tree, DFS
- Stars: 2

#### DFS, (tree-like) maximum subarray
```java
class Solution {
    int result = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        if(root == null) return 0;
        DFS(root);
        return result;
    }
    private int DFS(TreeNode root){
        if(root == null) return 0;
        int left = Math.max(0, DFS(root.left)), right = Math.max(0, DFS(root.right));
        result = Math.max(result, left+right+root.val);
        return Math.max(left, right)+root.val;
    }
}
```

### 41. First Missing Positive
- [Link](https://leetcode.com/problems/first-missing-positive/)
- Tags: Array
- Stars: 3

#### establish val2index and index2val mapping, O(n) time and O(1) space
1. ignore all elements that are <= 0 or > nums.length
2. establish val2index and index2val mapping for the remaining elements. i.e. `val2index[i] = i-1` and `index2val[i] = i+1`. 
3. iterate `nums`. For each num, swap this num to the index that it is supposed to be. 
4. the condition `nums[nums[i]-1] != nums[i]` in the while loop is used to avoid infinite loop caused by duplicates. 
```java
class Solution {
    public int firstMissingPositive(int[] nums) {
        if(nums.length == 0) return 1;
        for(int i=0; i<nums.length; i++){
            while(nums[i]>0 && nums[i] <= nums.length && 
                  nums[i] != i+1 && nums[nums[i]-1] != nums[i])
                swap(nums, nums[i]-1, i);
        }
        for(int i=0; i<nums.length; i++)
            if(nums[i] != i+1) return i+1;
        return nums.length+1;
    }
    private void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

#### O(n) time and O(n) space
We only need to mark numbers in [1, nums.length]. 
```java
class Solution {
    public int firstMissingPositive(int[] nums) {
        if(nums.length == 0) return 1;
        boolean[] mark = new boolean[nums.length];
        for(int num: nums)
            if(num>0 && num<=nums.length) mark[num-1] = true;
        for(int i=0; i<mark.length; i++)
            if(!mark[i]) return i+1;
        return nums.length+1;
    }
}
```

### 4. Median of Two Sorted Arrays
- [Link](https://leetcode.com/problems/median-of-two-sorted-arrays/)
- Tags: Array, Binary Search, Divide and Conquer
- Stars: 3

#### binary search
To avoid boundary check, use 
```
l1 = mid1 == 0 ? Integer.MIN_VALUE : nums1[mid1-1];
r1 = mid1 == nums1.length ? Integer.MAX_VALUE : nums1[mid1];
l2 = mid2 == 0 ? Integer.MIN_VALUE : nums2[mid2-1];
r2 = mid2 == nums2.length ? Integer.MAX_VALUE : nums2[mid2];
```
.

```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        // make sure nums1.length >= nums2.length
        if(nums1.length < nums2.length) return findMedianSortedArrays(nums2, nums1);
        int l = 0, r = nums1.length;
        // mid: the index to divide the nums into two parts. range [0, nums.length], (nums.length+1) possible values in total
        // l1: the value just before mid1 
        // r1: the value just after mid1 
        int mid1 = 0, mid2 = 0, l1=0, r1=0, l2=0,  r2=0;
        while(l<r){
            mid1 = l + ((r-l)>>1);
            mid2 = (nums1.length+nums2.length+1)/2 - mid1;
            if(mid2 > nums2.length) {
                l = mid1+1;
                continue;
            }
            else if(mid2 < 0){
                r = mid1-1;
                continue;
            }
            l1 = mid1 == 0 ? Integer.MIN_VALUE : nums1[mid1-1];
            r1 = mid1 == nums1.length ? Integer.MAX_VALUE : nums1[mid1];
            l2 = mid2 == 0 ? Integer.MIN_VALUE : nums2[mid2-1];
            r2 = mid2 == nums2.length ? Integer.MAX_VALUE : nums2[mid2];
            if(l1 <= r2 && l2<=r1) break;
            if(l1 > r2) r = mid1 - 1;
            if(l2 > r1) l = mid1 + 1;
        }
        mid1 = l + ((r-l)>>1);
        mid2 = (nums1.length+nums2.length+1)/2 - mid1;
        l1 = mid1 == 0 ? Integer.MIN_VALUE : nums1[mid1-1];
        r1 = mid1 == nums1.length ? Integer.MAX_VALUE : nums1[mid1];
        l2 = mid2 == 0 ? Integer.MIN_VALUE : nums2[mid2-1];
        r2 = mid2 == nums2.length ? Integer.MAX_VALUE : nums2[mid2];
        if((nums1.length+nums2.length)%2==1) return Math.max(l1, l2);
        return (Math.max(l1, l2) + Math.min(r1, r2))/2.0;
    }
}
```

## Top 100 Liked Questions

### 461. Hamming Distance
- [Link](https://leetcode.com/problems/hamming-distance/)
- Tags: Bit Manipulation
- Stars: 3

#### Java Built-in Function
```java
class Solution {
    public int hammingDistance(int x, int y) {
        return Integer.bitCount(x ^ y);
    }
}
```

#### bit counting by groups
1. `x = (x&0x55555555) + ((x>>>1)&0x55555555)` can also be written as `x = x - ((x >>> 1) & 0x55555555)`. 
2. The following solution can be further simplified to `i = (i + (i >>> 4)) & 0x0f0f0f0f; i = i + (i >>> 8); i = i + (i >>> 16); return i & 0x3f;`
```java
class Solution {
    public int hammingDistance(int x, int y) {
        x ^= y;
        x = (x&0x55555555) + ((x>>>1)&0x55555555);
        x = (x&0x33333333) + ((x>>>2)&0x33333333);
        x = (x&0x0f0f0f0f) + ((x>>>4)&0x0f0f0f0f);
        x = (x&0x00ff00ff) + ((x>>>8)&0x00ff00ff);
        x = (x&0x0000ffff) + ((x>>>16)&0x0000ffff);
        return x;
    }
}
```

### 448. Find All Numbers Disappeared in an Array
- [Link](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)
- Tags: Array
- Stars: 2

#### O(n) time and O(1) space
Given an element, we can calculate the expected index. 
```java
class Solution {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> result = new ArrayList<>();
        for(int i=0; i<nums.length; i++){
            while(i != nums[i]-1){
                if(nums[nums[i]-1] == nums[i]) break;
                swap(nums, i, nums[i]-1);
            }
        }
        for(int i=0; i<nums.length; i++)
            if(i != nums[i]-1) 
                result.add(i+1);
        return result;
    }
    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

#### in-place marking, another O(n) time and O(1) space
```java
class Solution {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> result = new ArrayList<>();
        if(nums.length == 0) return result;
        // mark the existing numbers: use the existing numbers as index and set it to its opposite number
        for(int i=0; i<nums.length; i++){
            int num = Math.abs(nums[i])-1;
            if(nums[num] > 0) nums[num] = -nums[num];
        }
        // iterate through nums array and find all still-positive numbers
        for(int i=0; i<nums.length; i++)
            if(nums[i] > 0) result.add(i+1);
        return result;
    }
}
```

### 538. Convert BST to Greater Tree
- [Link](https://leetcode.com/problems/convert-bst-to-greater-tree/)
- Tags: Tree
- Stars: 1

#### recursive
```java
class Solution {
    int accu = 0;
    public TreeNode convertBST(TreeNode root) {
        if(root == null) return null;
        convertBST(root.right);
        accu += root.val;
        root.val = accu;
        convertBST(root.left);
        return root;
    }
}
```

### 543. Diameter of Binary Tree
- [Link](https://leetcode.com/problems/diameter-of-binary-tree/)
- Tags: Tree
- Stars: 1

#### recursive
1. Found that the two ends of the longest path must be leaf nodes, unless one of the leaf nodes is root. 
2. Given two leaf nodes, the path between them contains their highest-level common parent. 
```java
class Solution {
    int maxLen = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        if(root == null) return 0;
        DFTraversal(root);
        return maxLen-1;
    }
    private int DFTraversal(TreeNode root) {
        if(root == null) return 0;
        int leftLen = DFTraversal(root.left), rightLen = DFTraversal(root.right);
        int len = 1 + leftLen + rightLen;
        if(maxLen < len) maxLen = len;
        return 1 + Math.max(leftLen, rightLen);
    }
}
```

Updated 2019.9.13
- time: 100%
- space: 24.68%
- reviewFlag
- attention: The return value of dfs is the number of nodes along the deepest path from `root` to the leaf node.
```java
class Solution {
    int ret = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        dfs(root);
        return ret;
    }
    public int dfs(TreeNode root) {
        if (root == null) return 0;
        int left = dfs(root.left), right = dfs(root.right);
        ret = Math.max(ret, left + right);
        return Math.max(left, right) + 1;
    }
}
```

### 437. Path Sum III
- [Link](https://leetcode.com/problems/path-sum-iii/)
- Tags: Tree
- Stars: 3

#### backtracking
```java
class Solution {
    int result = 0;
    public int pathSum(TreeNode root, int sum) {
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        backtrack(root, map, 0, sum);
        return result;
    }
    private void backtrack(TreeNode root, HashMap<Integer, Integer> map, int curr, int sum){
        if(root == null) return ;
        curr += root.val;
        result += map.getOrDefault(curr-sum, 0);
        map.put(curr, map.getOrDefault(curr, 0)+1);
        backtrack(root.left, map, curr, sum);
        backtrack(root.right, map, curr, sum);
        map.put(curr, map.get(curr)-1);
    }
}
```

### 572. Subtree of Another Tree
- [Link](https://leetcode.com/problems/subtree-of-another-tree/)
- Tags: Tree
- Stars: 2

#### recursive
- attention: `isSubtree` and `isEqual` are different DFS process. Do not try to integrate into a single function.

```java
class Solution {
    public boolean isSubtree(TreeNode s, TreeNode t) {
        if(s == null) return false;
        if(isEqual(s, t)) return true;
        return isSubtree(s.left, t) || isSubtree(s.right, t);
    }
    private boolean isEqual(TreeNode a, TreeNode b){
        if(a == null || b==null){
            if(a==null && b==null) return true;
            return false;
        }
        if(a.val != b.val) return false;
        return isEqual(a.left, b.left) && isEqual(a.right, b.right);
    }
}
```

### 96. Unique Binary Search Trees
- [Link](https://leetcode.com/problems/unique-binary-search-trees/)
- Tags: Dynamic Programming, Tree
- Stars: 2

#### DP
An additional node n only has two directions available for connection:
```
(subtree a)
        \
         (node n)
        /
(subtree b)
```
Subtree a and b together contains all the nodes from 1 to n-1, and all the nodes in subtree a must be smaller than any node of subtree b. 
For each possible combination of subtree a and b, the additional node n brings about `f(#nodes of a) * f(#nodes of b)` additional unique BSTs. 
Therefore, the DP formula is `f(n) = sum([f(i) * f(n-1-i) for i in range(0, n)])`

```java
class Solution {
    public int numTrees(int n) {
        int[] dp = new int[n+1];
        dp[0] = 1;
        for(int i=0; i<n; i++) {
            for(int j=0; j<=i; j++) {
                dp[i+1] += dp[j] * dp[i-j];
            }
        }
        return dp[n];
    }
}
```

### 438. Find All Anagrams in a String
- [Link](https://leetcode.com/problems/find-all-anagrams-in-a-string/)
- Tags: Hash Table
- Stars: 1

#### sliding window
```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> result = new ArrayList<>();
        if(s.length() < p.length()) return result;
        int[] pattern = getPattern(p, 0, p.length());
        int[] stat = getPattern(s, 0, p.length()-1);
        int i=0;
        while(true){
            if(i + p.length() - 1 == s.length()) break;
            stat[s.charAt(i + p.length()-1)-'a']++;
            if(isSame(stat, pattern)) result.add(i);
            stat[s.charAt(i)-'a']--;
            i++;
        }
        return result;
    }
    private int[] getPattern(String s, int start, int end){
        int[] pattern = new int[26];
        for(int i=start; i<end; i++)
            pattern[s.charAt(i)-'a']++;
        return pattern;
    }
    private boolean isSame(int[] a, int[] b){
        for(int i=0; i<26; i++)
            if(a[i] != b[i]) return false;
        return true;
    }
}
```

### 20. Valid Parentheses
- [Link](https://leetcode.com/problems/valid-parentheses/)
- Tags: String, Stack
- Stars: 1

#### stack
```java
class Solution {
    public boolean isValid(String s) {
        if(s.length() == 0) return true;
        Stack<Character> st = new Stack<>();
        for(int i=0; i<s.length(); i++){
            if(s.charAt(i) == '(') st.add(')');
            else if(s.charAt(i) == '[') st.add(']');
            else if(s.charAt(i) == '{') st.add('}');
            else {
                if(st.isEmpty() || (char)(st.pop()) != s.charAt(i)) return false;
            }
        }
        return st.size() == 0;
    }
}
```

### 581. Shortest Unsorted Continuous Subarray
- [Link](https://leetcode.com/problems/shortest-unsorted-continuous-subarray/)
- Tags: Array
- Stars: 3

#### my solution, sort and compare, O(nlogn) time and O(n) space, suboptimal
```java
class Solution {
    public int findUnsortedSubarray(int[] nums) {
        if(nums.length == 0) return 0;
        int[] copy = Arrays.copyOfRange(nums, 0, nums.length);
        Arrays.sort(copy);
        int i=0, j=nums.length-1;
        while(i<j && nums[i] == copy[i]) i++;
        while(i<j && nums[j] == copy[j]) j--;
        if(i == j) return 0;
        return j-i+1;
    }
}
```

#### my solution, binary search
```java
class Solution {
    public int findUnsortedSubarray(int[] nums) {
        if(nums.length == 0) return 0;
        int i=0, j=nums.length-1;
        // skip the ordered subarray started from head. 
        while(i<j && nums[i] <= nums[i+1]) i++;
        // go left to ensure all duplicates of nums[i] has an index >= i. 
        if(i == j) return 0;
        else while(i>0 && nums[i-1] == nums[i]) i--;
        // skip the ordered subarray ended with tail. 
        while(i<j && nums[j-1] <= nums[j]) j--;
        // go right to ensure all duplicates of nums[j] has an index <= j. 
        if(i == j) return 0;
        else while(j<nums.length-1 && nums[j+1] == nums[j]) j++;
        // get the minVal and maxVal of subarray between i and j
        int minVal = nums[i], maxVal = nums[j];
        for(int k=i; k<=j; k++){
            if(minVal > nums[k]) minVal = nums[k];
            if(maxVal < nums[k]) maxVal = nums[k];
        }
        // find idx s.t. all elements in nums[0:idx] are < minVal
        // notice we set target of binary search to minVal+1 instead of minVal
        int minIdx = Arrays.binarySearch(nums, 0, i, minVal+1);
        if(minIdx < 0) minIdx = -(minIdx+1);
        // find idx s.t. all elements in nums[idx:] are >= maxVal
        int maxIdx = Arrays.binarySearch(nums, j+1, nums.length, maxVal);
        if(maxIdx < 0) maxIdx = -(maxIdx+1);
        return maxIdx - minIdx;
    }
}
```

#### GENIUS!!
```java
class Solution {
    public int findUnsortedSubarray(int[] nums) {
        if(nums.length == 0) return 0;
        int start=-1, end=-2, currMax = nums[0], currMin = nums[nums.length-1];
        for(int i=0; i<nums.length; i++){
            currMax = Math.max(currMax, nums[i]);
            if(currMax > nums[i]) end = i;
            currMin = Math.min(currMin, nums[nums.length-i-1]);
            if(currMin < nums[nums.length-i-1]) start = nums.length-i-1;
        }
        return end-start+1;
    }
}
```

## Others

### 129. Sum Root to Leaf Numbers
- [Link](https://leetcode.com/problems/sum-root-to-leaf-numbers/)
- Tags: Tree, DFS
- Stars: 1

#### DFS
```java
class Solution {
    int result = 0;
    public int sumNumbers(TreeNode root) {
        if(root == null) return 0;
        DFS(root, 0);
        return result;
    }
    public void DFS(TreeNode root, int curr){
        curr *= 10;
        curr += root.val;
        if(root.left == null && root.right == null){
            result += curr;
            return ;
        }
        if(root.left != null) DFS(root.left, curr);
        if(root.right != null) DFS(root.right, curr);
    }
}
```

## First 300 Questions

### 270. Closest Binary Search Tree Value
- [Link](https://leetcode.com/problems/closest-binary-search-tree-value/)
- Tags: Binary Search, Tree
- Stars: 2

#### 2019.9.11 iterative
- time: 100%
- space: 100%
```java
class Solution {
    public int closestValue(TreeNode root, double target) {
        int ret = root.val;
        double diff = Math.abs(root.val - target);
        while(root != null) {
            if (diff == 0) break;
            double curr = Math.abs(root.val-target);
            if (curr < diff) {
                ret = root.val;
                diff = curr;
            } 
            if (target < root.val) root = root.left;
            else root = root.right;
        }
        return ret;
    }
}
```

#### 2019.9.11 recursive
- time: 100%
- space: 97.44%
```java
class Solution {
    public int closestValue(TreeNode root, double target) {
        if (root.val == target) return root.val;
        if (target > root.val) {
            if (root.right == null) return root.val;
            int right = closestValue(root.right, target);
            if (Math.abs(right-target) < Math.abs(root.val-target)) return right;
            return root.val;
        }
        if (root.left == null) return root.val;
        int left = closestValue(root.left, target);
        if (Math.abs(left-target) < Math.abs(root.val-target)) return left;
        return root.val;
    }
}
```

### 256. Paint House
- [Link](https://leetcode.com/problems/paint-house/)
- Tags: Dynamic Programming
- Stars: 2

#### 2019.9.11 DP state machine
- time: 100%
- space: 5.88%
```java
class Solution {
    public int minCost(int[][] costs) {
        if (costs.length == 0) return 0;
        int[] dp = costs[0];
        for(int i=1; i<costs.length; i++) {
            int red = Math.min(dp[1], dp[2]) + costs[i][0],
                blue = Math.min(dp[0], dp[2]) + costs[i][1],
                green = Math.min(dp[0], dp[1]) + costs[i][2];
            dp[0] = red;
            dp[1] = blue;
            dp[2] = green;
        }
        return Math.min(dp[0], Math.min(dp[1], dp[2]));
    }
}
```

### 82. Remove Duplicates from Sorted List II
- [Link](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/)
- Tags: Linked List
- Stars: 3
- reviewFlag

#### 2019.9.6
- time: 75.73%
- space: 100%
- attention: Always remember to set `tail.next = null` when you use a `tail` point in LinkedList problems
```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        ListNode dmy = new ListNode(0), tail = dmy, cur = head;
        while(cur != null) {
            if (cur.next == null) {
                tail.next = cur;
                break;
            } else if (cur.val == cur.next.val) {
                int target = cur.val;
                while(cur != null && cur.val == target) {
                    cur = cur.next;
                }
            } else {
                tail.next = cur;
                tail = tail.next;
                cur = cur.next;
                tail.next = null;
            }
        }
        return dmy.next;
    }
}
```

### 148. Sort List
- [Link](https://leetcode.com/problems/sort-list/)
- Tags: Linked List, Sort
- Stars: 3
- exploreFlag

#### 2019.9.6 Merge Sort
- time: 97.54%
- space: 78.95%
```java
class Solution {
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode slow = head, fast = head.next;
        while(fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode right = sortList(slow.next);
        slow.next = null;
        head = sortList(head);
        return merge(head, right);
    }
    public ListNode merge(ListNode l1, ListNode l2) {
        ListNode head = null;
        if (l1 != null) {
            if (l2 != null) {
                if (l1.val < l2.val) {
                    head = l1;
                    l1 = l1.next;
                } else {
                    head = l2;
                    l2 = l2.next;
                }
            } else {
                head = l1;
                l1 = l1.next;
            }
        } else return l2;
        ListNode tail = head;
        while(l1 != null || l2 != null) {
            if (l1 == null) {
                tail.next = l2;
                break;
            } else if (l2 == null) {
                tail.next = l1;
                break;
            } else if (l1.val < l2.val) {
                tail.next = l1;
                l1 = l1.next;
                tail = tail.next;
                tail.next = null;
            } else {
                tail.next = l2;
                l2 = l2.next;
                tail = tail.next;
                tail.next = null;
            }
        }
        return head;
    }
}
```

### 277. Find the Celebrity
- [Link](https://leetcode.com/problems/find-the-celebrity/)
- Tags: Array
- Stars: 3

#### 2019.9.6 
- time: 62.09%
- space: 58.33%
```java
public class Solution extends Relation {
    public int findCelebrity(int n) {
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);
        for(int i=0; i<n-1; i++) {
            if (!candidates[i]) continue;
            for(int j=i+1; j<n; j++) {
                if (!candidates[j]) continue;
                if (knows(i, j)) {
                    candidates[i] = false;
                    break;
                } else {
                    candidates[j] = false;
                }
            }
        }
        for(int i=0; i<n; i++) {
            if (candidates[i]) {
                for(int j=0; j<n; j++) {
                    if (i == j) continue;
                    if (knows(i, j) || !knows(j, i)) return -1;
                }
                return i;
            }
        }
        return -1;
    }
}
```

Optimized 2019.9.6
- time: 62.09%
- space: 58.33%
- interviewLevel
```java
public class Solution extends Relation {
    public int findCelebrity(int n) {
        int cel = 0;
        for(int i=0; i<n; i++) {
            if (cel == i) continue;
            if (knows(cel, i)) {
                cel = i;
            }
        }
        for(int i=0; i<n; i++) {
            if (cel == i) continue;
            if (knows(cel, i) || !knows(i, cel)) return -1;
        }
        return cel;
    }
}
```

### 267. Palindrome Permutation II
- [Link](https://leetcode.com/problems/palindrome-permutation-ii/)
- Tags: Backtracking
- Stars: 3

#### 2019.9.6 
- time: 100%
- space: 100%
- attention: "Basically, we use only 128 total character which is used mostly during program. But total number of Character in ASCII table is 256 (0 to 255). 0 to 31(total 32 character ) is called as ASCII control characters (character code 0-31). 32 to 127 character is called as ASCII printable characters (character code 32-127). 128 to 255 is called as The extended ASCII codes (character code 128-255)."

```java
class Solution {
    List<String> ret = new ArrayList<>();
    int len;
    public List<String> generatePalindromes(String s) {
        len = s.length();
        int[] stat = new int[128];
        for(char c: s.toCharArray()) stat[c]++;
        int count = 0, mid = len/2;
        char[] chrs = new char[len];
        for(int i=0; i<128; i++) {
            if (stat[i] % 2 == 1) {
                count++;
                chrs[mid] = (char)i;
            }
        }
        if (count >= 2) return ret;
        backtrack(stat, chrs, 0, len-1);
        return ret;
    }
    public void backtrack(int[] stat, char[] curr, int l, int r) {
        if (l >= r) {
            ret.add(new String(curr));
            return;
        }
        for(int i=0; i<128; i++) {
            if (stat[i] > 1) {
                stat[i] -= 2;
                curr[l] = curr[r] = (char)i;
                backtrack(stat, curr, l+1, r-1);
                stat[i] += 2;
            }
        }
    }
}
```

### 227. Basic Calculator II
- [Link](https://leetcode.com/problems/basic-calculator-ii/)
- Tags: String
- Stars: 3

#### 2019.9.6 
- time: 17.46%
- space: 97.01%
```java
class Solution {
    public int calculate(String s) {
        Stack<Integer> nums = new Stack<>();
        Stack<Character> ops = new Stack<>();
        int p = 0, len = s.length();
        while(p<len) {
            char c = s.charAt(p);
            if (c == ' ') p++;
            else if (Character.isDigit(c)) {
                int i=p, num = 0;
                do {
                    num *= 10;
                    num += s.charAt(i) - '0';
                    i++;
                } while(i<len && Character.isDigit(s.charAt(i)));
                nums.add(num);
                p = i;
            } else {
                if (c == '*' || c == '/') {
                    while(!ops.isEmpty() && (ops.peek() == '*' || ops.peek() == '/')) {
                        compute(nums, ops);
                    }
                    ops.add(c);
                } else {
                    while(!ops.isEmpty()) {
                        compute(nums, ops);
                    }
                    ops.add(c);
                }
                p++;
            }
        }
        while(!ops.isEmpty()) compute(nums, ops);
        return nums.pop();
    }
    public void compute(Stack<Integer> nums, Stack<Character> ops) {
        int b = nums.pop(), a = nums.pop(), ret;
        char c = ops.pop();
        if (c == '+') ret = a+b;
        else if (c == '-') ret = a-b;
        else if (c == '*') ret = a*b;
        else ret = a/b;
        nums.add(ret);
    }
}
```

### 285. Inorder Successor in BST
- [Link](https://leetcode.com/problems/inorder-successor-in-bst/)
- Tags: Tree
- Stars: 4
- reviewFlag

#### 2019.9.6 DFS
- time: 100%
- space: 5.26%
- attention: Function `inorderSuccessor` is defined to return the node with the smallest value that is greater than `p.val` in the subtree `root`. If no such node exists (all the nodes in the subtree are smaller or equal than `p.val`), return `null`.
```java
class Solution {
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        if (root == null) return null;
        if (root.val <= p.val) return inorderSuccessor(root.right, p);
        TreeNode left = inorderSuccessor(root.left, p);
        return left == null ? root : left;
    }
}
```

### 186. Reverse Words in a String II
- [Link](https://leetcode.com/problems/reverse-words-in-a-string-ii/)
- Tags: String
- Stars: 3

#### 2019.9.6
- time: 100%
- space: 46%
```java
class Solution {
    public void reverseWords(char[] s) {
        reverse(s, 0, s.length-1);
        int start = 0;
        while(start < s.length) {
            int end = start;
            while(end < s.length && s[end] != ' ') end++;
            reverse(s, start, end-1);
            start = end+1;
        }
    }
    public void reverse(char[] s, int l, int r) {
        while(l<r) {
            swap(s, l++, r--);
        }
    }
    public void swap(char[] s, int i, int j) {
        char c = s[i];
        s[i] = s[j];
        s[j] = c;
    }
}
```

### 261. Graph Valid Tree
- [Link](https://leetcode.com/problems/graph-valid-tree/)
- Tags: DFS, BFS, Union Find, Graph
- Stars: 4
- exploreFlag

#### 2019.9.6 Union Find
- time: 100%
- space: 97.30%
```java
class Solution {
    public boolean validTree(int n, int[][] edges) {
        if (n != edges.length+1) return false;
        int[] nums = new int[n];
        for(int i=0; i<n; i++)
            nums[i] = i;
        for(int[] e: edges)
            union(nums, e[0], e[1]);
        int root = find(nums, 0);
        for(int i=1; i<n; i++)
            if (find(nums, i) != root) return false;
        return true;
    }
    public int find(int[] nums, int k) {
        int i=k;
        while(nums[i] != i) {
            int pa = nums[i], gpa = nums[pa];
            nums[i] = gpa;
            i = pa;
        }
        return i;
    }
    public void union(int[] nums, int i, int j) {
        int a = find(nums, i), b = find(nums, j);
        nums[a] = b;
    }
}
```

Optimized 2019.9.6
- time: 100%
- space: 100%
- interviewLevel
```java
class Solution {
    public boolean validTree(int n, int[][] edges) {
        if (n != edges.length+1) return false;
        int[] nums = new int[n];
        Arrays.fill(nums, -1);
        for(int[] e: edges) {
            int a = find(nums, e[0]), b = find(nums, e[1]);
            if (a == b) return false;
            nums[a] = b;
        }
        return true;
    }
    public int find(int[] nums, int k) {
        if (nums[k] == -1) return k;
        nums[k] = find(nums, nums[k]);
        return nums[k];
    }
}
```

### 253. Meeting Rooms II
- [Link](https://leetcode.com/problems/meeting-rooms-ii/)
- Tags: Heap, Greedy, Sort
- Stars: 3

#### 2019.9.6 minHeap
- time: 69.95%
- space: 71.79%
```java
class Solution {
    public int minMeetingRooms(int[][] intervals) {
        if (intervals.length == 0) return 0;
        Arrays.sort(intervals, new Comparator<int[]>() {
           @Override
            public int compare(int[] a, int[] b) {
                return a[0]-b[0];
            }
        });
        PriorityQueue<Integer> heap = new PriorityQueue<>();
        int ret = 1;
        for(int[] interval: intervals) {
            int start = interval[0], end = interval[1];
            while(!heap.isEmpty() && heap.peek() <= start) heap.poll();
            heap.add(end);
            ret = Math.max(ret, heap.size());
        }
        return ret;
    }
}
```

#### 2019.9.6 two pointers
- time: 100%
- space: 71.79%
```java
class Solution {
    public int minMeetingRooms(int[][] intervals) {
        int len = intervals.length;
        if (len == 0) return 0;
        int[] starts = new int[len], ends = new int[len];
        for(int i=0; i<len; i++) {
            starts[i] = intervals[i][0];
            ends[i] = intervals[i][1];
        }
        Arrays.sort(starts);
        Arrays.sort(ends);
        int i=0, j=0, ret=0, count = 0;
        while(i<len && j<len) {
            if (ends[j] <= starts[i]) {
                count--;
                j++;
            } else {
                count++;
                i++;
                ret = Math.max(ret, count);
            }
        }
        return ret;
    }
}
```

### 255. Verify Preorder Sequence in Binary Search Tree
- [Link](https://leetcode.com/problems/verify-preorder-sequence-in-binary-search-tree/)
- Tags: Stack, Tree
- Stars: 5
- reviewFlag

#### 2019.9.6 inorder + preorder --> rebuild tree
- time: 24.43%
- space: 100%
```java
class Solution {
    public boolean verifyPreorder(int[] preorder) {
        int[] inorder = preorder.clone();
        Arrays.sort(inorder);
        return verifyPreorder(preorder, inorder, 0, 0, preorder.length, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
    public boolean verifyPreorder(int[] preorder, int[] inorder, int pIdx, int iIdx, int len, int min, int max) {
        if (len == 0) return true;
        int rootVal = preorder[pIdx];
        if (rootVal<min || rootVal > max) return false;
        int rootiIdx = findIdx(inorder, iIdx, rootVal), 
            leftLen = rootiIdx-iIdx, rightLen = len - leftLen - 1;
        return verifyPreorder(preorder, inorder, pIdx+1, iIdx, leftLen, min, rootVal-1) &&
            verifyPreorder(preorder, inorder, pIdx+1+leftLen, rootiIdx+1, rightLen, rootVal+1, max);
    }
    public int findIdx(int[] nums, int start, int target) {
        for(int i=start; i<nums.length; i++) {
            if (nums[i] == target) return i;
        }
        return nums.length;
    }
}
```

#### 2019.9.6 Stack way of thinking 
- time: 38.02%
- space: 100%
```java
class Solution {
    public boolean verifyPreorder(int[] preorder) {
        Stack<Integer> st = new Stack<>();
        int curr = Integer.MIN_VALUE;
        for(int num: preorder) {
            if (num < curr) return false;
            if (st.isEmpty() || st.peek() > num) {
                st.add(num);
            } else {
                while(!st.isEmpty() && st.peek() < num) {
                    curr = Math.max(curr, st.pop());
                }
                st.add(num);
            }
        }
        return true;
    }
}
```

Optimized 2019.9.6 O(n) time + O(1) space
- time: 81.02%
- space: 100%
- attention: Actually, you don't need a stack shown above. You just need to do a search operation in a sorted subarray and update `curr`.
```java
class Solution {
    public boolean verifyPreorder(int[] preorder) {
        if (preorder.length == 0) return true;
        int start = 0, curr = Integer.MIN_VALUE;
        for(int i=1; i<preorder.length; i++) {
            if (preorder[i] < curr) return false;
            if (preorder[i] < preorder[i-1]) continue;
            int idx = binarySearch(preorder, start, i, preorder[i]);
            curr = preorder[idx];
            if (idx == start) start = i;
        }
        return true;
    }
    public int binarySearch(int[] nums, int start, int end, int target) {
        if (start == end) return start;
        if (target > nums[start]) return start;
        if (target < nums[end-1]) return end;
        int l = start, r = end-1;
        while(l<r) {
            int mid = l + ((r-l)>>1);
            if (nums[mid] > target) l = mid+1;
            else r = mid;
        }
        return l;
    }
}
```

#### 2019.9.6 
- time: 100%
- space: 100%
- cheatFlag
```java
class Solution {
    int idx = 0;
    public boolean verifyPreorder(int[] preorder) {
        if (preorder.length == 0) return true;
        return verifyPreorder(preorder, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
    public boolean verifyPreorder(int[] preorder, int min, int max) {
        if (idx == preorder.length) return true;
        int rootVal = preorder[idx];
        if (rootVal < min || rootVal > max) return false;
        idx++;
        return verifyPreorder(preorder, min, rootVal-1) || verifyPreorder(preorder, rootVal+1, max);
    }
}
```

### 251. Flatten 2D Vector
- [Link](https://leetcode.com/problems/flatten-2d-vector/)
- Tags: Design
- Stars: 2

#### 2019.9.6
- time: 81.78%
- space: 38.89%
```java
class Vector2D {
    int i=0, j=0;
    int[][] v;
    public Vector2D(int[][] v) {
        this.v = v;
    }
    public int next() {
        int ret = v[i][j];
        j++;
        while (i<v.length && j == v[i].length) {
            i++;
            j = 0;
        }
        return ret;
    }
    public boolean hasNext() {
        while (i<v.length && j >= v[i].length) {
            i++;
            j = 0;
        }
        return i<v.length;
    }
}
```

### 298. Binary Tree Longest Consecutive Sequence
- [Link](https://leetcode.com/problems/binary-tree-longest-consecutive-sequence/)
- Tags: Tree
- Stars: 2

#### 2019.9.6 DFS
- time: 100%
- space: 100%
```java
class Solution {
    int ans = 0;
    public int longestConsecutive(TreeNode root) {
        dfs(root);
        return ans;
    }
    public int dfs(TreeNode root) {
        if (root == null) return 0;
        int leftLen = dfs(root.left), rightLen = dfs(root.right), ret = 1;
        if (root.left != null) ret = Math.max(ret, root.val+1 == root.left.val ? leftLen+1 : 1);
        if (root.right != null) ret = Math.max(ret, root.val+1 == root.right.val ? rightLen+1 : 1);
        ans = Math.max(ret, ans);
        return ret;
    }
}
```

### 254. Factor Combinations
- [Link](https://leetcode.com/problems/factor-combinations/)
- Tags: Backtracking
- Stars: 4

#### 2019.9.6 backtracking
- time: 62.91%
- space: 100%
```java
class Solution {
    List<List<Integer>> ret = new LinkedList<>();
    int n;
    List<Integer> factors;
    public List<List<Integer>> getFactors(int n) {
        if (n<=2) return ret;
        this.n = n;
        factors = precomputeFactors(n);
        backtrack(new LinkedList<>(), n, 0);
        return ret;
    }
    public List<Integer> precomputeFactors(int n) {
        List<Integer> ret = new ArrayList<>();
        for(int i=2; i<n; i++) {
            if (n%i == 0) ret.add(i);
        }
        return ret;
    }
    public void backtrack(LinkedList<Integer> currList, int n, int start) {
        if (n<=1) {
            ret.add(new LinkedList<>(currList));
            return;
        }
        int len = factors.size();
        for(int i=start; i<len; i++) {
            int factor = factors.get(i);
            if (factor > n) break;
            if (n%factor == 0) {
                currList.addLast(factor);
                backtrack(currList, n/factor, i);
                currList.removeLast();
            }
        }
    }
}
```

#### 2019.9.6 backtracking with decreasing upper bound
- time: 91.61%
- space: 8.33%
- attention: A decreasing upper bound in each backtrack will reduce the time complexity significantly!!
- cheatFlag
```java
class Solution {
    List<List<Integer>> ret = new LinkedList<>();
    int n;
    public List<List<Integer>> getFactors(int n) {
        if (n<=2) return ret;
        this.n = n;
        backtrack(new LinkedList<>(), n, 2, (int)Math.sqrt(n));
        return ret;
    }
    public void backtrack(LinkedList<Integer> currList, int n, int start, int upper) {
        if (n<=1) {
            ret.add(new LinkedList<>(currList));
            return;
        }
        for(int i=start; i<=n; i++) {
            if (i>upper) i = n;
            if (n%i == 0 && i < this.n) {
                currList.addLast(i);
                backtrack(currList, n/i, i, (int)Math.sqrt(n/i));
                currList.removeLast();
            }
        }
    }
}
```

### 247. Strobogrammatic Number II
- [Link](https://leetcode.com/problems/strobogrammatic-number-ii/)
- Tags: Math, Recursion
- Stars: 2

#### 2019.9.5 backtracking
- time: 100%
- space: 100%
- interviewLevel
- attention: For case `n==1`, `"0"` should be a possible answer. Thus, be careful of it when you are thinking of the leading zero problem.
```java
class Solution {
    List<String> ret = new ArrayList<>();
    public List<String> findStrobogrammatic(int n) {
        backtrack(new char[n], 0, n-1);
        return ret;
    }
    public void backtrack(char[] curr, int l, int r) {
        if (l > r) {
            ret.add(new String(curr));
            return;
        }
        curr[l] = curr[r] = '1';
        backtrack(curr, l+1, r-1);
        curr[l] = curr[r] = '8';
        backtrack(curr, l+1, r-1);
        if (l > 0 || curr.length == 1) {
            curr[l] = curr[r] = '0';
            backtrack(curr, l+1, r-1);
        }
        if (l != r) {
            curr[l] = '6';
            curr[r] = '9';
            backtrack(curr, l+1, r-1);
            curr[l] = '9';
            curr[r] = '6';
            backtrack(curr, l+1, r-1);
        }
    }
}
```

### 259. 3Sum Smaller
- [Link](https://leetcode.com/problems/3sum-smaller/)
- Tags: Array, Two Pointers
- Stars: 5
- cheatFlag

#### 2019.9.5 two pointers
- time: 79.42%
- space: 100%
- interviewLevel
```java
class Solution {
    public int threeSumSmaller(int[] nums, int target) {
        int ret = 0;
        Arrays.sort(nums);
        for(int i=0; i<nums.length-2; i++) {
            int temp = target - nums[i];
            int l=i+1, r=nums.length-1;
            while(l<r) {
                int sum = nums[l] + nums[r];
                if (sum < temp) {
                    ret += r-l;
                    l++;
                } else {
                    r--;
                }
            }
        }
        return ret;
    }
}
```

### 294. Flip Game II
- [Link](https://leetcode.com/problems/flip-game-ii/)
- Tags: Backtracking, Minmax
- Stars: 5
- exploreFlag

#### 2019.9.4 O(N!!) backtracking
- time: 65.91%
- space: 100%
- attention: During backtracking, if you may return the value before restore to be original array, remember to check whether this array will be used or not. e.g. In this case, you may want to write `if (!backtrack(chrs)) return true;` between `chrs[i] = chrs[i-1] = '-'` and `chrs[i] = chrs[i-1] = '+'`. However, the chrs may be further used and edited even if you return true at some point, but `chrs[i] = chrs[i-1] = '+'` will no longer be executed! This will cause an error.
```java
class Solution {
    public boolean canWin(String s) {
        return backtrack(s.toCharArray());
    }
    public boolean backtrack(char[] chrs) {
        for(int i=1; i<chrs.length; i++) {
            if (chrs[i] == '-') i++;
            else if (chrs[i-1] == '+') {
                chrs[i] = chrs[i-1] = '-';
                boolean componentCanWin = backtrack(chrs);
                chrs[i] = chrs[i-1] = '+';
                if (!componentCanWin) return true;
            }
        }
        return false;
    }
}
```

### 249. Group Shifted Strings
- [Link](https://leetcode.com/problems/group-shifted-strings/)
- Tags: Hash Table, String
- Stars: 3

#### 2019.9.3
- time: 100%
- space: 100%
- language: Simly replacing `map.computeIfAbsent(key, k->new ArrayList<>()).add(s)` with `map.putIfAbsent(key, new ArrayList<>()); map.get(key).add(s);` improves from 32 ms to 1 ms!!! The `putIfAbsent` method puts the specified value to the given key only if the key does not exists or the original value is null.
```java
class Solution {
    public List<List<String>> groupStrings(String[] strings) {
        Map<String, List<String>> map = new HashMap<>();
        for(String s: strings) {
            String key = convertToKey(s);
            map.putIfAbsent(key, new ArrayList<>());
            map.get(key).add(s);
        }
        List<List<String>> ret = new ArrayList<>();
        for(List<String> list: map.values()) ret.add(list);
        return ret;
    }
    public String convertToKey(String s) {
        if (s.length() == 0 || s.charAt(0) == 'a') return s;
        char[] chrs = s.toCharArray();
        int diff = chrs[0] - 'a';
        for(int i=0; i<chrs.length; i++) {
            if (chrs[i]-'a' >= diff) chrs[i] -= diff;
            else chrs[i] += (26-diff);
        }
        return new String(chrs);
    }
}
```

### 250. Count Univalue Subtrees
- [Link](https://leetcode.com/problems/count-univalue-subtrees/)
- Tags: Tree
- Stars: 2

#### 2019.9.3 DFS
- time: 87.99%
- space: 66.67%
- attention: `if(dfs(root.left) && dfs(root.right)) xxx` may not work, because the second `dfs` statement will not be executed if the first `dfs` returns `false`
```java
class Solution {
    int count = 0;
    public int countUnivalSubtrees(TreeNode root) {
        if (root == null) return 0;
        dfs(root);
        return count;
    }
    public boolean dfs(TreeNode root) {
        if (root == null) return true;
        boolean l = dfs(root.left), r = dfs(root.right);
        if (l && r) {
            if (root.left != null && root.left.val != root.val) return false;
            if (root.right != null && root.right.val != root.val) return false;
            count++;
            return true;
        }
        return false;
    }
}
```

### 286. Walls and Gates
- [Link](https://leetcode.com/problems/walls-and-gates/)
- Tags: BFS
- Stars: 3
- reviewFlag

#### 2019.9.3 BFS
- time: 47.35%
- space: 28.13%
```java
class Solution {
    Queue<int[]> qu;
    int m, n;
    int[][] rooms;
    public void wallsAndGates(int[][] rooms) {
        if (rooms.length == 0 || rooms[0].length == 0) return;
        this.rooms = rooms;
        m = rooms.length; 
        n = rooms[0].length;
        qu = new LinkedList<>();
        for(int i=0; i<m; i++) 
            for(int j=0; j<n; j++)
                if (rooms[i][j] == 0) qu.add(new int[]{i, j, 0});
        while(!qu.isEmpty()) {
            int[] pos = qu.poll();
            int i=pos[0], j=pos[1], d=pos[2];
            if (d == 0 || d < rooms[i][j]) {
                rooms[i][j] = d;
                addToQueueAfterCheck(i+1, j, d+1);
                addToQueueAfterCheck(i-1, j, d+1);
                addToQueueAfterCheck(i, j+1, d+1);
                addToQueueAfterCheck(i, j-1, d+1);
            }
        }
    }
    public void addToQueueAfterCheck(int i, int j, int d) {
        if (i<0 || i>=m || j<0 || j>=n) return;
        if (d >= rooms[i][j]) return;
        qu.add(new int[]{i, j, d});
    }
}
```

Optimized 2019.9.3
- time: 65.76%
- space: 78.13%
- attention: with BFS, you can record `(i, j)` instead of `(i, j, d)`.
```java
class Solution {
    Queue<int[]> qu;
    int m, n;
    int[][] rooms;
    public void wallsAndGates(int[][] rooms) {
        if (rooms.length == 0 || rooms[0].length == 0) return;
        this.rooms = rooms;
        m = rooms.length; 
        n = rooms[0].length;
        qu = new LinkedList<>();
        for(int i=0; i<m; i++) 
            for(int j=0; j<n; j++)
                if (rooms[i][j] == 0) qu.add(new int[]{i, j});
        int count = qu.size(), level = 0;
        while(!qu.isEmpty()) {
            int[] pos = qu.poll();
            count--;
            int i=pos[0], j=pos[1];
            if (rooms[i][j] == 0 || rooms[i][j] == Integer.MAX_VALUE) {
                rooms[i][j] = level;
                addToQueueAfterCheck(i+1, j);
                addToQueueAfterCheck(i-1, j);
                addToQueueAfterCheck(i, j+1);
                addToQueueAfterCheck(i, j-1);
            }
            if (count == 0) {
                count = qu.size();
                level++;
            }
        }
    }
    public void addToQueueAfterCheck(int i, int j) {
        if (i<0 || i>=m || j<0 || j>=n) return;
        if (rooms[i][j] == Integer.MAX_VALUE) {
            qu.add(new int[]{i, j});
        }
    }
}
```

### 156. Binary Tree Upside Down
- [Link](https://leetcode.com/problems/binary-tree-upside-down/)
- Tags: Tree
- Stars: 2

#### 2019.9.3 
- time: 100%
- space: 100%
- interviewLevel
```java
class Solution {
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null || root.left == null) return root;
        TreeNode left = root.left, newRoot = upsideDownBinaryTree(left);
        left.left = root.right;
        left.right = root;
        root.left = root.right = null;
        return newRoot;
    }
}
```

### 94. Binary Tree Inorder Traversal
- [Link](https://leetcode.com/problems/binary-tree-inorder-traversal/)
- Tags: Hash Table, Stack, Tree
- Stars: 2

#### 2019.9.3
- time: 57.26%
- space: 100%
```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> ret = new ArrayList<>();
        if (root == null) return ret;
        Stack<TreeNode> st = new Stack<>();
        TreeNode curr = root;
        while(curr != null) {
            st.add(curr);
            curr = curr.left;
        }
        while(!st.isEmpty()) {
            TreeNode node = st.pop();
            ret.add(node.val);
            if (node.right != null) {
                node = node.right;
                while(node != null) {
                    st.add(node);
                    node = node.left;
                }
            }
        }
        return ret;
    }
}
```

Updated 2019.9.13
- time: 57.61%
- space: 100%
```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> ret = new ArrayList<>();
        Stack<TreeNode> st = new Stack<>();
        TreeNode p = root;
        while(p != null || !st.isEmpty()) {
            while(p != null) {
                st.add(p);
                p = p.left;
            }
            p = st.pop();
            ret.add(p.val);
            p = p.right;
        }
        return ret;
    }
}
```

### 245. Shortest Word Distance III
- [Link](https://leetcode.com/problems/shortest-word-distance-iii/)
- Tags: Array
- Stars: 3

#### 2019.9.2 two pointers
- time: 100%
- space: 100%
- interviewLevel
```java
class Solution {
    public int shortestWordDistance(String[] words, String word1, String word2) {
        int ret = Integer.MAX_VALUE, len=words.length;
        if (!word1.equals(word2)) {
            int i=moveTo(words, word1, 0), j=moveTo(words, word2, 0);
            while(i<len && j<len) {
                ret = Math.min(ret, Math.abs(j-i));
                if (i<j) i = moveTo(words, word1, i+1); 
                else j = moveTo(words, word2, j+1);
            }
        } else {
            int i=moveTo(words, word1, 0);
            while(i<len) {
                int j = moveTo(words, word1, i+1);
                if (j < len) ret = Math.min(ret, j-i);
                i = j;
            }
        }
        return ret;
    }
    public int moveTo(String[] words, String template, int start) {
        for(int i=start; i<words.length; i++) if (words[i].equals(template)) return i;
        return words.length;
    }
}
```

#### 2019.9.2 HashMap
- time: 12.57%
- space: 100%
```java
class Solution {
    public int shortestWordDistance(String[] words, String word1, String word2) {
        Map<String, List<Integer>> map = new HashMap<>();
        for(int i=0; i<words.length; i++) 
            map.computeIfAbsent(words[i], k->new ArrayList<>()).add(i);
        int ret = Integer.MAX_VALUE;
        if (word1.equals(word2)) {
            List<Integer> list = map.get(word1);
            for(int i=1; i<list.size(); i++) ret = Math.min(ret, list.get(i)-list.get(i-1));
        } else {
            List<Integer> list1 = map.get(word1), list2 = map.get(word2);
            int i=0, j=0, len1=list1.size(), len2=list2.size();
            while(i<len1 && j<len2) {
                int a = list1.get(i), b = list2.get(j);
                if (a<b) {
                    ret = Math.min(ret, b-a);
                    i++;
                } else {
                    ret = Math.min(ret, a-b);
                    j++;
                }
            }
        }
        return ret;
    }
}
```

### 244. Shortest Word Distance II
- [Link](https://leetcode.com/problems/shortest-word-distance-ii/)
- Tags: Hash Table, Design
- Stars: 3

#### 2019.9.2 two pointers
- time: 69.60%
- space: 100%
```java
class WordDistance {
    Map<String, List<Integer>> map = new HashMap<>();
    public WordDistance(String[] words) {
        for(int i=0; i<words.length; i++) 
            map.computeIfAbsent(words[i], k->new ArrayList<>()).add(i);
    }
    public int shortest(String word1, String word2) {
        List<Integer> list1 = map.get(word1), list2 = map.get(word2);
        int i=0, j=0, len1=list1.size(), len2=list2.size(), ret=Integer.MAX_VALUE;
        while(i<len1 && j<len2) {
            int a = list1.get(i), b = list2.get(j);
            ret = Math.min(ret, Math.abs(a-b));
            if (a<b) i++;
            else j++;
        }
        return ret;
    }
}
```

### 243. Shortest Word Distance
- [Link](https://leetcode.com/problems/shortest-word-distance/)
- Tags: Array
- Stars: 3

#### 2019.9.2 two pointers
- time: 100%
- space: 100%
- interviewLevel
```java
class Solution {
    public int shortestDistance(String[] words, String word1, String word2) {
        int i=moveTo(words, word1, 0), j=moveTo(words, word2, 0), len=words.length, ret = Integer.MAX_VALUE;
        while(i<len && j<len) {
            ret = Math.min(ret, Math.abs(j-i));
            if (i<j) i = moveTo(words, word1, i+1); 
            else j = moveTo(words, word2, j+1);
        }
        return ret;
    }
    public int moveTo(String[] words, String template, int start) {
        for(int i=start; i<words.length; i++) if (words[i].equals(template)) return i;
        return words.length;
    }
}
```

### 266. Palindrome Permutation
- [Link](https://leetcode.com/problems/palindrome-permutation/)
- Tags: Hash Table
- Stars: 1

#### 2019.9.2 
- time: 79.24%
- space: 100%
- attention: bit manipulation may not work. e.g. the XOR result of each char in "abdg" is 0!!!
```java
class Solution {
    public boolean canPermutePalindrome(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for(char c: s.toCharArray()) map.put(c, map.getOrDefault(c, 0) + 1);
        int count = 0;
        for(int val: map.values()) if (val % 2 == 1) count++;
        return count<2;
    }
}
```

Another
- time: 100%
- space: 100%
```java
class Solution {
    public boolean canPermutePalindrome(String s) {
        int[] stat = new int[256];
        for(char c: s.toCharArray()) stat[c]++;
        int count = 0;
        for(int num: stat) if(num%2 == 1) count++;
        return count<2;
    }
}
```

### 281. Zigzag Iterator
- [Link](https://leetcode.com/problems/zigzag-iterator/)
- Tags: Design
- Stars: 3

#### 2019.9.2 
- time: 79.24%
- space: 100%
- language: `List.iterator()`
- interviewLevel
```java
public class ZigzagIterator {
    Queue<Iterator<Integer>> qu = new LinkedList<>();
    public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
        if (v1.size() > 0) qu.add(v1.iterator());
        if (v2.size() > 0) qu.add(v2.iterator());
    }
    public int next() {
        Iterator<Integer> iter = qu.poll();
        int ret = iter.next();
        if (iter.hasNext()) qu.add(iter);
        return ret;
    }
    public boolean hasNext() {
        return !qu.isEmpty();
    }
}
```

### 280. Wiggle Sort
- [Link](https://leetcode.com/problems/wiggle-sort/)
- Tags: Array, Sort
- Stars: 3

#### 2019.9.2 sort
- time: 25.64%
- space: 100%
```java
class Solution {
    public void wiggleSort(int[] nums) {
        Arrays.sort(nums);
        for(int i=2; i<nums.length; i+=2) swap(nums, i, i-1);
    }
    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

#### 2019.9.2 quick search by virtual indexing
- time: 26.67%
- space: 100%
```java
class Solution {
    int len, mid;
    public void wiggleSort(int[] nums) {
        len = nums.length;
        mid = (len-1)/2;
        findNthValue(nums, 0, len-1, mid);
    }
    private void findNthValue(int[] nums, int l, int r, int N) {
        if (l >= r) return;
        while(true) {
            int j = partition(nums, l, r);
            if (j > N) r = j-1;
            else if (j < N) l = j+1;
            else return;
        }
    }
    private int partition(int[] nums, int l, int r) {
        int i=l, j=r+1, pivot = get(nums, l);
        while(true) {
            while(get(nums, ++i) < pivot && i<r);
            while(pivot < get(nums, --j) && l<j);
            if (i>=j) break;
            swap(nums, i, j);
        }
        swap(nums, l, j);
        return j;
    }
    private void swap(int[] nums, int vi, int vj) {
        if (vi == vj) return;
        int i = virtual2real(vi), j = virtual2real(vj), temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    private int get(int[] nums, int i) {
        return nums[virtual2real(i)];
    }
    private int virtual2real(int i) {
        if (i <= mid) return (mid-i)*2;
        return (len-1-i)*2 + 1;
    }
    private int real2virtual(int j) {
        if (j%2 == 0) return mid - (j/2);
        return len-1-(j-1)/2;
    }
}
```

#### 2019.9.2 
- time: 78.54%
- space: 100%
- interviewLevel
- cheatFlag
```java
class Solution {
    public void wiggleSort(int[] nums) {
        for(int i=0; i<nums.length; i++) {
            if (i%2 == 1) {
                if (nums[i] < nums[i-1]) swap(nums, i, i-1);
            } else if (i > 0 && nums[i] > nums[i-1]) swap(nums, i, i-1);
        }
    }
    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

### 394. Decode String
- [Link](https://leetcode.com/problems/decode-string/)
- Tags: Stack, DFS
- Stars: 4

#### 2019.8.29 recursive
- time: 100%
- space: 100%
```java
class Solution {
    int p = 0, len;
    String s;
    public String decodeString(String s) {
        if (s.length() == 0) return "";
        this.len = s.length();
        this.s = s;
        StringBuilder sb = new StringBuilder();
        while(p<len) sb.append(decode());
        return sb.toString();
    }
    public String decode() {
        if (p >= len) return "";
        if (Character.isDigit(s.charAt(p))) {
            int k = parseInt();
            p++;
            StringBuilder sb = new StringBuilder();
            while(true) {
                char c = s.charAt(p);
                if (Character.isDigit(c)) {
                    sb.append(decode());
                } else if (c == ']') {
                    p++;
                    break;
                } else sb.append(parseString());
            }
            String temp = sb.toString();
            if (k == 1) return temp;
            for(int i=1; i<k; i++) sb.append(temp);
            return sb.toString();
        }
        return parseString();
    }
    public String parseString() {
        int start = p;
        while(p<len) {
            char c = s.charAt(p);
            if (Character.isDigit(c) || c == ']') break;
            p++;
        }
        return s.substring(start, p);
    }
    public int parseInt() {
        int ret = 0;
        while(Character.isDigit(s.charAt(p))) {
            ret = 10*ret + (s.charAt(p) - '0');
            p++;
        }
        return ret;
    }
}
```

#### 2019.8.29 iterative
- time: 100%
- space: 100%
```java
class Solution {
    public String decodeString(String s) {
        StringBuilder sb = new StringBuilder();
        int p = 0, len = s.length();
        Stack<Integer> count = new Stack<>();
        Stack<StringBuilder> sbs = new Stack<>();
        while(p<len) {
            char c = s.charAt(p);
            if (Character.isDigit(c)) {
                int k=0;
                do {k = 10*k + (s.charAt(p++) - '0');} while(Character.isDigit(s.charAt(p)));
                p++;
                count.add(k);
                sbs.add(sb);
                sb = new StringBuilder();
            } else if (c == ']') {
                p++;
                int k = count.pop();
                StringBuilder prefix = sbs.pop();
                for(int i=0; i<k; i++) prefix.append(sb.toString());
                sb = prefix;
            } else {
                sb.append(c);
                p++;
            }
        }
        return sb.toString();
    }
}
```

### 233. Number of Digit One
- [Link](https://leetcode.com/problems/number-of-digit-one/)
- Tags: Math
- Stars: 4
- exploreFlag

#### 2019.8.29
- time: 100%
- space: 16.67%

`nums[i]` is the total number of digit 1 appearing in all numbers in the scope of `[0, 10^i)`.
```java
class Solution {
    public int countDigitOne(int n) { // take n == 213 as an example
        if (n<0) return 0;
        int[] nums = new int[11];
        for(int i=1, base = 1; i<11; i++, base*=10) 
            nums[i] = 10*nums[i-1] + base;
        int base = 1, count = 0, ret = 0, curr = 0;
        while(n>0) { // e.g. base == 10 
            count++; // count becomes 2
            int digit = n%10; // digit == 1
            int temp = digit*nums[count-1]; 
            if (digit>1) temp += base; // temp is the total number of digit 1 in all numbers of `[0, digit*base)`
            ret += temp;
            if (digit == 1) ret += curr + 1; // `temp + (digit==1?curr+1:0)` is responsible for `[digit*base, digit*base+curr]`
            curr += base*digit;
            base *= 10;
            n /= 10;
        }
        return ret;
    }
}
```

### 282. Expression Add Operators
- [Link](https://leetcode.com/problems/expression-add-operators/)
- Tags: Divide and Conquer
- Stars: 4

#### 2019.8.26 backtrack
- time: 88.13%
- space: 56.76%
- attention: the overflow problem in case of `"2147483648" -2147483648`. 
- attention: a string of number containing leading zeros should not be considered as a valid number, except it is zero itself.

```java
class Solution {
    List<String> result = new ArrayList<>();
    String s;
    int target, len;
    public List<String> addOperators(String num, int target) {
        this.s = num;
        this.len = num.length();
        this.target = target;
        List<String> list = new ArrayList<>();
        long number = 0;
        for(int i=0; i<len; i++) {
            if (s.charAt(0) == '0' && i > 0) break;
            number *= 10;
            number += s.charAt(i) - '0';
            list.add(Long.toString(number));
            backtrack(list, i+1, 0, number, '+');
            list.remove(list.size() - 1);
        }
        return result;
    }
    public void backtrack(List<String> list, int start, long accu, long curr, char lastOp) {
        if (start >= len) {
            if (compute(accu, curr, lastOp) == target) result.add(String.join("", list));
            return;
        }
        int number = 0;
        for(int i=start; i<len; i++) {
            if (s.charAt(start) == '0' && i > start) break;
            number *= 10;
            number += s.charAt(i) - '0';
            list.add("+");
            list.add(Integer.toString(number));
            backtrack(list, i+1, compute(accu, curr, lastOp), number, '+');
            list.set(list.size()-2, "-");
            backtrack(list, i+1, compute(accu, curr, lastOp), number, '-');
            list.set(list.size()-2, "*");
            backtrack(list, i+1, accu, curr*number, lastOp);
            list.remove(list.size()-1);
            list.remove(list.size()-1);
        }
    }
    public long compute(long a, long b, char op) {
        if (op == '+') return a+b;
        else if (op == '-') return a-b;
        return a*b;
    }
}
```

### 224. Basic Calculator
- [Link](https://leetcode.com/problems/basic-calculator/)
- Tags: Math, Stack
- Stars: 3

#### 2019.8.25 two step
- time: 13.80%
- space: 69.23%
- reference: https://www.cnblogs.com/journal-of-xjx/p/5940030.html

```java
class Solution {
    public int calculate(String s) {
        List<Object> list = new ArrayList<>();
        Stack<Character> st = new Stack<>();
        int i = 0, len = s.length();
        while(i < len) {
            char c = s.charAt(i);
            if (Character.isDigit(c)) {
                int j = i+1;
                while(j<len && Character.isDigit(s.charAt(j))) j++;
                list.add(Integer.parseInt(s.substring(i, j)));
                i = j;
            } else if (c == ' ') i++;
            else if (c == ')') {
                while (true) {
                    char op = st.pop();
                    if (op == '(') break;
                    else list.add(op);
                }
                i++;
            } else if (c == '(') {
                st.add(c);
                i++;
            } else {
                while(!st.isEmpty() && st.peek() != '(') list.add(st.pop());
                st.add(c);
                i++;
            }
        }
        while(!st.isEmpty()) list.add(st.pop());
        Stack<Integer> nums = new Stack<>();
        for(Object item: list) {
            if (item instanceof Integer) nums.add((int)item);
            else {
                int b = nums.pop(), a = nums.pop();
                if ((char)item == '+') nums.add(a+b);
                else nums.add(a-b);
            }
        }
        return nums.pop();
    }
}
```

#### 2019.8.25 one step
- time: 18.01%
- space: 70.77%

```java
class Solution {
    public int calculate(String s) {
        Stack<Integer> nums = new Stack<>();
        Stack<Character> ops = new Stack<>();
        int i=0, len = s.length();
        while(i<len) {
            char c = s.charAt(i);
            if (c == ' ') i++;
            else if (c == '(') {
                ops.add(c);
                i++;
            } else if (c == ')') {
                while(ops.peek() != '(') compute(nums, ops);
                ops.pop();
                i++;
            } else if (Character.isDigit(c)) {
                int j=i+1;
                while(j<len && Character.isDigit(s.charAt(j))) j++;
                nums.add(Integer.parseInt(s.substring(i, j)));
                i = j;
            } else {
                while(!ops.isEmpty() && ops.peek() != '(') compute(nums, ops);
                ops.add(c);
                i++;
            }
        }
        while(!ops.isEmpty()) compute(nums, ops);
        return nums.pop();
    }
    private void compute(Stack<Integer> nums, Stack<Character> ops) {
        if (ops.peek() == '(') return;
        int b = nums.pop(), a = nums.pop();
        if (ops.pop() == '+') nums.add(a+b);
        else nums.add(a-b);
    }
}
```

#### 2019.8.25
- time: 36.11%
- space: 100%
- interviewLevel

```java
class Solution {
    public int calculate(String s) {
        Stack<Integer> st = new Stack<>();
        int i = 0, len = s.length(), result = 0, sign = 1;
        while(i<len) {
            char c = s.charAt(i);
            if (c == ' ') ;
            else if (Character.isDigit(c)) {
                int j=i, number = 0;
                while(j<len && Character.isDigit(s.charAt(j))) {
                    number *= 10;
                    number += s.charAt(j) - '0';
                    j++;
                }
                result += number * sign;
                i = j - 1;
            } else if (c == '+') {
                sign = 1;
            } else if (c == '-') {
                sign = -1;
            } else if (c == '(') {
                st.add(result);
                st.add(sign);
                result = 0;
                sign = 1;
            } else {
                result = result*st.pop() + st.pop();
            }
            i++;
        }
        return result;
    }
}
```

### 221. Maximal Square
- [Link](https://leetcode.com/problems/maximal-square/)
- Tags: Dynamic Programming
- Stars: 4

#### 2019.8.22 DP
- time: 99.51%
- space: 100%
- attention: if the ordinary idea doesn't work, try to think in a DFS or DP way.

```java
class Solution {
    public int maximalSquare(char[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) return 0;
        int[][] dp = new int[matrix.length][matrix[0].length];
        for(int i=0; i<matrix.length; i++) if (matrix[i][0] == '1') dp[i][0] = 1;
        for(int j=0; j<matrix[0].length; j++) if (matrix[0][j] == '1') dp[0][j] = 1;
        for(int i=1; i<matrix.length; i++)
            for(int j=1; j<matrix[0].length; j++) 
                dp[i][j] = matrix[i][j] == '0' ? 0 : 1 + Math.min(dp[i-1][j-1], Math.min(dp[i-1][j], dp[i][j-1]));
        int result = 0;
        for(int i=0; i<matrix.length; i++)
            for(int j=0; j<matrix[0].length; j++) 
                if (dp[i][j] > result) result = dp[i][j];
        return result*result;
    }
}
```

### 188. Best Time to Buy and Sell Stock IV
- [Link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)
- Tags: Dynamic Programming
- Stars: 5

#### 2019.8.21 state machine (DP)
- time: 90.74%
- space: 100%
- interviewLevel
- attention: `if (k >= prices.length/2) return quickMaxProfit(prices);` is to avoid TLE for some large `k` cases.

```java
class Solution {
    public int maxProfit(int k, int[] prices) {
        if (k >= prices.length/2) return quickMaxProfit(prices);
        int[] states = new int[2*k+1];
        for(int i=1; i<=2*k; i+=2) states[i] = Integer.MIN_VALUE;
        for(int p: prices) {
            int i=2*k;
            while(i>0) {
                states[i] = Math.max(states[i], states[i-1] + p);
                i--;
                states[i] = Math.max(states[i], states[i-1] - p);
                i--;
            }
        }
        int result = 0;
        for(int i=2; i<=2*k; i+=2) if (states[i] > result) result = states[i];
        return result;
    }
    public int quickMaxProfit(int[] prices) {
        int result = 0;
        for(int i=1; i<prices.length; i++) result += Math.max(0, prices[i]-prices[i-1]);
        return result;
    }
}
```

#### 2019.8.21 DP
- time: 69.85%
- space: 25%

```java
class Solution {
    public int maxProfit(int k, int[] prices) {
        if (k >= prices.length/2) return quickMaxProfit(prices);
        int[][] dp = new int[k+1][prices.length];
        for(int i=1; i<k+1; i++) {
            int currMaxB4Sell = -prices[0];
            for(int j=1; j<prices.length; j++) {
                dp[i][j] = Math.max(dp[i][j-1], prices[j] + currMaxB4Sell);
                currMaxB4Sell = Math.max(currMaxB4Sell, dp[i-1][j-1] - prices[j]);
            }
        }
        return dp[k][prices.length-1];
    }
    public int quickMaxProfit(int[] prices) {
        int result = 0;
        for(int i=1; i<prices.length; i++) result += Math.max(0, prices[i]-prices[i-1]);
        return result;
    }
}
```

### 137. Single Number II
- [Link](https://leetcode.com/problems/single-number-ii/)
- Tags: Bit Manipulation
- Stars: 3

#### 3n+1 bit manipulation
- time: 100.00%
- space: 99.33%

For **Bit manipulation** problems, think **bitwisely**!!

In *Single number I*, each bit of the answer encounters one for `2n+1` times. Therefore, when iterating through the `nums` array, we can use XOR operations to eliminate `2n` occurances for each bit, remaining the only `1` from `2n+1` to be the right value of that bit.

However, this case has a `3n+1` pattern, which demands for a 2-bit counter to deal with `3n`. The main idea is as follows. We iterate through `nums` and record the number of occurances of 1 for each bit. Only three values for each counter are allowed: not occured yet (two-bit code is 00), occured for once (two-bit code is 01), occured for twice (two-bit code is 10). Whenever a bit has encountered with 3 ones (two-bit code is 11), we artifically add another one for that particular bit to convert the counter of that bit into 0 (11 + 1 --> 00). Following this logic, we eliminate `3n` from `3n+1` and only the `1` from `3n+1` remains. 

```java
class Solution {
    public int singleNumber(int[] nums) {
        if(nums.length == 0) return 0;
        int a = 0, carry = 0;
        for(int num: nums) {
            carry |= a & num;
            a ^= num;
            int bitsOccurred3times = carry & a;
            a ^= bitsOccurred3times;
            carry ^= bitsOccurred3times;
        }
        return a;
    }
}
```

### 89. Gray Code
- [Link](https://leetcode.com/problems/gray-code/)
- Tags: Backtracking
- Stars: 1

#### reverse order of last iteration
- time: 100.00%
- space: 7.27%

```java
class Solution {
    public List<Integer> grayCode(int n) {
        if(n == 0) return Arrays.asList(0);
        List<Integer> result = new ArrayList<>();
        result.add(0);
        result.add(1);
        for(int i=1; i<n; i++) {
            int j = result.size() - 1;
            for(; j>=0; j--) {
                result.add((1<<i) | (result.get(j)));
            }
        }
        return result;
    }
}
```

### 24. Swap Nodes in Pairs
- [Link](https://leetcode.com/problems/swap-nodes-in-pairs/)
- Tags: Linked List
- Stars: 1

#### recursive
- time: 100%
- space: 100%
- interviewLevel

```java
class Solution {
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode curr = head, next = head.next;
        head = swapPairs(next.next);
        next.next = curr;
        curr.next = head;
        return next;
    }
}
```

#### recursive
- time: 100%
- space: 100%

```java
class Solution {
    public ListNode swapPairs(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode newHead = head.next;
        do {
            ListNode curr = head, next = head.next;
            head = next.next;
            next.next = curr;
            curr.next = (head == null || head.next == null) ? head : head.next;
        } while (head != null && head.next != null);
        return newHead;
    }
}
```

### 153. Find Minimum in Rotated Sorted Array
- [Link](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)
- Tags: Array, Binary Search
- Stars: 3

#### binary search
- time: 100%
- space: 98.62%
- interviewLevel
- attention: in this line `nums[mid] >= nums[0]`, it must be `>=`, not `>`. That's what a sorted array means!

```java
class Solution {
    public int findMin(int[] nums) {
        int l = 0, r = nums.length-1;
        while(l < r) {
            int mid = l + (r-l)/2;
            if(nums[mid] >= nums[0]) l = mid + 1;
            else r = mid;
        }
        return Math.min(nums[0], nums[l]);
    }
}
```

Another version
- time: 100%
- space: 100%
```java
class Solution {
    public int findMin(int[] nums) {
        int l = 0, r = nums.length - 1;
        while(l < r) {
            int mid = l + r >> 1;
            if (nums[mid] <= nums[nums.length-1]) r = mid;
            else l = mid + 1;
        }
        return nums[l];
    }
}
```

### 16. 3Sum Closest
- [Link](https://leetcode.com/problems/3sum-closest/)
- Tags: Array, Two Pointers
- Stars: 2

#### two pointers
- time: 44.28%
- space: 100%
- interviewLevel

```java
class Solution {
    public int threeSumClosest(int[] nums, int target) {
        int result = nums[0] + nums[1] + nums[2];
        Arrays.sort(nums);
        for(int i=0; i<nums.length - 2; i++) {
            int l = i+1, r = nums.length - 1;
            while(l<r) {
                int curr = nums[i] + nums[l] + nums[r];
                if (Math.abs(curr - target) < Math.abs(result - target))
                    result = curr;
                if (curr > target) r--;
                else if (curr < target) l++;
                else {
                    r--; l++;
                }
            }
        }
        return result;
    }
}
```

### 114. Flatten Binary Tree to Linked List
- [Link](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)
- Tags: Tree, DFS
- Stars: 2

#### DFS
- time: 48.17%
- space: 68.05%

```java
class Solution {
    public void flatten(TreeNode root) {
        if(root == null) return;
        helper(root);
    }
    
    public TreeNode helper(TreeNode root) {
        // root should not be a null value
        if (root.left == null && root.right == null) return root;
        if (root.left == null) return helper(root.right);
        TreeNode left = root.left, right = root.right;
        root.left = null;
        root.right = left;
        TreeNode leftLastNode = helper(left);
        if (right != null) {
            leftLastNode.right = right;
            return helper(right);
        }
        return leftLastNode;
    }
}
```

#### reverse post order traversal
- time: 48.17%
- space: 100%
- interviewLevel

```java
class Solution {
    TreeNode suffix = null;
    public void flatten(TreeNode root) {
        if (root == null) return;
        flatten(root.right);
        flatten(root.left);
        root.left = null;
        root.right = suffix;
        suffix = root;
    }
}
```

#### non-recursive, O(1) space
- time: 48.17%
- space: 57.97%
- interviewLevel

```java
class Solution {
    public void flatten(TreeNode root) {
        TreeNode curr = root;
        while(curr != null) {
            if (curr.left != null) {
                TreeNode leftMax = curr.left;
                while(leftMax.right != null) leftMax = leftMax.right;
                leftMax.right = curr.right;
                curr.right = curr.left;
                curr.left = null;
            }
            curr = curr.right;
        }
    }
}
```

### 109. Convert Sorted List to Binary Search Tree
- [Link](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/)
- Tags: Linked List, DFS
- Stars: 2

#### DFS on linked list
- time: 100%
- space: 100%

```java
class Solution {
    public TreeNode sortedListToBST(ListNode head) {
        if(head == null) return null;
        ListNode midNode = getMidNode(head);
        TreeNode root = new TreeNode(midNode.val);
        
        ListNode right = midNode.next;
        midNode.next = null;
        root.right = sortedListToBST(right);
        
        if (head != midNode) {
            ListNode leftMax = getPreNode(head, midNode);
            leftMax.next = null;
            root.left = sortedListToBST(head);
        }
        return root;
    }
    public ListNode getMidNode(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode slow = head, fast = head;
        do {
            slow = slow.next;
            fast = fast.next.next;
        } while (fast != null && fast.next != null);
        return slow;
    }
    public ListNode getPreNode(ListNode head, ListNode target) {
        if(head == target) return null;
        ListNode curr = head;
        while(curr != null && curr.next != target) curr = curr.next;
        return curr;
    }
}
```

#### DFS on linked list, 2nd version
- time: 100%
- space: 100%
- interviewLevel

```java
class Solution {
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) return null;
        return sortedListToBST(head, null);
    }
    public TreeNode sortedListToBST(ListNode head, ListNode tail) {
        if (head == null || head == tail) return null;
        ListNode midNode = getMidListNode(head, tail);
        TreeNode root = new TreeNode(midNode.val);
        root.left = sortedListToBST(head, midNode);
        root.right = sortedListToBST(midNode.next, tail);
        return root;
    }
    public ListNode getMidListNode(ListNode head, ListNode tail) {
        if (head.next == tail) return head;
        ListNode slow = head, fast = head;
        do {
            slow = slow.next;
            fast = fast.next.next;
        } while(fast != tail && fast.next != tail);
        return slow;
    }
}
```

### 284. Peeking Iterator
- [Link](https://leetcode.com/problems/peeking-iterator/)
- Tags: Design
- Stars: 2

#### 20190728
- time: 80.94%
- space: 100%

```java
class PeekingIterator implements Iterator<Integer> {
    Integer peekVal;
    Iterator<Integer> iterator;
	public PeekingIterator(Iterator<Integer> iterator) {
	    this.iterator = iterator;
        if(iterator.hasNext()) peekVal = iterator.next();
	}
	public Integer peek() {
        return peekVal;
	}
	@Override
	public Integer next() {
        Integer ret = peekVal;
        peekVal = iterator.hasNext() ? iterator.next() : null;
        return ret;
	}
	@Override
	public boolean hasNext() {
	    return peekVal != null || iterator.hasNext();
	}
}
```

### 113. Path Sum II
- [Link](https://leetcode.com/problems/path-sum-ii/)
- Tags: Tree, DFS
- Stars: 2

#### DFS
- time: 100%
- space: 100%
- interviewLevel

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        if (root == null) return result;
        backtrack(root, new ArrayList<>(), sum);
        return result;
    }
    public void backtrack(TreeNode root, List<Integer> currList, int target) {
        currList.add(root.val);
        if (root.left != null) backtrack(root.left, currList, target - root.val);
        if (root.right != null) backtrack(root.right, currList, target - root.val);
        if (root.left == null && root.right == null && target == root.val) result.add(new ArrayList<>(currList));
        currList.remove(currList.size() - 1);
    }
}
```

### 80. Remove Duplicates from Sorted Array II
- [Link](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/)
- Tags: Array, Two Pointers
- Stars: 4

#### stupid solution
- time: 78.56%
- space: 100%

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        if (nums.length == 0) return 0;
        int len = 0, count = 0, i = 0;
        while (i<nums.length) {
            if (i == 0 || nums[i] == nums[i-1]) {
                count++;
                if (count > 2) {
                    i++;
                    continue;
                }
                nums[len++] = nums[i++];
            } else {
                count = 1;
                nums[len++] = nums[i++];
            }
        }
        return len;
    }
}
```

#### two pointers, clean and grace
- time: 100%
- space: 100%
- interviewLevel

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        int len = 0;
        for(int num : nums) 
            if (len < 2 || num > nums[len-2]) 
                nums[len++] = num;
        return len;
    }
}
```

### 299. Bulls and Cows
- [Link](https://leetcode.com/problems/bulls-and-cows/)
- Tags: Hash Table
- Stars: 3

#### 2019.7.28
- time: 57.24%
- space: 100%

```java
class Solution {
    public String getHint(String secret, String guess) {
        int[] stat = new int[10], stat2 = new int[10];
        int bull = 0, cow = 0;
        for(char c: secret.toCharArray()) stat[c-'0'] += 1;
        for(char c: guess.toCharArray()) stat2[c-'0'] += 1;
        for(int i=0; i<guess.length(); i++) {
            char c = guess.charAt(i);
            if (guess.charAt(i) == secret.charAt(i)) {
                bull++;
                stat[c-'0']--;
                stat2[c-'0']--;
            }
        }
        for(int i=0; i<guess.length(); i++) {
            char c = guess.charAt(i);
            if (stat[c-'0'] > 0 && stat2[c-'0'] > 0) {
                cow++;
                stat[c-'0']--;
                stat2[c-'0']--;
            }
        }
        return bull + "A" + cow + "B";
    }
}
```

#### one stat
- time: 100%
- space: 61.31%
- interviewLevel

```java
class Solution {
    public String getHint(String secret, String guess) {
        int[] stat = new int[10];
        int bulls = 0, cows = 0;
        for(int i=0; i<guess.length(); i++) {
            int s = secret.charAt(i) - '0';
            int g = guess.charAt(i) - '0';
            if (s == g) bulls++;
            else {
                if (stat[s] < 0) cows++;
                if (stat[g] > 0) cows++;
                stat[s]++;
                stat[g]--;
            }
        }
        return bulls + "A" + cows + "B";
    }
}
```

### 120. Triangle
- [Link](https://leetcode.com/problems/triangle/)
- Tags: Array, Dynamic Programming
- Stars: 3

#### DP top-down
- time: 84.85%
- space: 99.16%

```java
class Solution {
    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle.size() == 0 || triangle.get(0).size() == 0) return 0;
        int[] nums = new int[triangle.get(triangle.size() - 1).size()];
        for(int i=0; i<triangle.size(); i++) {
            List<Integer> row = triangle.get(i);
            for(int j=row.size()-1; j>=0; j--) {
                if (j == 0) nums[j] = row.get(j) + nums[j];
                else if (j == row.size() - 1) nums[j] = row.get(j) + nums[j-1];
                else nums[j] = row.get(j) + Math.min(nums[j-1], nums[j]);
            }
        }
        int result = Integer.MAX_VALUE;
        for(int num: nums) result = Math.min(result, num);
        return result;
    }
}
```

#### DP bottom-up
- time: 99.83%
- space: 99.00%
- interviewLevel

[This post](https://leetcode.com/problems/triangle/discuss/38730/DP-Solution-for-Triangle) can be helpful.

```java
class Solution {
    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle.size() == 0 || triangle.get(0).size() == 0) return 0;
        int[] dp = new int[triangle.size() + 1];
        for (int i=triangle.size()-1; i>=0; i--) {
            List<Integer> row = triangle.get(i);
            for(int j=0; j<row.size(); j++) dp[j] = row.get(j) + Math.min(dp[j], dp[j+1]);
        }
        return dp[0];
    }
}
```

### 106. Construct Binary Tree from Inorder and Postorder Traversal
- [Link](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)
- Tags: Array, Tree, DFS
- Stars: 1

#### recursive
- time: 15.94%
- space: 10.03%

```java
class Solution {
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        return buildTree(inorder, 0, postorder, 0, inorder.length);
    }
    public TreeNode buildTree(int[] inorder, int iIdx, int[] postorder, int pIdx, int len) {
        if (len == 0) return null;
        TreeNode root = new TreeNode(postorder[pIdx + len - 1]);
        int mid = findIdx(inorder, root.val, iIdx);
        int leftLen = mid - iIdx, rightLen = len - 1 - leftLen;
        root.left = buildTree(inorder, iIdx, postorder, pIdx, leftLen);
        root.right = buildTree(inorder, iIdx+leftLen+1, postorder, pIdx+leftLen, rightLen);
        return root;
    }
    public int findIdx(int[] nums, int target, int start) {
        for(int i=start; i<nums.length; i++)
            if (nums[i] == target) return i;
        return -1;
    }
}
```

### 208. Implement Trie (Prefix Tree)
- [Link](https://leetcode.com/problems/implement-trie-prefix-tree/)
- Tags: Design, Trie
- Stars: 4

#### Naive
- time: 33.84%
- space: 99.88%

```java
class Trie {
    boolean[] marked;
    Trie[] letters;
    public Trie() {}
    public void insert(String word) {
        if (word.length() == 0) return ;
        int w = word.charAt(0) - 'a';
        if (letters == null) {
            letters = new Trie[26];
            marked = new boolean[26];
        }
        if (letters[w] == null) letters[w] = new Trie();
        if (word.length() == 1) marked[w] = true;
        else letters[w].insert(word.substring(1, word.length()));
    }
    public boolean search(String word) {
        if (word == null || word.length() == 0) return true;
        if (letters == null) return false;
        int w = word.charAt(0) - 'a';
        if (word.length() == 1) return marked[w];
        return letters[w] != null && letters[w].search(word.substring(1, word.length()));
    }
    public boolean startsWith(String prefix) {
        if (prefix == null || prefix.length() == 0) return true;
        if (letters == null) return false;
        int w = prefix.charAt(0) - 'a';
        if (letters[w] == null) return false;
        return letters[w].startsWith(prefix.substring(1, prefix.length()));
    }
}
```

Optimized 2019.9.6
- time: 86.90%
- space: 100%
- interviewLevel
- cheatFlag
- attention: use class `Node` instead of `Trie` itself can better reduce resource consumption.
```java
class Trie {
    private final Node root = new Node();
    public Trie() {}
    public void insert(String word) {
        Node node = root;
        for(char c: word.toCharArray()) {
            int idx = c-'a';
            if (node.next[idx] == null) node.next[idx] = new Node();
            node = node.next[idx];
        }
        node.isWord = true;
    }
    public boolean search(String word) {
        Node node = root;
        for(char c: word.toCharArray()) {
            int idx = c-'a';
            if (node.next[idx] == null) return false;
            node = node.next[idx];
        }
        return node.isWord;
    }
    public boolean startsWith(String prefix) {
        Node node = root;
        for(char c: prefix.toCharArray()) {
            int idx = c-'a';
            if (node.next[idx] == null) return false;
            node = node.next[idx];
        }
        return true;
    }
    private class Node {
        boolean isWord = false;
        Node[] next = new Node[26];
    }
}
```

### 116. Populating Next Right Pointers in Each Node
- [Link](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)
- Tags: Tree, DFS
- Stars: 3

#### DFS
- time: 100%
- space: 99.31%
- interviewLevel

```java
class Solution {
    public Node connect(Node root) {
        if (root == null) return null;
        Node childPre = null;
        if (root.left != null) {
            root.left.next = root.right;
            childPre = root.left;
        }
        if (root.right != null) childPre = root.right;
        if (root.next != null && childPre != null) childPre.next = root.next.left;
        connect(root.left);
        connect(root.right);
        return root;
    }
}
```

### 147. Insertion Sort List
- [Link](https://leetcode.com/problems/insertion-sort-list/)
- Tags: Linked List, Sort
- Stars: 2

#### 2019.7.29
- time: 67.41%
- space: 100%

```java
class Solution {
    public ListNode insertionSortList(ListNode head) {
        if (head == null) return head;
        ListNode curr = head.next;
        head.next = null;
        while (curr != null) {
            ListNode next = curr.next;
            head = insertNode(head, curr);
            curr = next;
        }
        return head;
    }
    public ListNode insertNode(ListNode head, ListNode curr) {
        if (curr.val < head.val) {
            curr.next = head;
            return curr;
        }
        ListNode p = head, last = null;
        while (p != null && p.val <= curr.val) {
            last = p;
            p = p.next;
        }
        curr.next = last.next;
        last.next = curr;
        return head;
    }
}
```

### 86. Partition List
- [Link](https://leetcode.com/problems/partition-list/)
- Tags: Linked List, Two Pointers
- Stars: 2

#### 2019.7.29
- time: 100%
- space: 100%

```java
class Solution {
    public ListNode partition(ListNode head, int x) {
        if (head == null) return null;
        ListNode left = new ListNode(0), right = new ListNode(0), leftHead = left, rightHead = right, curr = head;
        while(curr != null) {
            ListNode next = curr.next;
            curr.next = null;
            if (curr.val < x) left = append(left, curr);
            else right = append(right, curr);
            curr = next;
        }
        left.next = rightHead.next;
        return leftHead.next;
    }
    public ListNode append(ListNode tail, ListNode node) {
        tail.next = node;
        return node;
    }
}
```

### 264. Ugly Number II
- [Link](https://leetcode.com/problems/ugly-number-ii/)
- Tags: Math, Dynamic Programming, Heap
- Stars: 4

#### min heap
- time: 6.79%
- space: 5.25%
```java
class Solution {
    public int nthUglyNumber(int n) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        HashSet<Integer> set = new HashSet<>();
        int count = 1;
        minHeap.add(1);
        while(count < n) {
            int curr = minHeap.poll();
            count++;
            safeInsertHeap(minHeap, set, curr, 2);
            safeInsertHeap(minHeap, set, curr, 3);
            safeInsertHeap(minHeap, set, curr, 5);
        }
        return minHeap.poll();
    }
    public void safeInsertHeap(PriorityQueue<Integer> heap, HashSet<Integer> set, int num, int k) {
        if (num > Integer.MAX_VALUE/k) return;
        num *= k;
        if(!set.contains(num) && num > 0) {
            heap.add(num);
            set.add(num);
        }
    }
}
```

#### 2019.8.31
- time: 54.99%
- space: 54.55%
- reference: https://leetcode.com/problems/ugly-number-ii/
```java
class Solution {
    public int nthUglyNumber(int n) {
        int[] ugly = new int[n], indices = new int[3], primes = new int[]{2, 3, 5};
        Arrays.fill(ugly, Integer.MAX_VALUE);
        ugly[0] = 1;
        for(int i=1; i<n; i++) {
            for(int j=0; j<3; j++) ugly[i] = Math.min(ugly[i], primes[j]*ugly[indices[j]]);
            for(int j=0; j<3; j++) if (primes[j]*ugly[indices[j]] <= ugly[i]) indices[j]++;
        }
        return ugly[n-1];
    }
}
```

### 201. Bitwise AND of Numbers Range
- [Link](https://leetcode.com/problems/bitwise-and-of-numbers-range/)
- Tags: Bit Manipulation
- Stars: 2

#### 2019.7.29
- time: 100%
- space: 5.59%
- interviewLevel

```java
class Solution {
    public int rangeBitwiseAnd(int m, int n) {
        int count = 0;
        while(m!=n) {
            m>>=1; n>>=1;
            count++;
        }
        return m<<count;
    }
}
```

### 223. Rectangle Area
- [Link](https://leetcode.com/problems/rectangle-area/)
- Tags: Math
- Stars: 3

#### 2019.7.29
- time: 100%
- space: 5.21%
- interviewLevel

```java
class Solution {
    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int area1 = getArea(A, B, C, D);
        int area2 = getArea(E, F, G, H);
        int x = getOverlap(A, C, E, G);
        int y = getOverlap(B, D, F, H);
        int area3 = (x < 0 || y < 0) ? 0 : x*y;
        return area1 + area2 - area3;
    }
    public int getOverlap(int x1, int x2, int x3, int x4) {
        if (x2 < x3 || x1 > x4) return 0;
        return Math.min(x2, x4) - Math.max(x1, x3);
    }
    public int getArea(int x1, int y1, int x2, int y2) {
        return (x2-x1) * (y2 - y1);
    }
}
```

### 187. Repeated DNA Sequences
- [Link](https://leetcode.com/problems/repeated-dna-sequences/)
- Tags: Hash Table, Bit Manipulation
- Stars: 3
- reviewFlag

#### bit encoding
- time: 29.88%
- space: 98.85%

```java
class Solution {
    public List<String> findRepeatedDnaSequences(String s) {
        List<String> result = new ArrayList<>();
        int num = 0, mask = (1<<20) - 1;
        HashMap<Integer, Integer> map = new HashMap<>();
        for(int i=0; i<s.length(); i++) {
            char c = s.charAt(i);
            num <<= 2;
            if (c == 'A') num += 1;
            else if (c == 'C') num += 2;
            else if (c == 'G') num += 3;
            num &= mask;
            if (i >= 9) {
                map.put(num, map.getOrDefault(num, 0) + 1);
                if (map.get(num) == 2) result.add(s.substring(i-9, i+1));
            }
        }
        return result;
    }
}
```

#### bit encoding 2
- time: 78.66%
- space: 99.42%
- interviewLevel

Using two HashSet instead of a HashMap
```java
class Solution {
    public List<String> findRepeatedDnaSequences(String s) {
        List<String> result = new ArrayList<>();
        int num = 0, mask = (1<<20) - 1;
        HashSet<Integer> seen = new HashSet<>(), repeated = new HashSet<>();
        for(int i=0; i<s.length(); i++) {
            char c = s.charAt(i);
            num <<= 2;
            if (c == 'A') num += 1;
            else if (c == 'C') num += 2;
            else if (c == 'G') num += 3;
            num &= mask;
            if (i >= 9 && !seen.add(num) && repeated.add(num))
                result.add(s.substring(i-9, i+1));
        }
        return result;
    }
}
```

### 228. Summary Ranges
- [Link](https://leetcode.com/problems/summary-ranges/)
- Tags: Array
- Stars: 1

#### two pointers
- time: 100%
- space: 100%
- interviewLevel

```java
class Solution {
    public List<String> summaryRanges(int[] nums) {
        List<String> result = new ArrayList<>();
        if (nums.length == 0) return result;
        int start = nums[0], end = nums[0];
        for(int i=1; i<nums.length; i++) {
            if (nums[i] == nums[i-1] + 1) end++;
            else {
                result.add(getRange(start, end));
                start = end = nums[i];
            }
        }
        result.add(getRange(start, end));
        return result;
    }
    public String getRange(int start, int end) {
        if (start == end) return Integer.toString(start);
        return start + "->" + end;
    }
}
```

### 275. H-Index II
- [Link](https://leetcode.com/problems/h-index-ii/)
- Tags: Binary Search
- Stars: 3
- reviewFlag

#### binary search with out of array check!
- time: 100%
- space: 100%
- attention: `h` might be 0, which means `mid` might be `citations.length`. That's why we need this line: `if (citations[citations.length - 1] == 0) return 0;`

```java
class Solution {
    public int hIndex(int[] citations) {
        if (citations.length == 0) return 0;
        if (citations[citations.length - 1] == 0) return 0;
        int l = 0, r = citations.length - 1;
        while(l < r) {
            int mid = l + ((r-l)>>1);
            int h = citations.length - mid;
            if (citations[mid] < h) l = mid + 1;
            else if (citations[mid] > h) r = mid;
            else return h;
        }
        return citations.length - l;
    }
}
```

Updated 2019.9.13
- time: 100%
- space: 100%
- interviewLevel
```java
class Solution {
    public int hIndex(int[] nums) {
        if (nums.length == 0 || nums[nums.length-1] == 0) return 0;
        int l = 1, r = nums.length;
        while(l<r) {
            int mid = l + r + 1 >> 1;
            if (nums[nums.length-mid] >= mid) l = mid;
            else r = mid - 1;
        }
        return l;
    }
}
```

### 95. Unique Binary Search Trees II
- [Link](https://leetcode.com/problems/unique-binary-search-trees-ii/)
- Tags: Dynamic Programming, Tree
- Stars: 3

#### backtrack-like solution
- time: 100%
- space: 99.33%
- interviewLevel

```java
class Solution {
    List<TreeNode> result = new ArrayList<>();
    public List<TreeNode> generateTrees(int n) {
        if(n == 0) return result;
        result.add(new TreeNode(1));
        for(int i=2; i<=n; i++) {
            int len = result.size();
            for(int j=0; j<len; j++) {
                TreeNode root = result.get(j);
                TreeNode node = new TreeNode(i);
                node.left = root;
                result.set(j, node);
                DFS(root, i, root);
            }
        }
        return result;
    }
    public void DFS(TreeNode root, int val, TreeNode curr) {
        if (curr == null) return;
        TreeNode right = curr.right;
        curr.right = new TreeNode(val);
        curr.right.left = right;
        result.add(copy(root));
        curr.right = right;
        if (right != null) DFS(root, val, right);
    }
    public TreeNode copy(TreeNode root) {
        if (root == null) return null;
        TreeNode newRoot = new TreeNode(root.val);
        newRoot.left = copy(root.left);
        newRoot.right = copy(root.right);
        return newRoot;
    }
}
```

#### recursive
- time: 71.55%
- space: 21.93%

```java
class Solution {
    public List<TreeNode> generateTrees(int n) {
        if (n == 0) return new ArrayList<>();
        return generateTrees(1, n);
    }
    public List<TreeNode> generateTrees(int l, int r) {
        List<TreeNode> result = new ArrayList<>();
        if (l > r) return result;
        for(int i=l; i<=r; i++) {
            TreeNode root = new TreeNode(i);
            List<TreeNode> leftList = generateTrees(l, i-1);
            List<TreeNode> rightList = generateTrees(i+1, r);
            if (leftList.size() == 0) leftList.add(null);
            if (rightList.size() == 0) rightList.add(null);
            for (TreeNode left: leftList)
                for (TreeNode right: rightList) {
                    root.left = left;
                    root.right = right;
                    result.add(copy(root));
                }
        }
        return result;
    }
    public TreeNode copy(TreeNode root) {
        if (root == null) return null;
        TreeNode newRoot = new TreeNode(root.val);
        newRoot.left = copy(root.left);
        newRoot.right = copy(root.right);
        return newRoot;
    }
}
```

### 74. Search a 2D Matrix
- [Link](https://leetcode.com/problems/search-a-2d-matrix/)
- Tags: Array, Binary Search
- Stars: 2

#### 2-dimensional binary search
- time: 100%
- space: 7.69%
- attention: out-of-bound check for binary search

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix[0].length == 0) return false;
        int l = 0, r = matrix.length - 1;
        while(l < r) {
            int mid = l + ((r-l)>>1);
            if (matrix[mid][0] > target) r = mid - 1;
            else if (matrix[mid][0] < target) l = mid + 1;
            else l = r = mid;
        }
        int i = matrix[l][0] > target ? l-1 : l;
        if (i < 0) return false;
        int j = Arrays.binarySearch(matrix[i], 0, matrix[i].length, target);
        return j>=0;
    }
}
```

#### turning to ordinary 1-d binary search
- time: 100%
- space: 7.53%
- interviewLevel

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix[0].length == 0) return false;
        int m = matrix.length, n = matrix[0].length;
        if (target < matrix[0][0] || target > matrix[m-1][n-1]) return false;
        int l = 0, r = m*n - 1;
        while (l < r) {
            int mid = l + ((r-l)>>1);
            int i=mid/n, j=mid%n;
            if (matrix[i][j] > target) r = mid-1;
            else if (matrix[i][j] < target) l = mid+1;
            else l = r = mid;
        }
        int i=l/n, j=l%n;
        return matrix[i][j] == target;
    }
}
```

### 274. H-Index
- [Link](https://leetcode.com/problems/h-index/)
- Tags: Hash Table, Sort
- Stars: 4

#### binary search by value
- time: 60.53%
- space: 100%
- attention: out-of-bound check for binary search.

Similar to **H-Index II**

```java
class Solution {
    public int hIndex(int[] citations) {
        if (citations.length == 0) return 0;
        int max = citations[0];
        for(int citation: citations) max = Math.max(max, citation);
        if (max == 0) return 0;
        int l = 1, r = citations.length;
        while (l < r) {
            int mid = l + ((r-l)>>1);
            int count = countgte(citations, mid);
            if (mid < count) l = mid + 1;
            else if (mid > count) r = mid - 1;
            else l = r = mid;
        }
        int count = countgte(citations, l);
        if (count < l) return l - 1;
        return l;
    }
    public int countgte(int[] nums, int target) {
        int count = 0;
        for(int num: nums)
            if (num >= target) count++;
        return count;
    }
}
```

#### sort and count
- time: 60.53%
- space: 100%

sort: O(nlogn)
count: O(n)

Not applicable to **H-Index II**

```java
class Solution {
    public int hIndex(int[] citations) {
        if (citations.length == 0) return 0;
        Arrays.sort(citations);
        if (citations[citations.length - 1] == 0) return 0;
        for (int i=1; i<=citations.length; i++) {
            if (citations[citations.length - i] < i) return i-1;
        }
        return citations.length;
    }
}
```

#### Bucket sort
- time: 100%
- space: 100%
- interviewLevel

O(n) time

```java
class Solution {
    public int hIndex(int[] citations) {
        int[] buckets = new int[citations.length + 1];
        for (int c: citations) buckets[Math.min(c, citations.length)]++;
        int accumulate = 0;
        for (int i=buckets.length-1; i>=0; i--) {
            accumulate += buckets[i];
            if (accumulate >= i) return i;
        }
        return 0;
    }
}
```

### 209. Minimum Size Subarray Sum
- [Link](https://leetcode.com/problems/minimum-size-subarray-sum/)
- Tags: Array, Two Pointers, Binary Search
- Stars: 2

#### Two Pointers / Sliding Window
- time: 99.95%
- space: 34.71%
- interviewLevel

```java
class Solution {
    public int minSubArrayLen(int s, int[] nums) {
        if (nums.length == 0) return 0;
        int l = 0, r = 1, curr = nums[0], result = Integer.MAX_VALUE;
        while(l < nums.length) {
            if (curr >= s) {
                result = Math.min(result, r-l);
                curr -= nums[l++];
            }
            else {
                if (r >= nums.length) break;
                curr += nums[r++];
            }
        }
        if (result == Integer.MAX_VALUE) result = 0;
        return result;
    }
}
```

### 92. Reverse Linked List II
- [Link](https://leetcode.com/problems/reverse-linked-list-ii/)
- Tags: Linked List
- Stars: 3

#### 2019.7.30
- time: 100%
- space: 100%

```java
class Solution {
    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (head == null) return null;
        ListNode handle = new ListNode(0);
        handle.next = head;
        ListNode prefixTail = moveForward(handle, m-1),
                start = prefixTail.next,
                end = moveForward(start, n-m),
                tail = end.next;
        end.next = null;
        prefixTail.next = null;
        reverse(start);
        prefixTail.next = end;
        start.next = tail;
        return handle.next;      
    }
    public ListNode moveForward(ListNode handle, int n) {
        for(int i=0; i<n; i++)
            handle = handle.next;
        return handle;
    }
    public ListNode reverse(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode handle = new ListNode(0);
        while (head != null) {
            ListNode next = head.next;
            head.next = handle.next;
            handle.next = head;
            head = next;
        }
        return handle.next;
    }
}
```

#### 2019.9.13 Method by [大雪菜]
- time: 100%
- space: 100%
- interviewLevel
- reviewFlag
```java
class Solution {
    public ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode dmy = new ListNode(0);
        dmy.next = head;
        ListNode a = dmy, c = dmy;
        for(int i=0; i<m-1; i++) a = a.next;
        for(int i=0; i<n; i++) c = c.next;
        ListNode b = a.next, d = c.next;
        for(ListNode p=b, q=b.next; q!=d;) {
            ListNode next = q.next;
            q.next = p;
            p = q;
            q = next;
        }
        a.next = c;
        b.next = d;
        return dmy.next;
    }
}
```

### 117. Populating Next Right Pointers in Each Node II
- [Link](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)
- Tags: Tree, DFS
- Stars: 4

#### DFS
- time: 100%
- space: 96.76%
- interviewLevel

A tree like below:
```
        a
    b       c
  d  e        f
g               h
```
g.next should be h, but their parent nodes are not adjacent!

Thus, when running DFS, we need to `connect(root.right)` before `connect(root.left)`. 
Also, the `getLeftChild` method should recursively get the most left child on the same level!


```java
class Solution {
    public Node connect(Node root) {
        if (root == null) return null;
        if (root.left != null) root.left.next = root.right;
        Node child = getRightChild(root);
        if (root.next != null && child != null) child.next = getLeftChild(root.next);
        connect(root.right);
        connect(root.left);
        return root;
    }
    public Node getRightChild(Node root) {
        return root.right == null ? root.left : root.right;
    }
    public Node getLeftChild(Node root) {
        if (root == null) return null;
        if (root.left != null) return root.left;
        if (root.right != null) return root.right;
        return getLeftChild(root.next);
    }
}
```

### 63. Unique Paths II
- [Link](https://leetcode.com/problems/unique-paths-ii/)
- Tags: Array, Dynamic Programming
- Stars: 1

#### DP
- time: 100%
- space: 88.51%

```java
class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid.length == 0 || obstacleGrid[0].length == 0) return 0;
        int m = obstacleGrid.length, n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = obstacleGrid[0][0] == 1 ? 0 : 1;
        for(int j=1; j<n; j++)
            dp[0][j] = obstacleGrid[0][j] == 1 ? 0 : dp[0][j-1];
        for(int i=1; i<m; i++)
            dp[i][0] = obstacleGrid[i][0] == 1 ? 0 : dp[i-1][0];
        for(int i=1; i<m; i++)
            for(int j=1; j<n; j++)
                dp[i][j] = obstacleGrid[i][j] == 1 ? 0 : dp[i-1][j] + dp[i][j-1];
        return dp[m-1][n-1];
    }
}
```

#### DP with space optimization
- time: 100%
- space: 62.23%

```java
class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid.length == 0 || obstacleGrid[0].length == 0) return 0;
        int m = obstacleGrid.length, n = obstacleGrid[0].length;
        int[] dp = new int[n];
        dp[0] = obstacleGrid[0][0] == 1 ? 0 : 1;
        for(int j=1; j<n; j++) dp[j] = obstacleGrid[0][j] == 1 ? 0 : dp[j-1];
        for(int i=1; i<m; i++) {
            dp[0] = obstacleGrid[i][0] == 1 ? 0 : dp[0];
            for(int j=1; j<n; j++) 
                dp[j] = obstacleGrid[i][j] == 1 ? 0 : dp[j-1] + dp[j];
        }
        return dp[n-1];
    }
}
```

### 60. Permutation Sequence
- [Link](https://leetcode.com/problems/permutation-sequence/)
- Tags: Math, Backtracking
- Stars: 3

#### compute digit by digit
- time: 99.49%
- space: 100%

```java
class Solution {
    public String getPermutation(int n, int k) {
        k--;
        StringBuilder sb = new StringBuilder();
        List<Integer> list = new ArrayList<>();
        for(int i=1; i<=9; i++) list.add(i);
        for(int i=n-1; i>=1; i--) {
            int idx = k / factor(i);
            int val = popIndexAndSort(list, idx);
            sb.append(val);
            k %= factor(i);
        }
        sb.append(list.get(0));
        return sb.toString();
    }
    public int factor(int n) {
        if (n == 0) return 1;
        return n * factor(n-1);
    }
    public int popIndexAndSort(List<Integer> list, int idx) {
        int ret = list.get(idx);
        list.remove(idx);
        return ret;
    }
}
```

### 151. Reverse Words in a String
- [Link](https://leetcode.com/problems/reverse-words-in-a-string/)
- Tags: String
- Stars: 1

#### 2019.7.31
- time: 72.51%
- space: 84.49%
```java
class Solution {
    public String reverseWords(String s) {
        int l = skipWhitespaces(s, 0);
        if (l == s.length()) return "";
        int r = l+1;
        List<String> list = new ArrayList<>();
        while(r < s.length()) {
            if (s.charAt(r) == ' ') {
                list.add(s.substring(l, r));
                l = skipWhitespaces(s, r+1);
                r = l+1;
            } else r++;
        }
        if (l < s.length()) list.add(s.substring(l, s.length()));
        StringBuilder sb = new StringBuilder();
        for(int i=list.size()-1; i>=0; i--) {
            sb.append(list.get(i));
            if (i > 0) sb.append(' ');
        }
        return sb.toString();
    }
    public int skipWhitespaces(String s, int start) {
        if (start >= s.length()) return s.length();
        while(start < s.length() && s.charAt(start) == ' ') start++;
        return start;
    }
}
```

#### 2019.9.14 double reverse method [大雪菜]
- time: 84.35%
- space: 64.52%
```java
class Solution {
    public String reverseWords(String s) {
        StringBuilder sb = new StringBuilder();
        char[] chrs = s.toCharArray();
        reverse(chrs, 0, chrs.length-1);
        for(int i=0; i<chrs.length; i++) {
            if (chrs[i] == ' ') continue;
            int j = i;
            while(j<chrs.length && chrs[j] != ' ') j++;
            reverse(chrs, i, j-1);
            if (sb.length() > 0) sb.append(' ');
            for(int k=i; k<j; k++) {
                sb.append(chrs[k]);
            }
            i = j;
        }
        return sb.toString();
    }
    public void reverse(char[] chrs, int i, int j) {
        while(i < j) {
            char c = chrs[i];
            chrs[i] = chrs[j];
            chrs[j] = c;
            i++;
            j--;
        }
    }
}
```

### 61. Rotate List
- [Link](https://leetcode.com/problems/rotate-list/)
- Tags: Linked List, Two Pointers
- Stars: 2

#### 2019.8.4
- time: 29.76%
- space: 68.73%

```java
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) return null;
        int len = countLen(head);
        k %= len;
        if (k == 0) return head;
        ListNode curr = head;
        for(int i=0; i<len-k-1; i++) curr = curr.next;
        ListNode newHead = curr.next, tail = moveToTail(newHead);
        curr.next = null;
        tail.next = head;
        return newHead;
    }
    public int countLen(ListNode head) {
        int count = 0;
        while (head != null) {
            head = head.next;
            count++;
        }
        return count;
    }
    public ListNode moveToTail(ListNode head) {
        if (head == null) return null;
        while (head.next != null) head = head.next;
        return head;
    }
}
```

Updated (Concise Version)

```java
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) return null;
        int count = 1;
        ListNode tail = head;
        while (tail.next != null) {
            count++;
            tail = tail.next;
        }
        tail.next = head;
        k %= count;
        for(int i=0; i<count-k; i++) tail = tail.next;
        ListNode newHead = tail.next;
        tail.next = null;
        return newHead;
    }
}
```

### 229. Majority Element II
- [Link](https://leetcode.com/problems/majority-element-ii/)
- Tags: Array
- Stars: 5

#### Boyer-Moore Majority Vote Algorithm
- time: 74.98%
- space: 100%

Reference:
https://gregable.com/2013/10/majority-vote-algorithm-find-majority.html  
https://leetcode.com/problems/majority-element-ii/discuss/63537/My-understanding-of-Boyer-Moore-Majority-Vote  

```java
class Solution {
    public List<Integer> majorityElement(int[] nums) {
        List<Integer> result = new ArrayList<>();
        if (nums.length == 0) return result;
        int cand1 = nums[0], cand2 = 0, count1 = 1, count2 = 0;
        for(int num: nums) {
            if (count1 == 0) {
                cand1 = num;
                count1++;
            } else if (num == cand1) {
                count1++;
            } else if (count2 == 0) {
                cand2 = num;
                count2++;
            } else if (num == cand2) {
                count2++;
            } else {
                count1--;
                count2--;
            }
        }
        count1 = count2 = 0;
        for(int num: nums) {
            if (num == cand1) count1++;
            if (num == cand2) count2++;
        }
        if (count1 > nums.length/3) result.add(cand1);
        if (count2 > nums.length/3 && cand1 != cand2) result.add(cand2);
        return result;
    }
}
```

### 6. ZigZag Conversion
- [Link](https://leetcode.com/problems/zigzag-conversion/)
- Tags: String
- Stars: 2

#### 2019.8.11
- time: 31.06%
- space: 89.36%
```java
class Solution {
    public String convert(String s, int numRows) {
        if (s.length() == 0 || numRows == 1) return s;
        int[] rows = new int[s.length()];
        int incre = 1, i = 1;
        while(i < rows.length) {
            rows[i] = rows[i-1] + incre;
            if (rows[i] == numRows-1) incre = -1;
            else if (rows[i] == 0) incre = 1;
            i++;
        }
        StringBuilder sb = new StringBuilder();
        for(i=0; i<numRows; i++) 
            for (int j=0; j<rows.length; j++) if (rows[j] == i) sb.append(s.charAt(j));
        return sb.toString();
    }
}
```

Updated

```java
class Solution {
    public String convert(String s, int numRows) {
        if (s.length() == 0 || numRows == 1) return s;
        StringBuilder[] sbs = new StringBuilder[numRows];
        sbs[0] = new StringBuilder();
        sbs[0].append(s.charAt(0));
        int incre = 1, i = 1, currRow = 0;
        while(i<s.length()) {
            currRow += incre;
            if (sbs[currRow] == null) sbs[currRow] = new StringBuilder();
            sbs[currRow].append(s.charAt(i));
            i++;
            if (currRow == 0) incre = 1;
            else if (currRow == numRows-1) incre = -1;
        }
        for(int k=1; k<numRows; k++) {
            if (sbs[k] == null) break;
            sbs[0].append(sbs[k]);
        }
        return sbs[0].toString();
    }
}
```

#### 2019.9.14 [大雪菜]
- time: 96.04%
- space: 92.55%
- cheatFlag
```java
class Solution {
    public String convert(String s, int n) {
        if (n == 1) return s;
        StringBuilder sb = new StringBuilder();
        int d = 2*(n-1), len = s.length();
        for(int i=0; i<len; i+=d) sb.append(s.charAt(i));
        for(int i=1; i<n-1; i++) {
            for(int j=i, k=d-i; j<len || k<len; j+=d, k+=d) {
                if (j<len) sb.append(s.charAt(j));
                if (k<len) sb.append(s.charAt(k));
            }
        }
        for(int i=n-1; i<len; i+=d) sb.append(s.charAt(i));
        return sb.toString();
    }
}
```

### 93. Restore IP Addresses
- [Link](https://leetcode.com/problems/restore-ip-addresses/)
- Tags: String, Backtracking
- Stars: 3

#### 2019.8.11 backtrack
- time: 27.03%
- space: 100%

```java
class Solution {
    List<String> result = new ArrayList<>();
    public List<String> restoreIpAddresses(String s) {
        if (s.length() == 0) return result;
        backtrack(s, 0, new ArrayList<>());
        return result;
    }
    public void backtrack(String s, int start, List<String> currList) {
        if (currList.size() == 4) {
            if (start < s.length()) return;
            result.add(String.join(".", currList));
            return;
        }
        if (start >= s.length()) return;
        if (s.charAt(start) == '0') {
            currList.add("0");
            backtrack(s, start+1, currList);
            currList.remove(currList.size() - 1);
            return;
        }
        int end = start+1;
        while(end<=s.length() && 
              (end-start<3 || Integer.parseInt(s.substring(start, end))<256)) {
            currList.add(s.substring(start, end));
            backtrack(s, end, currList);
            currList.remove(currList.size() - 1);
            end++;
        }
    }
}
```

#### 2019.8.11 
- time: 90.49%
- space: 100%

```java
class Solution {
    List<String> result = new ArrayList<>();
    public List<String> restoreIpAddresses(String s) {
        if (s.length() == 0) return result;
        for(int a=1; a<=3; a++)
            for(int b=1; b<=3; b++)
                for(int c=1; c<=3; c++)
                    for(int d=1; d<=3; d++){
                        if (a+b+c+d != s.length()) continue;
                        int A=Integer.parseInt(s.substring(0, a));
                        int B=Integer.parseInt(s.substring(a, a+b));
                        int C=Integer.parseInt(s.substring(a+b, a+b+c));
                        int D=Integer.parseInt(s.substring(a+b+c, a+b+c+d));
                        if (A<256 && B<256 && C<256 && D<256) {
                            String ans = A+"."+B+"."+C+"."+D;
                            if (ans.length() == s.length() + 3) 
                                result.add(ans);
                        }
                    }
        return result;
    }
}
```

### 31. Next Permutation
- [Link](https://leetcode.com/problems/next-permutation/)
- Tags: Array
- Stars: 4

#### 2019.8.11
- time: 90.57%
- space: 34%

The function `getNextEleIdx` is to find the smallest element in the subarray `nums[start:]` that is greater than `target`.

```java
class Solution {
    public void nextPermutation(int[] nums) {
        if (nums.length == 0) return;
        for(int i=nums.length-2; i>=0; i--) {
            if (nums[i] < nums[i+1]) {
                int idx = getNextEleIdx(nums, i+1, nums[i]);
                swap(nums, i, idx);
                Arrays.sort(nums, i+1, nums.length);
                return;
            }
        }
        Arrays.sort(nums);
    }
    public int getNextEleIdx(int[] nums, int start, int target) {
        int idx = start, val = nums[start];
        for(int i=start; i<nums.length; i++) 
            if (nums[i] > target && nums[i] < val) {
                idx = i;
                val = nums[i];
            }
        return idx;
    }
    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

#### 2019.8.11 Narayana Pandita's algorithm
- time: 90.57%
- space: 30%

Reference: https://leetcode.com/problems/next-permutation/discuss/13867/C%2B%2B-from-Wikipedia

Similar to the solution above. 
The difference is: when calling `getNextEleIdx` function, the subarray `nums[start:]` is a decreasing array.
Therefore, we can simplify the function by just iterate the subarray from the last element to `start`.
Also, we no longer need to `Arrays.sort(nums, i+1, nums.length)`. 
Instead, we can simply reverse the subarray, since it's a decreasing array even after `swap(nums, i, idx)`.

```java
class Solution {
    public void nextPermutation(int[] nums) {
        if (nums.length == 0) return;
        for(int i=nums.length-2; i>=0; i--) {
            if (nums[i] < nums[i+1]) {
                int idx = getNextEleIdx(nums, i+1, nums[i]);
                swap(nums, i, idx);
                reverse(nums, i+1, nums.length);
                return;
            }
        }
        Arrays.sort(nums);
    }
    public int getNextEleIdx(int[] nums, int start, int target) {
        int i=nums.length-1;
        for(; i>=start; i--) if (nums[i] > target) return i;
        return i;
    }
    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    public void reverse(int[] nums, int l, int r) {
        r--;
        while(l < r) swap(nums, l++, r--);
    }
}
```

### 143. Reorder List
- [Link](https://leetcode.com/problems/reorder-list/)
- Tags: Linked List
- Stars: 3

#### 2019.8.11
- time: 100%
- space: 100%
- attention: `l = head.next` should be operated after `slow.next = null`. Consider the case: `[1,2]` as input
- attention: `tail` is a handle that represents for current tail. When we set `tail = head`, we could not initiate `l` as `head`. Instead, `l = head.next`.

```java
class Solution {
    public void reorderList(ListNode head) {
        if (head == null || head.next == null) return;
        ListNode slow = head, fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode r = reverse(slow.next), l, tail = head;
        slow.next = null;
        l = head.next;
        while(l != null && r != null) {
            tail.next = r;
            ListNode rNext = r.next;
            r.next = l;
            tail = l;
            l = l.next;
            r = rNext;
        }
        tail.next = null;
        if (l != null) tail.next = l;
        if (r != null) tail.next = r;
    }
    public ListNode reverse(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode newHead = new ListNode(0);
        while(head != null) {
            ListNode next = head.next;
            head.next = newHead.next;
            newHead.next = head;
            head = next;
        }
        return newHead.next;
    }
}
```

### 18. 4Sum
- [Link](https://leetcode.com/problems/4sum/)
- Tags: Array, Hash Table, Two Pointers
- Stars: 2

#### 2019.8.11 O(n^3) Two Pointers
- time: 14.3%
- space: 100%

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> fourSum(int[] nums, int target) {
        if (nums.length < 4) return result;
        Arrays.sort(nums);
        for(int i=0; i<nums.length-3; i++) {
            if (i>0 && nums[i] == nums[i-1]) continue;
            int a = nums[i];
            for(int j=i+1; j<nums.length-2; j++) {
                if (j>i+1 && nums[j] == nums[j-1]) continue;
                int b = nums[j];
                int l = j+1, r = nums.length - 1;
                while(l < r) {
                    int sum = a + b + nums[l] + nums[r];
                    if (sum < target) do {l++;} while (l<r && nums[l] == nums[l-1]);
                    else if (sum > target) do {r--;} while (l<r && nums[r] == nums[r+1]);
                    else {
                        result.add(Arrays.asList(a, b, nums[l], nums[r]));
                        do {l++;} while (l<r && nums[l] == nums[l-1]);
                        do {r--;} while (l<r && nums[r] == nums[r+1]);
                    }
                }
            }
        }
        return result;
    }
}
```

### 71. Simplify Path
- [Link](https://leetcode.com/problems/simplify-path/)
- Tags: String, Stack
- Stars: 2

#### 2019.8.12
- time: 66.39%
- space: 86.67%
- attention: `if (A && B) ... else ...` is different from `if (A) { if (B) ... } else ...`.

```java
class Solution {
    public String simplifyPath(String path) {
        if(path.length() == 0) return "";
        String[] dirs = path.split("/");
        List<String> list = new ArrayList<>();
        for(String dir: dirs) {
            if (dir.length() == 0 || dir.equals(".")) continue;
            else if (dir.equals("..")) {
                if (list.size() > 0) list.remove(list.size()-1);
            }
            else list.add(dir);
        }
        return "/" + String.join("/", list);
    }
}
```

### 133. Clone Graph
- [Link](https://leetcode.com/problems/clone-graph/)
- Tags: DFS, BFS, Graph
- Stars: 2

#### 2019.8.12 HashMap
- time: 100%
- space: 97.65%

```java
class Solution {
    public Node cloneGraph(Node node) {
        Map<Node, Node> map = new HashMap<>();
        addNode(map, node);
        for(Node newNode: map.values()) {
            List<Node> neighbors = new ArrayList<>();
            for(Node src: newNode.neighbors) 
                neighbors.add(map.get(src));
            newNode.neighbors = neighbors;
        }
        return map.get(node);
    }
    public void addNode(Map<Node, Node> map, Node node) {
        if (map.containsKey(node)) return;
        map.put(node, new Node(node.val, node.neighbors));
        for(Node n: node.neighbors) addNode(map, n);
    }
}
```

Another version

```java
class Solution {
    Map<Node, Node> map = new HashMap<>();
    public Node cloneGraph(Node node) {
        if (node == null) return null;
        if (map.containsKey(node)) return map.get(node);
        Node newNode = new Node(node.val, new ArrayList<>());
        map.put(node, newNode);
        for(Node nb: node.neighbors) newNode.neighbors.add(cloneGraph(nb));
        return newNode;
    }
}
```

### 165. Compare Version Numbers
- [Link](https://leetcode.com/problems/compare-version-numbers/)
- Tags: String
- Stars: 3

#### 2019.8.12
- time: 90.76%
- space: 100%
- attention: The param for `String.split` is a regex. Thus, when splitting by `.`, pass in `"\\."`.

```java
class Solution {
    public int compareVersion(String version1, String version2) {
        int[] v1 = split(version1), v2 = split(version2);
        int i=0;
        while(i<v1.length || i<v2.length) {
            int num1 = i<v1.length ? v1[i] : 0;
            int num2 = i<v2.length ? v2[i] : 0;
            if (num1 > num2) return 1;
            else if (num1 < num2) return -1;
            i++;
        }
        return 0;
    }
    public int[] split(String s) {
        List<Integer> list = new ArrayList<>();
        int start = 0, end = 0;
        while(start < s.length()) {
            if (end < s.length() && s.charAt(end) != '.') end++;
            else {
                list.add(Integer.parseInt(s.substring(start, end)));
                start = ++end;
            }
        }
        int[] result = new int[list.size()];
        for(int i=0; i<list.size(); i++) result[i] = list.get(i);
        return result;
    }
}
```

Updated

```java
class Solution {
    public int compareVersion(String version1, String version2) {
        String[] v1 = version1.split("\\.");
        String[] v2 = version2.split("\\.");
        int i = 0;
        while(i<v1.length || i<v2.length) {
            int num1 = i<v1.length ? Integer.parseInt(v1[i]) : 0;
            int num2 = i<v2.length ? Integer.parseInt(v2[i]) : 0;
            if (num1 < num2) return -1;
            else if (num1 > num2) return 1;
            i++;
        }
        return 0;
    }
}
```

#### 2019.9.14
- time: 100%
- space: 100%
- attention: use `i<len1 || j<len2` instead of `i<len1 && j<len2`.
```java
class Solution {
    public int compareVersion(String s1, String s2) {
        int i=0, j=0, len1 = s1.length(), len2 = s2.length();
        while(i<len1 || j<len2) {
            int p = i, q = j;
            while(p<len1 && s1.charAt(p) != '.') p++;
            while(q<len2 && s2.charAt(q) != '.') q++;
            int res = compare(s1, i, p, s2, j, q);
            if (res != 0) return res;
            i = p+1;
            j = q+1;
        }
        return 0;
    }
    public int compare(String s1, int i, int p, String s2, int j, int q) {
        int a=0, b=0;
        for(int k=i; k<p; k++) {
            a *= 10;
            a += s1.charAt(k) - '0';
        }
        for(int k=j; k<q; k++) {
            b *= 10;
            b += s2.charAt(k) - '0';
        }
        if (a<b) return -1;
        else if (a>b) return 1;
        return 0;
    }
}
```

### 220. Contains Duplicate III
- [Link](https://leetcode.com/problems/contains-duplicate-iii/)
- Tags: Sort, OrderedMap
- Stars: 5

#### 2019.8.12 TreeMap
- time: 24.81%
- space: 97.73%
- attention: If you want to maintain a queue (push/pop) while performing `O(logN)` search operations, try `TreeMap`!
- attention: while computing distances of two elements, pay attention to the overflow problem. 
- language: the use of `TreeMap` and `TreeSet`

```java
class Solution {
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        TreeMap<Integer, Integer> map = new TreeMap<>();
        for(int i=0; i<nums.length; i++) {
            Integer l = map.floorKey(nums[i]), r = map.ceilingKey(nums[i]);
            if (l != null && Math.abs((long)l-(long)nums[i])<=(long)t) return true;
            if (r != null && Math.abs((long)r-(long)nums[i])<=(long)t) return true;
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
            if (i>=k) {
                map.put(nums[i-k], map.get(nums[i-k]) - 1);
                if (map.get(nums[i-k]) == 0) map.remove(nums[i-k]);
            }
        }
        return false;
    }
}
```

### 51. N-Queens
- [Link](https://leetcode.com/problems/n-queens/)
- Tags: Backtracking
- Stars: 3

#### 2019.8.17 backtrack with set marking
- time: 7.08%
- space: 10.81%

```java
class Solution {
    List<List<String>> result = new ArrayList<>();
    public List<List<String>> solveNQueens(int n) {
        if (n == 0) return result;
        char[][] board = new char[n][n];
        for(int i=0; i<n; i++) Arrays.fill(board[i], '.');
        backtrack(board, new HashSet<>(), 0);
        return result;
    }
    public void backtrack(char[][] board, Set<String> set, int i) {
        if (i == board.length) {
            result.add(convert(board));
            return;
        }
        for(int j=0; j<board[i].length; j++) {
            String vertical = j + "v";
            if (!set.add(vertical)) continue;
            String left = (j-i) + "l";
            if (!set.add(left)) {
                set.remove(vertical);
                continue;
            }
            String right = (j+i) + "r";
            if (!set.add(right)) {
                set.remove(vertical);
                set.remove(left);
                continue;
            }
            board[i][j] = 'Q';
            backtrack(board, set, i+1);
            board[i][j] = '.';
            set.remove(vertical); set.remove(left); set.remove(right);
        }
    }
    public List<String> convert(char[][] board) {
        List<String> result = new ArrayList<>();
        for(char[] row: board) result.add(new String(row));
        return result;
    }
}
```

#### 2019.8.17 backtrack with array marking
- time: 95.29%
- space: 100%
- language: convert a `char[]` into `String`

```java
class Solution {
    List<List<String>> result = new ArrayList<>();
    boolean[] vertical, left, right;
    int n;
    public List<List<String>> solveNQueens(int n) {
        if (n == 0) return result;
        this.n = n;
        vertical = new boolean[n]; 
        left = new boolean[2*n-1]; 
        right = new boolean[2*n-1];
        char[][] board = new char[n][n];
        for(int i=0; i<n; i++) Arrays.fill(board[i], '.');
        backtrack(board, 0);
        return result;
    }
    public void backtrack(char[][] board, int i) {
        if (i == n) {
            result.add(convert(board));
            return;
        }
        for(int j=0; j<n; j++) {
            if (vertical[j] || left[j-i+n-1] || right[j+i]) continue;
            vertical[j] = left[j-i+n-1] = right[j+i] = true;
            board[i][j] = 'Q';
            backtrack(board, i+1);
            board[i][j] = '.';
            vertical[j] = left[j-i+n-1] = right[j+i] = false;
        }
    }
    public List<String> convert(char[][] board) {
        List<String> result = new ArrayList<>();
        for(char[] row: board) result.add(new String(row));
        return result;
    }
}
```

### 52. N-Queens II
- [Link](https://leetcode.com/problems/n-queens-ii/)
- Tags: Backtracking
- Stars: 4

#### 2019.8.17 backtrack with array marking
- time: 95.79%
- space: 8.70%

```java
class Solution {
    int result = 0;
    int n;
    boolean[] vertical, left, right;
    public int totalNQueens(int n) {
        if (n == 0) return 0;
        this.n = n;
        vertical = new boolean[n]; 
        left = new boolean[2*n-1]; 
        right = new boolean[2*n-1];
        backtrack(0);
        return result;
    }
    private void backtrack(int i) {
        if (i == n) {
            result++;
            return ;
        }
        for(int j=0; j<n; j++) {
            if (vertical[j] || left[j-i+n-1] || right[j+i]) continue;
            vertical[j] = left[j-i+n-1] = right[j+i] = true;
            backtrack(i+1);
            vertical[j] = left[j-i+n-1] = right[j+i] = false;
        }
    }
}
```

### 145. Binary Tree Postorder Traversal
- [Link](https://leetcode.com/problems/binary-tree-postorder-traversal/)
- Tags: Stack, Tree
- Stars: 3
- reviewFlag

#### 2019.8.17 iterative Stack<Pair>
- time: 63.78%
- space: 100%
- attention: when pushing items into the stack, make sure to push the right child first.

```java
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        Stack<Pair> st = new Stack<>();
        st.add(new Pair(root));
        while(!st.isEmpty()) {
            Pair p = st.pop();
            if (p.visited) {
                result.add(p.node.val);
            } else {
                p.visited = true;
                st.add(p);
                if (p.node.right != null) st.add(new Pair(p.node.right));
                if (p.node.left != null) st.add(new Pair(p.node.left));
            }
        }
        return result;
    }
    
    public class Pair {
        TreeNode node;
        boolean visited;
        Pair(TreeNode n) {
            node = n;
            visited = false;
        }
    }
}
```

#### 2019.8.17 iterative Stack<TreeNode>
- time: 63.78%
- space: 100%

```java
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        Stack<TreeNode> st = new Stack<>();
        TreeNode curr = root;
        while(true) {
            while(curr.left != null) {
                st.add(curr);
                curr = curr.left;
            }
            if (curr.right != null) {
                st.add(curr);
                curr = curr.right;
                continue;
            } 
            while(!st.isEmpty() && 
                  (curr == st.peek().right || st.peek().right == null)) {
                result.add(curr.val);
                curr = st.pop();
            }
            result.add(curr.val);
            if (st.isEmpty()) break;
            else curr = st.peek().right;
        }
        return result;
    }
}
```

Updated 2019.9.3 addUntilLeftLeaf
- time: 64.84%
- space: 100%
```java
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> ret = new LinkedList<>();
        if (root == null) return ret;
        Stack<TreeNode> st = new Stack<>();
        addUntilLeftLeaf(st, root);
        while(!st.isEmpty()) {
            TreeNode node = st.pop();
            ret.add(node.val);
            if (!st.isEmpty() && st.peek().left == node) 
                addUntilLeftLeaf(st, st.peek().right);
        }
        return ret;
    }
    public void addUntilLeftLeaf(Stack<TreeNode> st, TreeNode curr) {
        if (curr == null) return;
        while(curr.left != null || curr.right != null) {
            while(curr.left != null) {
                st.add(curr);
                curr = curr.left;
            }
            if (curr.right != null) {
                st.add(curr);
                curr = curr.right;
            }
        }
        st.add(curr);
    }
}
```

#### 2019.9.3 reverse preorder
- time: 64.84%
- space: 100%
```java
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> ret = new LinkedList<>();
        if (root == null) return ret;
        Stack<TreeNode> st = new Stack<>();
        st.add(root);
        while(!st.isEmpty()) {
            TreeNode node = st.pop();
            ret.add(0, node.val);
            if (node.left != null) st.add(node.left);
            if (node.right != null) st.add(node.right);
        }
        return ret;
    }
}
```

### 154. Find Minimum in Rotated Sorted Array II
- [Link](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/)
- Tags: Array, Binary Search
- Stars: 4

#### 2019.8.18 clip (linear speed) + binary search
- time: 100%
- space: 6.25%
- attention: you have to compare the result of binary search with the clipped elements and return `Math.min(nums[i], nums[0])`. (Consider the case `[3,1,3]` and `[0,1,2,3,0]`)

```java
class Solution {
    public int findMin(int[] nums) {
        int i = 0, j = nums.length - 1;
        while(i<j && nums[i] == nums[j]) {i++; j--;}
        if (i>=j) return Math.min(nums[0], nums[i]);
        if (nums[i] < nums[j]) return Math.min(nums[i], nums[0]);
        int l = i, r = j;
        while(l<r) {
            int mid = l + ((r-l)>>1);
            if (nums[mid] >= nums[i]) l = mid + 1;
            else r = mid;
        }
        return nums[l];
    }
}
```

### 72. Edit Distance
- [Link](https://leetcode.com/problems/edit-distance/)
- Tags: String, Dynamic Programming
- Stars: 4

#### 2019.8.19 DP
- time: 94.96%
- space: 100%
- attention: You can add additional row and column to make it easier. 

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int len1 = word1.length(), len2 = word2.length();
        if (len1 == 0 || len2 == 0) return Math.max(len1, len2);
        int[][] dp = new int[len1+1][len2+1];
        for(int i=0; i<=len1; i++) dp[i][0] = i;
        for(int j=0; j<=len2; j++) dp[0][j] = j;
        for(int i=1; i<=len1; i++)
            for(int j=1; j<=len2; j++) {
                if (word1.charAt(i-1) == word2.charAt(j-1)) dp[i][j] = dp[i-1][j-1];
                else dp[i][j] = Math.min(dp[i-1][j-1], Math.min(dp[i-1][j], dp[i][j-1])) + 1;
            }
        return dp[len1][len2];
    }
}
```

### 37. Sudoku Solver
- [Link](https://leetcode.com/problems/sudoku-solver/)
- Tags: Hash Table, Backtracking
- Stars: 4

#### 2019.8.19 DFS
- time: 89.96%
- space: 70.18%
- attention: It may not exist that an empty cell has only one option (1-9) available. Therefore, we have to consider DFS solutions.

```java
class Solution {
    boolean[][] rowMarks = new boolean[9][9];
    boolean[][] colMarks = new boolean[9][9];
    boolean[][][] gridMarks = new boolean[3][3][9];
    public void solveSudoku(char[][] board) {
        for(int i=0; i<9; i++)
            for(int j=0; j<9; j++)
                if (board[i][j] != '.') markAs(i, j, board[i][j], true);
        DFS(board, 0, 0);
    }
    public boolean DFS(char[][] board, int r, int c) {
        if (r == 9) return true;
        if (c == 9) return DFS(board, r+1, 0);
        if (board[r][c] != '.') return DFS(board, r, c+1);
        for(int idx=0; idx<9; idx++) {
            if (rowMarks[r][idx] || colMarks[c][idx] || gridMarks[r/3][c/3][idx]) continue;
            board[r][c] = (char)(idx+'1');
            markAs(r, c, board[r][c], true);
            if (DFS(board, r, c+1)) return true;
            markAs(r, c, board[r][c], false);
            board[r][c] = '.';
        }
        return false;
    }
    public void markAs(int i, int j, char c, boolean flag) {
        int idx = c - '1';
        rowMarks[i][idx] = flag;
        colMarks[j][idx] = flag;
        gridMarks[i/3][j/3][idx] = flag;
    }
}
```

### 25. Reverse Nodes in k-Group
- [Link](https://leetcode.com/problems/reverse-nodes-in-k-group/)
- Tags: Linked List
- Stars: 3

#### 2019.8.19 recursive
- time: 100%
- space: 25.86%

```java
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null) return head;
        ListNode tail = head;
        for(int i=1; i<k; i++) {
            if (tail.next == null) return head;
            tail = tail.next;
        }
        ListNode remain = tail.next, curr = head;
        while(curr != tail) {
            ListNode next = curr.next;
            curr.next = tail.next;
            tail.next = curr;
            curr = next;
        }
        head.next = reverseKGroup(remain, k);
        return tail;
    }
}
```

#### 2019.8.19 iterative
- time: 36.90%
- space: 24.14%
- attention: for Linked List problems, avoid using the handle node.

```java
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode prefix = null, result = head;
        while(true) {
            if (head == null) break;
            ListNode tail = head;
            boolean stop = false;
            for(int i=1; i<k; i++) {
                if(tail.next == null) {
                    stop = true; 
                    break;
                } else tail = tail.next;
            }
            if (stop) break;
            if (prefix == null) result = tail;
            else prefix.next = tail;
            prefix = head;
            ListNode remain = tail.next, curr = head;
            while(curr != tail) {
                ListNode next = curr.next;
                curr.next = tail.next;
                tail.next = curr;
                curr = next;
            }
            head.next = remain;
            head = remain;
        }
        return result;
    }
}
```

Updated
```java
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        int n = 0;
        for(ListNode curr=head; curr!=null; curr=curr.next) n++;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        for(ListNode prefix=dummy, tail=head; n>=k; n-=k) {
            for(int i=0; i<k; i++) {
                ListNode next = tail.next;
                tail.next = prefix.next;
                prefix.next = tail;
                tail = next;
            }
            head.next = tail;
            prefix = head;
            head = tail = prefix.next;
        }
        return dummy.next;
    }
}
```

### 115. Distinct Subsequences
- [Link](https://leetcode.com/problems/distinct-subsequences/)
- Tags: String, Dynamic Programming
- Stars: 4

#### 2019.8.19 DP
- time: 5.84%
- space: 88.46%

```java
class Solution {
    public int numDistinct(String s, String t) {
        if (s.length() < t.length()) return 0;
        if (t.length() == 0) return 1;
        int len1 = s.length(), len2 = t.length();
        int[][] dp = new int[len1][len2];
        dp[0][0] = s.charAt(0) == t.charAt(0) ? 1 : 0;
        for(int i=1; i<len1; i++) {
            dp[i][0] = s.charAt(i) == t.charAt(0) ? dp[i-1][0] + 1 : dp[i-1][0];
            for(int j=1; j<len2 && j<=i; j++) {
                dp[i][j] += dp[i-1][j];
                if (s.charAt(i) == t.charAt(j)) dp[i][j] += dp[i-1][j-1];
            }
        }
        return dp[len1-1][len2-1];
    }
}
```

### 99. Recover Binary Search Tree
- [Link](https://leetcode.com/problems/recover-binary-search-tree/)
- Tags: Tree, DFS
- Stars: 5

#### 2019.8.19 
- time: 99.56%
- space: 100%

Consider the input tree as an ordered list (array) in which only two elements are mistakenly swapped. 
If we define the reverse-ordered node as the node whose value is smaller than its previous node's value, 
then we can find that there is at least one reverse-ordered node in the input. 
Once we find the last reverse-ordered node (marked as `A`), we know this is one of two nodes that are engaged in the final swap operation, and the other node `B` must have a greater value and is the first node that is greater than `A` in the input.

```java
class Solution {
    TreeNode last = null;
    public void recoverTree(TreeNode root) {
        if (root == null) return;
        // traverse to get the last reverse-ordered node
        TreeNode node = getLastReverseOrderNode(root);
        traverse(root, node);
    }
    public TreeNode getLastReverseOrderNode(TreeNode root) {
        if (root == null) return null;
        TreeNode ret = getLastReverseOrderNode(root.left);
        if (last != null && last.val >= root.val) ret = root;
        else last = root;
        TreeNode right = getLastReverseOrderNode(root.right);
        return right != null ? right : ret;
    }
    public boolean traverse(TreeNode root, TreeNode node) {
        if (root == null) return false;
        if (traverse(root.left, node)) return true;
        if (node.val <= root.val) {
            swap(node, root);
            return true;
        }
        return traverse(root.right, node);
    }
    public void swap(TreeNode a, TreeNode b) {
        int temp = a.val;
        a.val = b.val;
        b.val = temp;
    }
}
```

### 292. Nim Game
- [Link](https://leetcode.com/problems/nim-game/)
- Tags: Brainteaser, Minimax
- Stars: 2

#### Math solution
1. n might be very big. Thus, DP solutions doesn't work (for the reason that O(n) time is too slow). 
2. try to write down several answers for small n:  
`  n = 1 2 3 4 5 6 7 8 9 ...`  
`ans = t t t f t t t f t ...`  
We can see that the answers repeated in a `t t t f` pattern. Actually, the DP formula is  
`dp[i] = !dp[i-1] || !dp[i-2] || !dp[i-3]`. 

```java
class Solution {
    public boolean canWinNim(int n) {
        return n%4!=0;
    }
}
```

### 258. Add Digits
- [Link](https://leetcode.com/problems/add-digits/)
- Tags: Math
- Stars: 3

#### Math Solution
For num > 0, the possible answer is 1,2,3,4,5,6,7,8,9. These answers occur periodically as num increases. 

**For this kinds of questions, if you don't have any idea at first glance, try to write several answers for simple input cases to see if you can find something useful.** 

```java
class Solution {
    public int addDigits(int num) {
        if(num == 0) return 0;
        return num%9 == 0 ? 9 : num%9;
    }
}
```

### 171. Excel Sheet Column Number
- [Link](https://leetcode.com/problems/excel-sheet-column-number/)
- Tags: Math
- Stars: 1

#### O(n) time
```java
class Solution {
    public int titleToNumber(String s) {
        int result = 0;
        for(char c : s.toCharArray()){
            result *= 26;
            result += c - 'A' + 1;
        }
        return result;
    }
}
```

### 108. Convert Sorted Array to Binary Search Tree
- [Link](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)
- Tags: Tree, DFS
- Stars: 1

#### DFS
```java
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBST(nums, 0, nums.length-1);
    }
    public TreeNode sortedArrayToBST(int[] nums, int l, int r) {
        if(l > r) return null;
        int mid = l + ((r-l)>>1);
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBST(nums, l, mid-1);
        root.right = sortedArrayToBST(nums, mid+1, r);
        return root;
    }
}
```

### 107. Binary Tree Level Order Traversal II
- [Link](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)
- Tags: Tree, BFS
- Stars: 1

#### BFS + level-count
```java
class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null) return result;
        Queue<TreeNode> qu = new LinkedList<>();
        qu.add(root);
        int count = 1;
        List<Integer> layer = new ArrayList<>();
        while(!qu.isEmpty()){
            TreeNode node = qu.poll();
            count--;
            if(node.left != null) qu.add(node.left);
            if(node.right != null) qu.add(node.right);
            layer.add(node.val);
            if(count == 0){
                count = qu.size();
                result.add(layer);
                layer = new ArrayList<>();
            }
        }
        Collections.reverse(result);
        return result;
    }
}
```

#### BFS
```java
class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null) return result;
        DFS(result, root, 0);
        Collections.reverse(result);
        return result;
    }
    public void DFS(List<List<Integer>> result, TreeNode root, int level) {
        if(result.size() < level+1) result.add(new ArrayList<>());
        result.get(level).add(root.val);
        if(root.left != null) DFS(result, root.left, level+1);
        if(root.right != null) DFS(result, root.right, level+1);
    }
}
```

### 257. Binary Tree Paths
- [Link](https://leetcode.com/problems/binary-tree-paths/)
- Tags: Tree, DFS
- Stars: 1

#### DFS
```java
class Solution {
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> result = new ArrayList<>();
        if(root == null) return result;
        backtrack(result, root, "");
        return result;
    }
    private void backtrack(List<String> result, TreeNode root, String curr){
        curr += curr.length() == 0 ? root.val : "->"+root.val;
        if(root.left == null && root.right == null){
            result.add(curr);
            return ;
        }
        if(root.left != null) backtrack(result, root.left, curr);
        if(root.right != null) backtrack(result, root.right, curr);
    }
}
```

#### DFS + StringBuilder
```java
class Solution {
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> result = new ArrayList<>();
        if(root == null) return result;
        backtrack(result, root, new StringBuilder());
        return result;
    }
    private void backtrack(List<String> result, TreeNode root, StringBuilder sb){
        int start = sb.length();
        sb.append(sb.length() == 0? root.val : "->" + root.val);
        if(root.left == null && root.right == null){
            result.add(sb.toString());
        }
        else {
            if(root.left != null) backtrack(result, root.left, sb);
            if(root.right != null) backtrack(result, root.right, sb);
        }
        sb.delete(start, sb.length());
    }
}
```

### 118. Pascal's Triangle
- [Link](https://leetcode.com/problems/pascals-triangle/)
- Tags: Array
- Stars: 1

#### iterative adding row by row, beats 100% 100%!
```java
class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> result = new ArrayList<>();
        if(numRows == 0) return result;
        List<Integer> layer = new ArrayList<>();
        layer.add(1);
        result.add(layer);
        for(int level = 2; level <= numRows; level++){
            List<Integer> newLayer = new ArrayList<>();
            newLayer.add(1);
            for(int i=0; i<layer.size()-1; i++)
                newLayer.add(layer.get(i) + layer.get(i+1));
            newLayer.add(1);
            result.add(newLayer);
            layer = newLayer;
        }
        return result;
    }
}
```

### 27. Remove Element
- [Link](https://leetcode.com/problems/remove-element/)
- Tags: Array, Two Pointers
- Stars: 1

#### remove and swap with the last element, beats 100% in time
```java
class Solution {
    public int removeElement(int[] nums, int val) {
        if(nums.length == 0) return 0;
        int i=0, j = nums.length-1;
        while(i<=j){
            if(nums[i] != val) i++;
            else swap(nums, i, j--);
        }
        return j+1;
    }
    private void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

#### similar to [283. Move Zeroes](#283. Move Zeroes), keeps nums in order
```java
class Solution {
    public int removeElement(int[] nums, int val) {
        int p = 0;
        for(int i=0; i<nums.length; i++)
            if(nums[i] != val) nums[p++] = nums[i];
        return p;
    }
}
```

### 119. Pascal's Triangle II
- [Link](https://leetcode.com/problems/pascals-triangle-ii/)
- Tags: Array
- Stars: 2

#### ArrayList insert
```java
class Solution {
    public List<Integer> getRow(int rowIndex) {
        List<Integer> result = new ArrayList<>();
        result.add(1);
        for(int i=1; i<=rowIndex; i++){
            result.add(0, 0);
            for(int j=0; j<result.size()-1; j++)
                result.set(j, result.get(j) + result.get(j+1));
        }
        return result;
    }
}
```

#### Faster solution
Avoid inserting elements into the head during iteration. 
```java
class Solution {
    public List<Integer> getRow(int rowIndex) {
        Integer[] result = new Integer[rowIndex+1];
        result[rowIndex] = 1;
        for(int i=rowIndex-1; i>=0; i--){
            for(int j=i; j<rowIndex; j++)
                result[j] = result[j] == null? result[j+1] : result[j] + result[j+1];
        }
        return Arrays.asList(result);
    }
}
```

### 232. Implement Queue using Stacks
- [Link](https://leetcode.com/problems/implement-queue-using-stacks/)
- Tags: Stack, Design
- Stars: 1

#### two stacks
```java
import java.util.EmptyStackException;
class MyQueue {
    Stack<Integer> st1, st2;
    public MyQueue() {
        st1 = new Stack<>();
        st2 = new Stack<>();
    }    
    public void push(int x) {
        st1.add(x);
    }
    public int pop() {
        if(st2.isEmpty()){
            if(st1.isEmpty()) throw new EmptyStackException();
            while(!st1.isEmpty()) st2.add(st1.pop());
        }
        return st2.pop();
    }
    public int peek() {
        if(st2.isEmpty()){
            if(st1.isEmpty()) throw new EmptyStackException();
            while(!st1.isEmpty()) st2.add(st1.pop());
        }
        return st2.peek();
    }
    public boolean empty() {
        return st1.isEmpty() && st2.isEmpty();
    }
}
 ```

### 191. Number of 1 Bits
- [Link](https://leetcode.com/problems/number-of-1-bits/)
- Tags: Bit Manipulation
- Stars: 1

#### bit manipulation, beats 100% time and 100% space
```java
public class Solution {
    public int hammingWeight(int n) {
        n = (n&0x55555555) + ((n&0xAAAAAAAA)>>>1);
        n = (n&0x33333333) + ((n&0xCCCCCCCC)>>>2);
        n = (n&0x0F0F0F0F) + ((n&0xF0F0F0F0)>>>4);
        n = (n&0x00FF00FF) + ((n&0xFF00FF00)>>>8);
        n = (n&0x0000FFFF) + ((n&0xFFFF0000)>>>16);
        return n;
    }
}
```

### 83. Remove Duplicates from Sorted List
- [Link](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)
- Tags: Linked List
- Stars: 1

#### 2019.9.6
- time: 100%
- space: 100%
```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) return null;
        ListNode curr = head;
        while(curr.next != null) {
            if (curr.next.val == curr.val) {
                curr.next = curr.next.next;
            } else {
                curr = curr.next;
            }
        }
        return head;
    }
}
```

### 231. Power of Two
- [Link](https://leetcode.com/problems/power-of-two/)
- Tags: Math, Bit Manipulation
- Stars: 1
- References: https://leetcode.com/problems/power-of-two/discuss/63966/4-different-ways-to-solve-Iterative-Recursive-Bit-operation-Math

#### math
Refer to [Power of Three](#326-power-of-three)
```java
class Solution {
    public boolean isPowerOfTwo(int n) {
        return (n > 0) && (1073741824%n == 0);
    }
}
```

#### bit manipulation
```java
class Solution {
    public boolean isPowerOfTwo(int n) {
        return (n>0) && (n&(n-1)) == 0;
    }
}
```

### 35. Search Insert Position
- [Link](https://leetcode.com/problems/search-insert-position/)
- Tags: Array, Binary Search
- Stars: 1

#### 2019.9.12
- time: 100%
- space: 100%
```java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int l = 0, r = nums.length;
        while(l<r) {
            int mid = l + r >> 1;
            if (nums[mid] >= target) r = mid;
            else l = mid + 1;
        }
        return l;
    }
}
```

### 110. Balanced Binary Tree
- [Link](https://leetcode.com/problems/balanced-binary-tree/)
- Tags: Tree, DFS
- Stars: 1

#### DFS
```java
class Solution {
    boolean result = true;
    public boolean isBalanced(TreeNode root) {
        DFS(root);
        return result;
    }
    private int DFS(TreeNode root){
        if(root == null) return 0;
        int left = DFS(root.left), right = DFS(root.right);
        if(Math.abs(left-right) > 1) result = false;
        return 1 + Math.max(left, right);
    }
}
```

#### another DFS (without environmental variable)
```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        return DFS(root) != -1;
    }
    private int DFS(TreeNode root){
        if(root == null) return 0;
        int left = DFS(root.left);
        if(left == -1) return -1;
        int right = DFS(root.right);
        if(right == -1) return -1;
        if(Math.abs(left-right) > 1) return -1;
        return 1 + Math.max(left, right);
    }
}
```

### 263. Ugly Number
- [Link](https://leetcode.com/problems/ugly-number/)
- Tags: Math
- Stars: 1

#### divide
```java
class Solution {
    public boolean isUgly(int num) {
        if(num == 0) return false;
        while(num != 0 && num % 2 == 0) num /= 2;
        while(num != 0 && num % 3 == 0) num /= 3;
        while(num != 0 && num % 5 == 0) num /= 5;
        return num == 1 || num == 0;
    }
}
```

updated 2019.7.29
```java
class Solution {
    public boolean isUgly(int num) {
        if (num <= 0) return false;
        while(num%2 == 0) num /= 2;
        while(num%3 == 0) num /= 3;
        while(num%5 == 0) num /= 5;
        return num == 1;
    }
}
```

### 26. Remove Duplicates from Sorted Array
- [Link](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
- Tags: Array, Two Pointers
- Stars: 1

#### two pointers
```java
class Solution {
    public int removeDuplicates(int[] nums) {
        int i=0;
        for(int num: nums){
            if(i>0 && nums[i-1] == num) continue;
            nums[i++] = num;
        }
        return i;
    }
}
```

### 38. Count and Say
- [Link](https://leetcode.com/problems/count-and-say/)
- Tags: String
- Stars: 1

#### iterative
```java
class Solution {
    public String countAndSay(int n) {
        String s = "1";
        if(n == 1) return s;
        for(int i=2; i<=n; i++)
            s = getNextString(s);
        return s;
    }
    private String getNextString(String s){
        StringBuilder sb = new StringBuilder();
        int i=0, j=0;
        for(; j<s.length(); j++){
            if(j == i || s.charAt(i) == s.charAt(j)) continue;
            sb.append(j-i);
            sb.append(s.charAt(i));
            i = j;
        }
        sb.append(j-i);
        sb.append(s.charAt(i));
        return sb.toString();
    }
}
```

#### 2019.9.14 divide by chars [大雪菜]
- time: 62.55%
- space: 100%
- interviewLevel
- attention: standard template to divide string by same chars
```java
class Solution {
    public String countAndSay(int n) {
        String s = "1";
        for(int i=0; i<n-1; i++) {
            StringBuilder sb = new StringBuilder();
            for(int j=0; j<s.length(); j++) {
                int k=j;
                while(k<s.length() && s.charAt(k) == s.charAt(j)) k++;
                sb.append(k-j);
                sb.append(s.charAt(j));
                j = k-1;
            }
            s = sb.toString();
        }
        return s;
    }
}
```

### 225. Implement Stack using Queues
- [Link](https://leetcode.com/problems/implement-stack-using-queues/)
- Tags: Stack, Design
- Stars: 1

#### one queue
```java
class MyStack {
    Queue<Integer> qu;
    public MyStack() {
        qu = new LinkedList<>();
    }
    public void push(int x) {
        qu.add(x);
    }
    public int pop() {
        Queue<Integer> temp = new LinkedList<>();
        while(qu.size() > 1) temp.add(qu.poll());
        int result = qu.poll();
        qu = temp;
        return result;
    }
    public int top() {
        int temp = pop();
        qu.add(temp);
        return temp;
    }
    public boolean empty() {
        return qu.isEmpty();
    }
}
```

#### another one queue solution
```java
class MyStack {
    Queue<Integer> qu;
    public MyStack() {
        qu = new LinkedList<>();
    }
    public void push(int x) {
        qu.add(x);
        for(int i=0; i<qu.size()-1; i++)
            qu.add(qu.poll());
    }
    public int pop() {
        return qu.poll();
    }
    public int top() {
        return qu.peek();
    }
    public boolean empty() {
        return qu.isEmpty();
    }
}
```

### 67. Add Binary
- [Link](https://leetcode.com/problems/add-binary/)
- Tags: Math, String
- Stars: 1

#### StringBuilder
```java
class Solution {
    public String addBinary(String a, String b) {
        StringBuilder sb = new StringBuilder();
        int i = a.length()-1, j = b.length()-1, carry = 0;
        while(i>=0 || j>=0){
            if(i>=0) carry += a.charAt(i--)-'0';
            if(j>=0) carry += b.charAt(j--)-'0';
            sb.insert(0, carry&1);
            carry >>= 1;
        }
        while(carry > 0) {
            sb.insert(0, carry&1);
            carry >>= 1;
        }
        return sb.toString();
    }
}
```

### 112. Path Sum
- [Link](https://leetcode.com/problems/path-sum/)
- Tags: Tree, DFS
- Stars: 1

#### DFS
```java
class Solution {
    public boolean hasPathSum(TreeNode root, int sum) {
        if(root == null) return false;
        if(root.left == null && root.right == null && root.val == sum) return true;
        sum -= root.val;
        return hasPathSum(root.left, sum) || hasPathSum(root.right, sum);
    }
}
```

### 205. Isomorphic Strings
- [Link](https://leetcode.com/problems/isomorphic-strings/)
- Tags: Hash Table
- Stars: 1

#### HashMap, only beats 36.92% in time
```java
class Solution {
    public boolean isIsomorphic(String s, String t) {
        HashMap<Character, Character> map1 = new HashMap<>(), map2 = new HashMap<>();
        for(int i=0; i<s.length(); i++){
            char c1 = s.charAt(i), c2 = t.charAt(i);
            if(map1.containsKey(c1) && map2.containsKey(c2)){
                if(map1.get(c1) != c2 || map2.get(c2) != c1) return false;
                continue;
            }
            if(map1.containsKey(c1) || map2.containsKey(c2)) return false;
            map1.put(c1, c2);
            map2.put(c2, c1);
        }
        return true;
    }
}

// Similar Idea Using HashMap + HashSet
// class Solution {
//     public boolean isIsomorphic(String s, String t) {
//         HashMap<Character, Character> map = new HashMap<>();
//         HashSet<Character> set = new HashSet<>();
//         for(int i=0; i<s.length(); i++){
//             char c1 = s.charAt(i), c2 = t.charAt(i);
//             if(map.containsKey(c1)){
//                 if(!set.contains(c2) || map.get(c1) != c2) return false;
//                 continue;
//             }
//             if(set.contains(c2)) return false;
//             map.put(c1, c2);
//             set.add(c2);
//         }
//         return true;
//     }
// }
```

#### Great solution! beats 96.69% in time
```java
class Solution {
    public boolean isIsomorphic(String s, String t) {
        int[] m = new int[256], n = new int[256];
        for(int i=0; i<s.length(); i++){
            char c1 = s.charAt(i), c2 = t.charAt(i);
            if(m[c1] != n[c2]) return false;
            m[c1] = n[c2] = i+1;
        }
        return true;
    }
}
```

### 203. Remove Linked List Elements
- [Link](https://leetcode.com/problems/remove-linked-list-elements/)
- Tags: Linked List
- Stars: 2

#### partial recursive solution, beats 100% in time
```java
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        if(head == null) return null;
        if(head.val == val) return removeElements(head.next, val);
        ListNode curr = head;
        while(curr.next != null) {
            ListNode next = curr.next;
            if(next.val == val) curr.next = next.next;
            else curr = curr.next;
        }
        return head;
    }
}
```

#### recursive solution, beats 99.94% in time
```java
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        if(head == null) return null;
        head.next = removeElements(head.next, val);
        return head.val == val ? head.next : head;
    }
}
```

#### iterative solution, beats 99.94% in time
```java
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        ListNode fakeHead = new ListNode(0);
        fakeHead.next = head;
        ListNode curr = fakeHead;
        while(curr.next != null) {
            ListNode next = curr.next;
            if(next.val == val) curr.next = next.next;
            else curr = curr.next;
        }
        return fakeHead.next;
    }
}
```

### 88. Merge Sorted Array
- [Link](https://leetcode.com/problems/merge-sorted-array/)
- Tags: Array, Two Pointers
- Stars: 1

#### two pointers
```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i=m-1, j=n-1, k=m+n-1;
        while(i>=0 && j>=0) {
            if(nums2[j] > nums1[i]) nums1[k--] = nums2[j--];
            else nums1[k--] = nums1[i--];
        }
        while(j >= 0) nums1[k--] = nums2[j--];
    }
}
```

### 111. Minimum Depth of Binary Tree
- [Link](https://leetcode.com/problems/minimum-depth-of-binary-tree/)
- Tags: Tree, BFS, DFS
- Stars: 2

#### DFS
```java
class Solution {
    public int minDepth(TreeNode root) {
        if(root == null) return 0;
        if(root.left == null && root.right == null) return 1;
        int result = Integer.MAX_VALUE;
        if(root.left != null) result = minDepth(root.left);
        if(root.right != null) result = Math.min(minDepth(root.right), result);
        return result + 1;
    }
}
```

### 219. Contains Duplicate II
- [Link](https://leetcode.com/problems/contains-duplicate-ii/)
- Tags: Array, Hash Table
- Stars: 1

#### HashSet
```java
class Solution {
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        HashSet<Integer> set = new HashSet<>();
        for(int i=0; i<nums.length; i++) {
            if(!set.add(nums[i])) return true;
            if(set.size() > k) set.remove(nums[i-k]);
        }
        return false;
    }
}
```

### 290. Word Pattern
- [Link](https://leetcode.com/problems/word-pattern/)
- Tags: Hash Table
- Stars: 1

#### HashMap mapping
```java
class Solution {
    public boolean wordPattern(String pattern, String str) {
        String[] char2word = new String[26];
        HashMap<String, Integer> word2char = new HashMap<>();
        String[] words = str.split(" ");
        if(pattern.length() != words.length) return false;
        for(int i=0; i<words.length; i++) {
            if(char2word[pattern.charAt(i)-'a'] == null && 
               !word2char.containsKey(words[i])) {
                char2word[pattern.charAt(i)-'a'] = words[i];
                word2char.put(words[i], pattern.charAt(i)-'a');
            }
            else if(char2word[pattern.charAt(i)-'a'] == null || 
                    !word2char.containsKey(words[i])) 
                return false;
            else if(!char2word[pattern.charAt(i)-'a'].equals(words[i]) ||
                    word2char.get(words[i]) != pattern.charAt(i)-'a')
                return false;
        }
        return true;
    }
}
```

### 58. Length of Last Word
- [Link](https://leetcode.com/problems/length-of-last-word/)
- Tags: String
- Stars: 1

#### beats 100% in time
```java
class Solution {
    public int lengthOfLastWord(String s) {
        int i = s.length() - 1;
        while(i>=0 && s.charAt(i) == ' ') i--;
        if(i<0) return 0;
        int j = i-1;
        while(j>=0 && s.charAt(j) != ' ') j--;
        return i-j;
    }
}
```

#### String.split()
```java
class Solution {
    public int lengthOfLastWord(String s) {
        String[] list = s.split(" ");
        for(int i=list.length-1; i>=0; i--) 
            if(list[i].length() != 0) return list[i].length();
        return 0;
    }
}
```

### 168. Excel Sheet Column Title
- [Link](https://leetcode.com/problems/excel-sheet-column-title/)
- Tags: Math
- Stars: 2

#### shifted divide
```java
class Solution {
    public String convertToTitle(int n) {
        StringBuilder sb = new StringBuilder();
        while(true) {
            if(n == 0) break;
            n--;
            sb.insert(0, (char)(n%26 + 'A'));
            n /= 26;
        }
        return sb.toString();
    }
}
```

#### recursive shifted divide
```java
class Solution {
    public String convertToTitle(int n) {
        return n == 0 ? "" : convertToTitle(--n/26) + (char)(n%26 + 'A');
    }
}
```

### 204. Count Primes
- [Link](https://leetcode.com/problems/count-primes/)
- Tags: Hash Table, Math
- Stars: 1

#### O(n) time
```java
class Solution {
    public int countPrimes(int n) {
        if(n<2) return 0;
        boolean[] arr = new boolean[n];
        Arrays.fill(arr, true);
        int result = 0;
        for(int i=2; i<arr.length; i++) {
            if(arr[i]){
                result++;
                for(int j=i*2; j<arr.length; j+=i) arr[j] = false;
            }
        }
        return result;
    }
}
```

### 7. Reverse Integer
- [Link](https://leetcode.com/problems/reverse-integer/)
- Tags: Math
- Stars: 1

#### convert to long integer
```java
class Solution {
    public int reverse(int x) {
        long result = 0, num = (long)x;
        int sign = x < 0 ? -1 : 1;
        num *= sign;
        while(num > 0) {
            result *= 10;
            result += num%10;
            num /= 10;
        }
        result *= sign;
        if(result < Integer.MIN_VALUE || result > Integer.MAX_VALUE)
            return 0;
        return (int)result;
    }
}
```

#### Great Solution! no Long, no sign, check overflow in each iteration
```java
class Solution {
    public int reverse(int x) {
        int result = 0;
        while(x != 0) {
            int newResult = result*10 + x%10;
            if((newResult - x%10)/10 != result) return 0;
            result = newResult;
            x /= 10;
        }
        return result;
    }
}
```

### 260. Single Number III
- [Link](https://leetcode.com/problems/single-number-iii/)
- Tags: Bit Manipulation
- Stars: 3

#### XOR
To find n one-appearance elements based on XOR result, divide nums into n groupd s.t. each group only contains one one-appearance element. 
```java
class Solution {
    public int[] singleNumber(int[] nums) {
        int res = 0;
        for(int num: nums) res ^= num;
        int diff = res&(-res);
        int a = 0, b = 0;
        for(int num: nums) {
            if((num&diff) > 0) a ^= num;
            else b ^= num;
        }
        int[] result = { a, b };
        return result;
    }
}
```

### 216. Combination Sum III
- [Link](https://leetcode.com/problems/combination-sum-iii/)
- Tags: Array, Backtracking
- Stars: 1

#### backtrack
Notice that the numbers from 1 to 9 cannot be selected twice
```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> combinationSum3(int k, int n) {
        backtrack(new ArrayList<>(), 0, 1, k, n);
        return result;
    }
    private void backtrack(List<Integer> list, int curr, int start, int k, int n) {
        if(k == 0) {
            if(curr == n) result.add(new ArrayList<>(list));
            return ;
        }
        for(int i=start; i<=9; i++) {
            if(curr + i > n) break;
            list.add(i);
            backtrack(list, curr+i, i+1, k-1, n);
            list.remove(list.size()-1);
        }
    }
}
```

### 230. Kth Smallest Element in a BST
- [Link](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
- Tags: Binary Search, Tree
- Stars: 1

#### recursive DFS
```java
class Solution {
    TreeNode result = null;
    int count;
    public int kthSmallest(TreeNode root, int k) {
        count = k;
        inOrderTraversal(root);
        return result.val;
    }
    public void inOrderTraversal(TreeNode root) {
        if(root == null) return ;
        inOrderTraversal(root.left);
        count--;
        if(count == 0) result = root;
        else inOrderTraversal(root.right);
    }
}
```

#### non-recursive DFS
```java
class Solution {
    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> st = new Stack<>();
        while(root.left != null) {
            st.add(root);
            root = root.left;
        }
        while(true) {
            k--;
            if(k == 0) return root.val;
            root = root.right;
            while(root != null) {
                st.add(root);
                root = root.left;
            }
            root = st.pop();
        }
    }
}
```

### 12. Integer to Roman
- [Link](https://leetcode.com/problems/integer-to-roman/)
- Tags: Math, String
- Stars: 2

#### divide from high digit to low
```java
class Solution {
    public String intToRoman(int num) {
        HashMap<Integer, Character> map = new HashMap<>();
        map.put(1, 'I');
        map.put(5, 'V');
        map.put(10, 'X');
        map.put(50, 'L');
        map.put(100, 'C');
        map.put(500, 'D');
        map.put(1000, 'M');
        StringBuilder result = new StringBuilder();
        int base = 1000;
        while(num > 0) {
            int div = num / base;
            num %= base;
            if(div == 0) {
                base /= 10;
                continue;
            }
            if(div == 4 || div == 9) {
                result.append(map.get(base));
                result.append(map.get((div+1)*base));
                continue;
            }
            if(div >= 5) {
                result.append(map.get(5*base));
                div -= 5;
            }
            for(int i=0; i<div; i++) result.append(map.get(base));
        }
        return result.toString();
    }
}
```

#### map every digit into a string with direct concat
```java
class Solution {
    public String intToRoman(int num) {
        String[][] table = {
            {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"},
            {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"},
            {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"},
            {"", "M", "MM", "MMM"},
        };
        int count = 0;
        StringBuilder sb = new StringBuilder();
        while(num > 0) {
            int digit = num % 10;
            num /= 10;
            sb.insert(0, table[count++][digit]);
        }
        return sb.toString();
    }
}
```

#### interview solution!! Simple and clear!! beats 100% and 100%!!
```java
class Solution {
    public String intToRoman(int num) {
        int[] values = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
        String[] strs = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" };
        StringBuilder result = new StringBuilder();
        for(int i=0; i<values.length; i++) {
            while(num >= values[i]) {
                result.append(strs[i]);
                num -= values[i];
            }
        }
        return result.toString();
    }
}
```

### 241. Different Ways to Add Parentheses
- [Link](https://leetcode.com/problems/different-ways-to-add-parentheses/)
- Tags: Divide and Conquer
- Stars: 2

#### recursive divide and conquer, beats 80% in time
```java
class Solution {
    public List<Integer> diffWaysToCompute(String input) {
        return diffWaysToCompute(input, 0, input.length()-1);
    }
    public List<Integer> diffWaysToCompute(String input, int l, int r) {
        List<Integer> result = new ArrayList<>();
        if(l > r) return result;
        for(int i=l; i<=r; i++) {
            char c = input.charAt(i);
            if(Character.isDigit(c)) continue;
            List<Integer> leftList = diffWaysToCompute(input, l, i-1);
            List<Integer> rightList = diffWaysToCompute(input, i+1, r);
            for(int a : leftList)
                for(int b : rightList)
                    result.add(compute(a, b, c));
        }
        if(result.size() == 0)
            result.add(Integer.parseInt(input.substring(l, r+1)));
        return result;
    }
    private int compute(int a, int b, char op) {
        if(op == '+') return a+b;
        if(op == '-') return a-b;
        return a*b;
    }
}
```

### 173. Binary Search Tree Iterator
- [Link](https://leetcode.com/problems/binary-search-tree-iterator/)
- Tags: Stack, Tree, Design
- Stars: 1

#### non-recursive DFS
```java
class BSTIterator {
    Stack<TreeNode> st = new Stack<>();
    public BSTIterator(TreeNode root) {
        pushUntilLeftMost(root);
    }
    public int next() {
        TreeNode temp = st.pop();
        pushUntilLeftMost(temp.right);
        return temp.val;
    }
    public boolean hasNext() {
        return !st.isEmpty();
    }
    private void pushUntilLeftMost(TreeNode node) {
        while(node != null) {
            st.add(node);
            node = node.left;
        }
    }
}
```

### 199. Binary Tree Right Side View
- [Link](https://leetcode.com/problems/binary-tree-right-side-view/)
- Tags: Tree, DFS, BFS
- Stars: 1

#### DFS
```java
class Solution {
    List<Integer> result = new ArrayList<>();
    public List<Integer> rightSideView(TreeNode root) {
        DFS(root, 1);
        return result;
    }
    private void DFS(TreeNode root, int h) {
        if(root == null) return ;
        if(h > result.size()) result.add(root.val);
        DFS(root.right, h+1);
        DFS(root.left, h+1);
    }
}
```

### 77. Combinations
- [Link](https://leetcode.com/problems/combinations/)
- Tags: Backtracking
- Stars: 1

#### backtrack, double 100%
```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> combine(int n, int k) {
        backtrack(new ArrayList<>(), 1, n, k);
        return result;
    }
    private void backtrack(List<Integer> currList, int start, int n, int k) {
        if(k == 0) {
            result.add(new ArrayList<>(currList));
            return ;
        }
        for(int i=start; i<=n-k+1; i++) {
            currList.add(i);
            backtrack(currList, i+1, n, k-1);
            currList.remove(currList.size()-1);
        }
    }
}
```

### 64. Minimum Path Sum
- [Link](https://leetcode.com/problems/minimum-path-sum/)
- Tags: Array, Dynamic Programming
- Stars: 1

#### DP
```java
class Solution {
    public int minPathSum(int[][] grid) {
        if(grid.length == 0 || grid[0].length == 0) return 0;
        int m = grid.length, n = grid[0].length;
        int[] dp = grid[0];
        for(int i=1; i<n; i++) dp[i] += dp[i-1];
        for(int i=1; i<m; i++) {
            dp[0] += grid[i][0];
            for(int j=1; j<n; j++) 
                dp[j] = Math.min(dp[j-1], dp[j]) + grid[i][j];
        }
        return dp[dp.length-1];
    }
}
```

### 59. Spiral Matrix II
- [Link](https://leetcode.com/problems/spiral-matrix-ii/)
- Tags: Array
- Stars: 2

#### recursive onion, double 100%
```java
class Solution {
    int[][] result;
    int n;
    public int[][] generateMatrix(int n) {
        this.n = n;
        result = new int[n][n];
        onion(0, 1);
        if(n%2 == 1) result[n/2][n/2] = n*n;
        return result;
    }
    private void onion(int k, int start) {
        if(2*k >= n) return ;
        for(int j=k; j<n-k-1; j++) result[k][j] = start++;
        for(int i=k; i<n-k-1; i++) result[i][n-k-1] = start++;
        for(int j=n-k-1; j>k; j--) result[n-k-1][j] = start++;
        for(int i=n-k-1; i>k; i--) result[i][k] = start++;
        onion(k+1, start);
    }
}
```

### 123. Best Time to Buy and Sell Stock III
- [Link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)
- Tags: Array, Dynamic Programming
- Stars: 5

#### 2019.8.21 O(n^2)
- time: 5.04%
- space: 7.32%

```java
class Solution {
    public int maxProfit(int[] prices) {
        int result = 0;
        for(int i=1; i<prices.length-2; i++) {
            int a = maxProfit(prices, 0, i), b = maxProfit(prices, i+1, prices.length-1);
            result = Math.max(result, a+b);
        }
        result = Math.max(result, maxProfit(prices, 0, prices.length-1));
        return result;
    }
    public int maxProfit(int[] prices, int l, int r) {
        int result = 0, dp = 0;
        for(int i=l+1; i<=r; i++) {
            dp = Math.max(dp + prices[i] - prices[i-1], 0);
            if (dp > result) result = dp;
        }
        return result;
    }
}
```

#### 2019.8.21 state machine (DP)
- time: 99.77%
- space: 100%
- interviewLevel
- attention: `s1` and `s3` must be init to 0.

Maximize the value of each state in each iteration.

```java
class Solution {
    public int maxProfit(int[] prices) {
        int s0=Integer.MIN_VALUE, s1=0, s2=Integer.MIN_VALUE, s3=0;
        for(int p: prices) {
            int a = Math.max(s0, -p),
                b = Math.max(s1, s0 + p),
                c = Math.max(s2, s1 - p),
                d = Math.max(s3, s2 + p);
            s0 = a; s1 = b; s2 = c; s3 = d;
        }
        return Math.max(s1, s3);
    }
}
```

#### 2019.8.21 DP
- time: 99.77%
- space: 100%
- interviewLevel
- reference: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/discuss/135704/Detail-explanation-of-DP-solution
- attention: `dp[i][j]` means the maximum profit gained from `prices[:j+1]` for all `i` transactions. Notice that `prices[j]` is not necessarily the sell price in the last transaction.

```java
class Solution {
    public int maxProfit(int[] prices) {
        if (prices.length == 0) return 0;
        int[][] dp = new int[3][prices.length];
        for(int i=1; i<3; i++) {
            int currMaxB4Sell = -prices[0];
            for(int j=1; j<prices.length; j++) {
                dp[i][j] = Math.max(dp[i][j-1], prices[j] + currMaxB4Sell);
                currMaxB4Sell = Math.max(currMaxB4Sell, dp[i-1][j-1] - prices[j]);
            }
        }
        return dp[2][prices.length-1];
    }
}
```

## 300 - 399 Questions

### 331. Verify Preorder Serialization of a Binary Tree
- [Link](https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/)
- Tags: Stack
- Stars: 3
- reviewFlag

#### 2019.9.12 Stack Simulation
- time: 31.89%
- space: 100%
```java
class Solution {
    public boolean isValidSerialization(String preorder) {
        String[] list = preorder.split(",");
        Stack<Boolean> st = new Stack<>();
        int i=0;
        while(i<list.length) {
            if (list[i].equals("#")) {
                while(!st.isEmpty() && st.peek()) st.pop();
                if (st.isEmpty()) return i == list.length-1;
                st.pop();
                st.add(true);
            } else {
                st.add(false);
            }
            i++;
        }
        return st.size() == 0;
    }
}
```

#### 2019.9.12 recursive
- time: 94.47%
- space: 100%
- attention: The `parse` function parses the string list and return the pointer where the parsing ends up. If encountered an invalid situation, the funciton will return `-1`.
- attention: Pay attention that the final condition should be `parse(list, 0) == list.length`, not `parse(list, 0) == -1`.
- interviewLevel
```java
class Solution {
    public boolean isValidSerialization(String preorder) {
        String[] list = preorder.split(",");
        return parse(list, 0) == list.length;
    }
    public int parse(String[] list, int start) {
        if (start >= list.length) return -1;
        if (list[start].equals("#")) return start+1;
        int left = parse(list, start+1);
        if (left == -1) return -1;
        int right = parse(list, left);
        if (right == -1) return -1;
        return right;
    }
}
```

#### 2019.9.12 SMART IDEA!!
- time: 94.43%
- space: 12.50%
- reference: https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/discuss/78551/7-lines-Easy-Java-Solution
```java
class Solution {
    public boolean isValidSerialization(String preorder) {
        int out = 1, in = 0;
        String[] nodes = preorder.split(",");
        for(String node: nodes) {
            in++;
            if (out-in < 0) return false;
            if (!node.equals("#")) out += 2;
        }
        return out - in == 0;
    }
}
```

### 376. Wiggle Subsequence
- [Link](https://leetcode.com/problems/wiggle-subsequence/)
- Tags: Dynamic Programming, Greedy
- Stars: 4

#### 2019.9.2
- time: 100%
- space: 100%
- attention: this problem should be discussed on two sides. The init direction up or down. Either of the case is DP problem, which is similar to **Longest Increasing Subsequence**.
```java
class Solution {
    public int wiggleMaxLength(int[] nums) {
        if (nums.length < 2) return nums.length;
        return Math.max(wiggleMaxLength(nums, 1), wiggleMaxLength(nums, -1));
    }
    public int wiggleMaxLength(int[] nums, int direction) {
        int ret = 1;
        for(int i=1; i<nums.length; i++) {
            if (direction == -1 && nums[i-1] > nums[i]) {
                ret++;
                direction = 1;
            } else if (direction == 1 && nums[i-1] < nums[i]) {
                ret++;
                direction = -1;
            }
        }
        return ret;
    }
}
```

### 310. Minimum Height Trees
- [Link](https://leetcode.com/problems/minimum-height-trees/)
- Tags: BFS, Graph
- Stars: 5

#### 2019.9.2 Tree DP
- time: 75.66%
- space: 66.67%
- attention: `node.directions` records the distances from `node` to deepest leaf node of `node.childs`.
- reference: https://leetcode.com/problems/minimum-height-trees/discuss/76052/Two-O(n)-solutions
```java
class Solution {
    Node[] nodes;
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        List<Integer> ret = new ArrayList<>();
        if (n == 0) return ret;
        nodes = new Node[n];
        for(int i=0; i<n; i++) nodes[i] = new Node(i);
        for(int[] e: edges) {
            nodes[e[0]].addEdge(e[1]);
            nodes[e[1]].addEdge(e[0]);
        }
        int min = Integer.MAX_VALUE;
        for(int i=0; i<n; i++) {
            int curr = dfs(nodes[i], i);
            if (min > curr) {
                min = curr;
                ret = new ArrayList<>();
                ret.add(i);
            } else if (min == curr) ret.add(i);
        }
        return ret;
    }
    public int dfs(Node node, int parentNodeLabel) {
        int ret = 0;
        for(int i=0; i<node.childs.size(); i++) {
            int childIdx = node.childs.get(i);
            Node child = nodes[childIdx];
            if (child.label == parentNodeLabel) continue;
            int distance = node.directions.get(i);
            if (distance == -1) {
                distance = dfs(child, node.label) + 1;
                node.directions.set(i, distance);
            }
            ret = Math.max(ret, distance);
        }
        return ret;
    }
    public class Node {
        int label;
        List<Integer> childs = new ArrayList<>();
        List<Integer> directions = new ArrayList<>();
        public Node(int idx) {
            label = idx;
        }
        public void addEdge(int i) {
            childs.add(i);
            directions.add(-1);
        }
    }
}
```

#### 2019.9.2 Topological Sort
- time: 34.60%
- space: 88.89%
- interviewLevel
- language: For `Set`, use `set.iterator().next()` to get its elements.
- attention: similar to **Topological Sort** (农村包围城市)
- reference: https://leetcode.com/problems/minimum-height-trees/discuss/76055/Share-some-thoughts
- cheat
```java
class Solution {
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 0) return new ArrayList<>();
        Set<Integer>[] graph = new Set[n];
        for(int i=0; i<n; i++) graph[i] = new HashSet<>();
        for(int[] e: edges) {
            graph[e[0]].add(e[1]);
            graph[e[1]].add(e[0]);
        }
        List<Integer> leafNodes = new ArrayList<>();
        for(int i=0; i<n; i++) if (graph[i].size() <= 1) leafNodes.add(i);
        while(n > 2) {
            n -= leafNodes.size();
            List<Integer> newLeafNodes = new ArrayList<>();
            for(int leaf: leafNodes) {
                int innerNode = graph[leaf].iterator().next();
                graph[innerNode].remove(leaf);
                if (graph[innerNode].size() == 1) newLeafNodes.add(innerNode);
            }
            leafNodes = newLeafNodes;
        }
        return leafNodes;
    }
}
```

### 377. Combination Sum IV
- [Link](https://leetcode.com/problems/combination-sum-iv/)
- Tags: Dynamic Programming
- Stars: 4

#### 2019.8.31
- time: 84.14%
- space: 100%
- attention: Counting problems （计数问题） are often DP problems, because it only cares about the `count` number, not details. Just like a tree problem (i.e. segment tree), the dfs or leaf nodes are not important. On the other hand, only the values of inner nodes matters.
```java
class Solution {
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target+1];
        Arrays.sort(nums);
        dp[0] = 1;
        for(int i=1; i<=target; i++) {
            for(int num: nums) {
                if (num > i) break;
                dp[i] += dp[i-num];
            }
        }
        return dp[target];
    }
}
```

### 355. Design Twitter
- [Link](https://leetcode.com/problems/design-twitter/)
- Tags: Hash Table, Heap, Design
- Stars: 3

#### 2019.8.31
- time: 82.17%
- space: 100%
- attention: in the `follow` function, make sure to deal with the case where `followerId == followeeId`
```java
class Twitter {
    public static int timestamp = 0;
    Map<Integer, List<Tweet>> userId2tweets = new HashMap<>();
    Map<Integer, Set<Integer>> userGraph = new HashMap<>();
    public Twitter() {}
    public void postTweet(int userId, int tweetId) {
        List<Tweet> list = userId2tweets.computeIfAbsent(userId, k->new ArrayList<>());
        list.add(new Tweet(tweetId, list, list.size()));
    }
    public List<Integer> getNewsFeed(int userId) {
        PriorityQueue<Tweet> maxHeap = new PriorityQueue<>();
        addToHeap(maxHeap, userId);
        if (userGraph.containsKey(userId)) 
            for(int followeeId: userGraph.get(userId)) addToHeap(maxHeap, followeeId);
        List<Integer> ret = new ArrayList<>();
        for(int i=0; i<10; i++) {
            if (maxHeap.isEmpty()) break;
            Tweet twt = maxHeap.poll();
            ret.add(twt.tweetId);
            if (twt.idx > 0) maxHeap.add(twt.list.get(twt.idx-1));
        }
        return ret;
    }
    public void addToHeap(PriorityQueue<Tweet> heap, int userId) {
        if (userId2tweets.containsKey(userId)) {
            List<Tweet> list = userId2tweets.get(userId);
            heap.add(list.get(list.size()-1));
        }
    }
    public void follow(int followerId, int followeeId) {
        if (followerId == followeeId) return;
        userGraph.computeIfAbsent(followerId, k->new HashSet<>()).add(followeeId);
    }
    public void unfollow(int followerId, int followeeId) {
        if (!userGraph.containsKey(followerId)) return;
        userGraph.get(followerId).remove(followeeId);
    }
    public class Tweet implements Comparable<Tweet> {
        int tweetId, idx, time;
        List<Tweet> list;
        public Tweet(int tId, List<Tweet> lst, int i) {
            tweetId = tId;
            list = lst;
            idx = i;
            time = timestamp++; 
        }
        public int compareTo(Tweet t) {
            return t.time - this.time;
        }
    }
}
```

### 307. Range Sum Query - Mutable
- [Link](https://leetcode.com/problems/range-sum-query-mutable/)
- Tags: Binary Indexed Tree, Segment Tree
- Stars: 5

#### 2019.8.31 accumulate array + regular reset
- time: 27.31%
- space: 25%
```java
class NumArray {
    int[] nums, accu;
    Map<Integer, Integer> map = new HashMap<>();
    int thresh;
    public NumArray(int[] nums) {
        this.nums = nums;
        thresh = (int)Math.sqrt(nums.length);
        accu = nums.clone();
        for(int i=1; i<nums.length; i++) accu[i] += accu[i-1];
    }
    public void update(int i, int val) {
        map.put(i, val-nums[i]);
        if (map.size() > thresh) {
            for(int idx: map.keySet()) nums[idx] += map.get(idx);
            accu = nums.clone();
            for(int idx=1; idx<nums.length; idx++) accu[idx] += accu[idx-1];
            map = new HashMap<>();
        }
    }
    public int sumRange(int i, int j) {
        int ret = accu[j] - (i == 0 ? 0 : accu[i-1]);
        for(int idx: map.keySet()) if (idx >= i && idx <= j) ret += map.get(idx);
        return ret;
    }
}
```

#### 2019.8.31 Binary Indexed Tree (Segment Tree)
- time: 67.39%
- space: 62.5%
```java
class NumArray {
    Node root;
    int[] nums;
    public NumArray(int[] nums) {
        this.nums = nums;
        root = buildTree(0, nums.length-1);
    }
    public Node buildTree(int l, int r) {
        if (l > r) return null;
        if (l == r) return new Node(nums[l], l, l);
        Node left = buildTree(l, l+((r-l)>>1)), 
            right = buildTree(l+((r-l)>>1)+1, r),
            parent = new Node(left.val+right.val, l, r);
        parent.left = left;
        parent.right = right;
        return parent;
    }
    public void update(int i, int val) {
        int diff = val - nums[i];
        nums[i] = val;
        update(i, diff, root);
    }
    public void update(int i, int diff, Node node) {
        if (node == null || i < node.minIdx || i > node.maxIdx) return;
        node.val += diff;
        update(i, diff, node.left);
        update(i, diff, node.right);
    }
    public int sumRange(int i, int j) {
        return sumRange(i, j, root);
    }
    public int sumRange(int i, int j, Node node) {
        if (node == null) return 0;
        i = Math.max(i, node.minIdx);
        j = Math.min(j, node.maxIdx);
        if (i>j) return 0;
        if (i == node.minIdx && j == node.maxIdx) return node.val;
        return sumRange(i, j, node.left) + sumRange(i, j, node.right);
    }
    public class Node {
        int val, minIdx, maxIdx;
        Node left, right;
        public Node(int v, int min, int max) {
            val = v;
            minIdx = min;
            maxIdx = max;
        }
    }
}
```

#### 2019.8.31 Binary Indexed Tree (Segment Tree) by Arrays
- time: 87.08%
- space: 100%
- interviewLevel
- attention: Segment Tree can be implemented by Arrays
- attention: Note that not all the leaf nodes are in the bottom level. Thus, the length of `tree` may be larger than `2*nums.length` but must be smaller than `2*(2*nums.length)`.
```java
class NumArray {
    int[] nums;
    int[] tree;
    public NumArray(int[] nums) {
        this.nums = nums;
        tree = new int[2*2*nums.length];
        buildTree(0, 0, nums.length-1);
    }
    public void buildTree(int node, int start, int end) {
        if (start >= end) {
            if (start == end) tree[node] = nums[start];
            return;
        }
        int mid = start + ((end-start)>>1);
        int leftNode = 2*node + 1, rightNode = 2*node + 2;
        buildTree(leftNode, start, mid);
        buildTree(rightNode, mid+1, end);
        tree[node] = tree[leftNode] + tree[rightNode];
    }
    
    public void update(int i, int val) {
        update(0, 0, nums.length-1, i, val);
    }
    public void update(int node, int start, int end, int idx, int val) {
        if (idx > end || idx < start) return;
        if (start == end) {
            nums[idx] = tree[node] = val;
            return;
        }
        int mid = start + ((end-start)>>1);
        int leftNode = 2*node + 1, rightNode = 2*node + 2;
        update(leftNode, start, mid, idx, val);
        update(rightNode, mid+1, end, idx, val);
        tree[node] = tree[leftNode] + tree[rightNode];
    }
    
    public int sumRange(int i, int j) {
        return sumRange(0, 0, nums.length-1, i, j);
    }
    public int sumRange(int node, int start, int end, int L, int R) {
        if (R < start || L > end) return 0;
        if (L <= start && end <= R) return tree[node];
        int mid = start + ((end-start)>>1);
        int leftNode = 2*node + 1, rightNode = 2*node + 2;
        return sumRange(leftNode, start, mid, L, R) + sumRange(rightNode, mid+1, end, L, R);
    }
}
```

### 332. Reconstruct Itinerary
- [Link](https://leetcode.com/problems/reconstruct-itinerary/)
- Tags: DFS, Graph
- Stars: 3

#### 2019.8.31
- time: 47.28%
- space: 62.69%
```java
class Solution {
    int[][] graph;
    Map<String, Integer> port2idx;
    String[] ports;
    int count = 0;
    public List<String> findItinerary(List<List<String>> tickets) {
        this.port2idx = new HashMap<>();
        for(List<String> ticket: tickets) {
            port2idx.put(ticket.get(0), -1);
            port2idx.put(ticket.get(1), -1);
            count++;
        }
        this.ports = new String[port2idx.size()];
        int idx = 0;
        for(String port: port2idx.keySet()) ports[idx++] = port;
        Arrays.sort(ports);
        for(int i=0; i<ports.length; i++) port2idx.put(ports[i], i);
        this.graph = new int[ports.length][ports.length];
        for(List<String> ticket: tickets) {
            int fromIdx = port2idx.get(ticket.get(0)), toIdx = port2idx.get(ticket.get(1));
            graph[fromIdx][toIdx]++;
        }
        List<String> ret = new ArrayList<>();
        ret.add("JFK");
        dfs(ret, "JFK");
        return ret;
    }
    public boolean dfs(List<String> ret, String curr) {
        if (count == 0) return true;
        int i = port2idx.get(curr);
        for(int j=0; j<ports.length; j++) {
            if (graph[i][j] == 0) continue;
            graph[i][j]--;
            count--;
            ret.add(ports[j]);
            if (dfs(ret, ports[j])) return true;
            ret.remove(ret.size()-1);
            count++;
            graph[i][j]++;
        }
        return false;
    }
}
```

#### 2019.8.31 post-order dfs
- time: 23.40%
- space: 88.06%
```java
class Solution {
    List<String> ret = new LinkedList<>();
    Map<String, PriorityQueue<String>> map = new HashMap<>();
    public List<String> findItinerary(List<List<String>> tickets) {
        for(List<String> ticket: tickets) 
            map.computeIfAbsent(ticket.get(0), k->new PriorityQueue<>()).add(ticket.get(1));
        dfs("JFK");
        return ret;
    }
    public void dfs(String curr) {
        while(map.containsKey(curr) && !map.get(curr).isEmpty())
            dfs(map.get(curr).poll());
        ret.add(0, curr);
    }
}
```

### 313. Super Ugly Number
- [Link](https://leetcode.com/problems/super-ugly-number/)
- Tags: Math, Heap
- Stars: 4

#### 2019.8.31 minHeap
- time: 12.24%
- space: 16.67%
```java
class Solution {
    public int nthSuperUglyNumber(int n, int[] primes) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        Set<Integer> set = new HashSet<>();
        minHeap.add(1);
        int ret = 0;
        for(int i=0; i<n; i++) {
            ret = minHeap.poll();
            for(int prime: primes) {
                int newEle = ret * prime;
                if (newEle/prime != ret) break;
                if (!set.contains(newEle)) {
                    minHeap.add(newEle);
                    set.add(newEle);
                }
            }
        }
        return ret;
    }
}
```

Optimized (without HashSet)
- time: 30.28%
- space: 33.33%
```java
class Solution {
    public int nthSuperUglyNumber(int n, int[] primes) {
        PriorityQueue<Pair> minHeap = new PriorityQueue<>();
        int ret = 1;
        for(int i=0; i<primes.length; i++) minHeap.add(new Pair(primes[i], i));
        for(int i=1; i<n; i++) {
            Pair p = minHeap.poll();
            ret = p.val;
            for(int j=p.idx; j<primes.length; j++) {
                int newEle = ret * primes[j];
                if (newEle/primes[j] != ret) break;
                minHeap.add(new Pair(newEle, j));
            }
        }
        return ret;
    }
    public class Pair implements Comparable<Pair> {
        int val, idx;
        public Pair(int v, int i) {val = v; idx = i;}
        @Override
        public int compareTo(Pair p) {
            return this.val - p.val;
        }
    }
}
```

#### 2019.8.31 
- time: 65.66%
- space: 100%
- reference: https://leetcode.com/problems/super-ugly-number/discuss/76343/108ms-easy-to-understand-java-solution
```java
class Solution {
    public int nthSuperUglyNumber(int n, int[] primes) {
        int[] ugly = new int[n], indices = new int[primes.length];
        Arrays.fill(ugly, Integer.MAX_VALUE);
        ugly[0] = 1;
        for(int i=1; i<n; i++) {
            for(int j=0; j<primes.length; j++) ugly[i] = Math.min(ugly[i], primes[j] * ugly[indices[j]]);
            for(int j=0; j<primes.length; j++) if (primes[j]*ugly[indices[j]] <= ugly[i]) indices[j]++;
        }
        return ugly[n-1];
    }
}
```

### 319. Bulb Switcher
- [Link](https://leetcode.com/problems/bulb-switcher/)
- Tags: Math, Brainteaser
- Stars: 3

#### 2019.8.31 
- time: 100%
- space: 33.33%
```java
class Solution {
    public int bulbSwitch(int n) {
        return (int)Math.sqrt(n);
    }
}
```

### 399. Evaluate Division
- [Link](https://leetcode.com/problems/evaluate-division/)
- Tags: Union Find, Graph
- Stars: 4

#### 2019.8.31 graph
- time: 6.94%
- space: 76.47%
```java
class Solution {
    Map<String, List<String>> node2childs = new HashMap<>();
    Map<String, List<Double>> node2vals = new HashMap<>();
    List<Double> ret = new ArrayList<>();
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        for(int i=0; i<equations.size(); i++) {
            List<String> eq = equations.get(i);
            String a = eq.get(0), b = eq.get(1);
            double v = values[i];
            node2childs.computeIfAbsent(a, k->new ArrayList<>()).add(b);
            node2vals.computeIfAbsent(a, k->new ArrayList<>()).add(v);
            if (v != 0) {
                node2childs.computeIfAbsent(b, k->new ArrayList<>()).add(a);
                node2vals.computeIfAbsent(b, k->new ArrayList<>()).add(1/v);
            }
        }
        for(List<String> query: queries) ret.add(bfs(query.get(0), query.get(1)));
        double[] result = new double[ret.size()];
        for(int i=0; i<ret.size(); i++) result[i] = ret.get(i);
        return result;
    }
    public double bfs(String a, String b) {
        if (!node2childs.containsKey(a) || !node2childs.containsKey(b)) return -1.0;
        if (a.equals(b)) return 1.0;
        Queue<String> qu = new LinkedList<>();
        Queue<Double> quVals = new LinkedList<>();
        Set<String> marks = new HashSet<>();
        qu.add(a);
        quVals.add(1.0);
        marks.add(a);
        while(!qu.isEmpty()) {
            String curr = qu.poll();
            double v = quVals.poll();
            List<String> childs = node2childs.get(curr);
            List<Double> vals = node2vals.get(curr);
            for(int i=0; i<childs.size(); i++) {
                String child = childs.get(i);
                if (marks.contains(child)) continue;
                if (b.equals(child)) return v*vals.get(i);
                qu.add(child);
                quVals.add(v*vals.get(i));
                marks.add(child);
            }
        }
        return -1.0;
    }
}
```

### 386. Lexicographical Numbers
- [Link](https://leetcode.com/problems/lexicographical-numbers/)
- Tags: 
- Stars: 3

#### 2019.8.29 minHeap
- time: 17.74%
- space: 33.33%
```java
class Solution {
    public List<Integer> lexicalOrder(int n) {
        List<Integer> ret = new ArrayList<>();
        PriorityQueue<Pair> heap = new PriorityQueue<>();
        int base = 1;
        while(n >= base) {
            heap.add(new Pair(base, Math.min(n+1, base*10)));
            base *= 10;
        }
        int count = 0;
        while(count < n) {
            count++;
            Pair p = heap.poll();
            ret.add(p.val);
            if (p.increment()) heap.add(p);
        }
        return ret;
    }
    private class Pair implements Comparable<Pair> {
        int val, top;
        String str;
        public Pair(int base, int limit) {
            val = base;
            top = limit;
            str = Integer.toString(base);
        }
        @Override
        public int compareTo(Pair p) {
            return this.str.compareTo(p.str);
        }
        public boolean increment() {
            val++;
            if (val == top) return false;
            str = Integer.toString(val);
            return true;
        }
    }
}
```

#### 2019.8.29 DFS
- time: 100%
- space: 26.67%
- reference: https://leetcode.com/problems/lexicographical-numbers/discuss/86231/Simple-Java-DFS-Solution
```java
class Solution {
    List<Integer> ret = new ArrayList<>();
    int count = 0;
    int n;
    public List<Integer> lexicalOrder(int n) {
        this.n = n;
        for(int i=1; i<=9; i++) dfs(i);
        return ret;
    }
    public void dfs(int root) {
        if (count == n || root > n) return;
        ret.add(root);
        count++;
        int child = root*10;
        for(int i=0; i<10; i++) {
            if (child+i > n) break;
            dfs(child+i);
        }
    }
}
```

#### 2019.8.29 O(1) space
- time: 38.44%
- space: 26.67%
- reference: https://leetcode.com/problems/lexicographical-numbers/discuss/86242/Java-O(n)-time-O(1)-space-iterative-solution-130ms
```java
class Solution {
    public List<Integer> lexicalOrder(int n) {
        List<Integer> ret = new ArrayList<>();
        int curr = 1;
        for(int i=0; i<n; i++) {
            ret.add(curr);
            if (curr*10 <= n) curr *= 10;
            else {
                while(curr%10 == 9 || curr+1 > n) curr /= 10;
                curr++;
            }
        }
        return ret;
    }
}
```

### 357. Count Numbers with Unique Digits
- [Link](https://leetcode.com/problems/count-numbers-with-unique-digits/)
- Tags: Math, Dynamic Programming, Backtracking
- Stars: 2

#### 2019.8.28
- time: 100%
- space: 14.29%
```java
class Solution {
    public int countNumbersWithUniqueDigits(int n) {
        if (n == 0) return 1;
        if (n > 10) n = 10;
        int[] nums = new int[10];
        for(int i=0; i<nums.length; i++) nums[i] = 10-i;
        for(int i=1; i<nums.length; i++) nums[i] *= nums[i-1];
        int ret = 0;
        for(int i=0; i<n; i++) {
            ret += nums[i];
            if (i>0) ret -= nums[i]/10; // remove leading zero numbers
        }
        return ret;
    }
}
```

### 304. Range Sum Query 2D - Immutable
- [Link](https://leetcode.com/problems/range-sum-query-2d-immutable/)
- Tags: Dynamic Programming
- Stars: 3

#### 2019.8.28
- time: 99.80%
- space: 100%
- interviewLevel
```java
class NumMatrix {
    int[][] matrix;
    public NumMatrix(int[][] matrix) {
        for(int i=0; i<matrix.length; i++) {
            for(int j=1; j<matrix[0].length; j++) matrix[i][j] += matrix[i][j-1];
            if (i > 0) for(int j=matrix[0].length-1; j>=0; j--) matrix[i][j] += matrix[i-1][j];
        }
        this.matrix = matrix;
    }
    public int sumRegion(int row1, int col1, int row2, int col2) {
        return matrix[row2][col2] 
            - (row1>0 ? matrix[row1-1][col2] : 0) 
            - (col1>0 ? matrix[row2][col1-1] : 0)
            + (row1>0 && col1>0 ? matrix[row1-1][col1-1] : 0);
    }
}
```

### 397. Integer Replacement
- [Link](https://leetcode.com/problems/integer-replacement/)
- Tags: Math, Bit Manipulation
- Stars: 4
- exploreFlag

#### 2019.8.28 DP
- time: 50.43%
- space: 100%
- attention: Don't forget the `n == Integer.MAX_VALUE` case (overflow problem)

```java
class Solution {
    HashMap<Integer, Integer> map = new HashMap<>();
    public int integerReplacement(int n) {
        if (n == 1) return 0;
        if (n == Integer.MAX_VALUE) return 32;
        if (map.containsKey(n)) return map.get(n);
        int ret = Integer.MAX_VALUE;
        if ((n&1) == 0) ret = integerReplacement(n>>1) + 1;
        else ret = Math.min(integerReplacement(n+1), integerReplacement(n-1)) + 1;
        map.put(n, ret);
        return ret;
    }
}
```

### 373. Find K Pairs with Smallest Sums
- [Link](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)
- Tags: Heap
- Stars: 4

#### 2019.8.28 Heap, O(klogk)
- time: 75.46%
- space: 37.04%
- language: use `list.add(Arrays.asList(a, b, ...))` to create a List with values.

```java
class Solution {
    int[] nums1, nums2;
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums1.length == 0 || nums2.length == 0) return result;
        k = Math.min(k, nums1.length*nums2.length);
        Set<Pair> set = new HashSet<>();
        this.nums1 = nums1; this.nums2 = nums2;
        PriorityQueue<Pair> qu = new PriorityQueue<>();
        qu.add(new Pair(0, 0));
        while(k>0) {
            Pair p = qu.poll();
            result.add(Arrays.asList(nums1[p.a], nums2[p.b]));
            k--;
            if (p.a+1 < nums1.length) {
                Pair temp = new Pair(p.a+1, p.b);
                if (set.add(temp)) qu.add(temp);
            }
            if (p.b+1 < nums2.length) {
                Pair temp = new Pair(p.a, p.b+1);
                if (set.add(temp)) qu.add(temp);
            }
        }
        return result;
    }
    private class Pair implements Comparable<Pair>{
        int a, b, sum;
        public Pair(int x, int y) {
            a = x; b = y; 
            sum = nums1[x] + nums2[y];
        }
        @Override
        public int compareTo(Pair p) {
            return this.sum - p.sum;
        }
        public int hashCode() {
            return Integer.hashCode(a) + Integer.hashCode(b);
        }
        public boolean equals(Object o) {
            Pair p = (Pair)o;
            return p.a == a && p.b == b;
        }
    }
}
```

Optimized
- time: 99.68%
- space: 66.67%
- interviewLevel
- language: `class xxx implements Comparable<xxx>` and `public int compareTo(xxx x)`
- language: if a Pair/Tuple does not involve comparisons and hash, you can simply use `int[]` instead. Or you can also use a self-defined comparator with `int[]` type.
- attention: This is essentially a **multi-way merge sort**!!
```java
class Solution {
    int[] nums1, nums2;
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums1.length == 0 || nums2.length == 0) return result;
        this.nums1 = nums1; this.nums2 = nums2;
        k = Math.min(k, nums1.length*nums2.length);
        PriorityQueue<Pair> qu = new PriorityQueue<>();
        for(int i=0; i<nums1.length; i++) qu.add(new Pair(i, 0));
        while(k>0) {
            Pair p = qu.poll();
            result.add(Arrays.asList(nums1[p.a], nums2[p.b]));
            k--;
            if (p.b+1 < nums2.length) qu.add(new Pair(p.a, p.b+1));
        }
        return result;
    }
    private class Pair implements Comparable<Pair> {
        int a, b, sum;
        public Pair(int x, int y) {
            a = x; b = y; sum = nums1[x] + nums2[y];
        }
        @Override
        public int compareTo(Pair p) {
            return this.sum - p.sum;
        }
    }
}
```

### 368. Largest Divisible Subset
- [Link](https://leetcode.com/problems/largest-divisible-subset/)
- Tags: Math, Dynamic Programming
- Stars: 4

#### 2019.8.28 DP
- time: 64.51%
- space: 72.73%
- attention: The backtrack method has O(2^n) time complexity, yet the DP method only requires O(n^2). Note that even single array DP may have O(n^2) time.
- attention: By the end of the code, `for(int num: nums) if (nums[lastIdx]%num == 0) result.add(num);` is wrong. Consider the testcase `[4,8,10,240]`. The final result should have a ascending `stat` value.

```java
class Solution {
    public List<Integer> largestDivisibleSubset(int[] nums) {
        List<Integer> result = new ArrayList<>();
        if (nums.length == 0) return result;
        Arrays.sort(nums);
        int[] stat = new int[nums.length];
        Arrays.fill(stat, 1);
        int lastIdx = 0, maxCount = 1;
        for(int i=1; i<nums.length; i++) 
            for(int j=0; j<i; j++) 
                if (nums[i]%nums[j] == 0 && stat[j]+1 > stat[i]) {
                    stat[i] = stat[j] + 1;
                    if (stat[i] > maxCount) {
                        maxCount = stat[i];
                        lastIdx = i;
                    }
                }
        int count = 0;
        for(int i=0; i<nums.length; i++) 
            if (nums[lastIdx]%nums[i] == 0 && stat[i] > count) {
                count = stat[i];
                result.add(nums[i]);
            }
        return result;
    }
}
```

Optimized 
- time: 93.24%
- space: 100%
```java
class Solution {
    public List<Integer> largestDivisibleSubset(int[] nums) {
        List<Integer> result = new ArrayList<>();
        if (nums.length == 0) return result;
        Arrays.sort(nums);
        int[] stat = new int[nums.length];
        Arrays.fill(stat, 1);
        int lastIdx = 0, maxCount = 1;
        for(int i=1; i<nums.length; i++) 
            for(int j=i-1; j>=0; j--) 
                if (nums[i]%nums[j] == 0 && stat[j]+1 > stat[i]) {
                    stat[i] = stat[j] + 1;
                    if (stat[i] > maxCount) {
                        maxCount = stat[i];
                        lastIdx = i;
                        break;
                    }
                }
        int count = 0;
        for(int i=0; i<nums.length; i++) 
            if (nums[lastIdx]%nums[i] == 0 && stat[i] > count) {
                count = stat[i];
                result.add(nums[i]);
            }
        return result;
    }
}
```

### 396. Rotate Function
- [Link](https://leetcode.com/problems/rotate-function/)
- Tags: Math
- Stars: 3

#### 2019.8.27 DP
- time: 100%
- space: 100%

Let `sum` be the sum of all elements in `A`. 
Then `F(1) = F(0) + sum - A.length * A[-1]`

```java
class Solution {
    public int maxRotateFunction(int[] A) {
        if (A.length == 0) return 0;
        int dp = 0, sum = 0, result = Integer.MIN_VALUE;
        for(int i=0; i<A.length; i++) {
            dp += i * A[i];
            sum += A[i];
        }
        result = Math.max(result, dp);
        for(int i=A.length-1; i>0; i--) {
            dp = dp + sum - A.length * A[i];
            result = Math.max(result, dp);
        }
        return result;
    }
}
```

### 372. Super Pow
- [Link](https://leetcode.com/problems/super-pow/)
- Tags: Math
- Stars: 4

#### 2019.8.26
- time: 6.78%
- space: 33.33%

`(a*b)%c == ((a%c)*(b%c))%c`  
`((a^x)^y) % m == (((a^x)%m)^y) % m`

```java
class Solution {
    Map<Pair, Integer> map = new HashMap<>();
    public int superPow(int a, int[] b) {
        return superPow(a, b, b.length-1);
    }
    public int superPow(int a, int[] b, int idx) {
        if (idx == 0) return superPow(a, b[0]);
        int digit = b[idx];
        if (digit == 0) return superPow(superPow(a, b, idx-1), 10);
        b[idx] = 0;
        return (superPow(a, b, idx) * superPow(a, digit)) % 1337;
    }
    public int superPow(int a, int b) {
        if (b == 0) return 1;
        Pair pair = new Pair(a, b);
        if (map.containsKey(pair)) return map.get(pair);
        int ret = 0;
        if (b == 1) ret = a%1337;
        else if (b%2 == 0) ret = ((superPow(a, b>>1))*(superPow(a, b>>1)))%1337;
        else ret = ((superPow(a, b-1))*(superPow(a, 1)))%1337;
        map.put(pair, ret);
        return ret;
    }
    public class Pair {
        int a, b;
        public Pair(int a, int b) {
            this.a = a;
            this.b = b;
        }
        public int hashCode() {
            return Integer.hashCode(a) + Integer.hashCode(b);
        }
        public boolean equals(Object o) {
            Pair p = (Pair)o;
            return this.a == p.a && this.b == p.b;
        }
    }
}
```

#### 2019.8.27
- time: 82.53%
- space: 100%
- attention: `a %= 1337` is useful when `a` is very large

In the function `superPow(int a, int b)`, 0<=b<=10, and 0<=a<=1336. Therefore you don't need to record the value in a HashMap.

```java
class Solution {
    public int superPow(int a, int[] b) {
        int result = 1;
        for(int digit: b) {
            result = superPow(result, 10);
            if (digit > 0) result = (result * superPow(a, digit)) % 1337;
        }
        return result;
    }
    public int superPow(int a, int b) {
        int result = 1;
        a %= 1337;
        for(int i=0; i<b; i++) result = (result * a) % 1337;
        return result;
    }
}
```

### 309. Best Time to Buy and Sell Stock with Cooldown
- [Link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)
- Tags: Dynamic Programming
- Stars: 3

#### 2019.8.26 state machine (DP)
- time: 100%
- space: 100%
- interviewLevel

3 states: buy, sell, cool  
For buy: `buy[i] = Math.max(buy[i-1], cool[i-1] - prices[i])`  
For sell: `sell[i] = buy[i-1] + prices[i]`  
For cool: `cool[i] = Math.max(cool[i-1], sell[i-1])`  

```java
class Solution {
    public int maxProfit(int[] prices) {
        int buy = Integer.MIN_VALUE, sell = 0, cool = 0;
        for(int price: prices) {
            int b = Math.max(buy, cool-price),
                s = buy + price,
                c = Math.max(cool, sell);
            buy = b;
            sell = s;
            cool = c;
        }
        return Math.max(sell, cool);
    }
}
```

### 318. Maximum Product of Word Lengths
- [Link](https://leetcode.com/problems/maximum-product-of-word-lengths/)
- Tags: Bit Manipulation
- Stars: 3

#### 2019.8.26
- time: 99.67%
- space: 94.74%
- interviewLevel

```java
class Solution {
    public int maxProduct(String[] words) {
        if (words.length < 2) return 0;
        int[] nums = new int[words.length];
        int result = 0;
        for(int i=0; i<words.length; i++) nums[i] = convert(words[i]);
        for(int i=0; i<nums.length-1; i++) 
            for(int j=i+1; j<nums.length; j++)
                if ((nums[i]&nums[j]) == 0) 
                    result = Math.max(result, words[i].length()*words[j].length());
        return result;
    }
    public int convert(String word) {
        int result = 0;
        for(char c: word.toCharArray()) result |= (1<<(c-'a'));
        return result;
    }
}
```

### 306. Additive Number
- [Link](https://leetcode.com/problems/additive-number/)
- Tags: Backtracking
- Stars: 3

#### 2019.8.26
- time: 100%
- space: 100%
- interviewLevel
- attention: use `long` type to avoid overflow
- attention: a positive string number with leading zeros is invalid
- attention: this problem requires **at least** three numbers. `(Long.toString(a) + Long.toString(b)).length() < len`

```java
class Solution {
    String s;
    int len;
    public boolean isAdditiveNumber(String num) {
        this.s = num;
        this.len = num.length();
        return backtrack(-1L, -1L, 0);
    }
    public boolean backtrack(long a, long b, int start) {
        if (start >= len) return (Long.toString(a) + Long.toString(b)).length() < len;
        long number = 0;
        for(int i=start; i<len; i++) {
            if (s.charAt(start) == '0' && i>start) break;
            number *= 10;
            number += s.charAt(i) - '0';
            if (a == -1) {
                if (backtrack(number, b, i+1)) return true;
            } else if (b == -1) {
                if (backtrack(a, number, i+1)) return true;
            } else if (a+b == number) {
                return backtrack(b, number, i+1);
            } else if (a+b < number) break;
        }
        return false;
    }
}
```

### 382. Linked List Random Node
- [Link](https://leetcode.com/problems/linked-list-random-node/)
- Tags: Reservoir Sampling
- Stars: 3

#### 2019.8.26
- time: 38.58%
- space: 20%

```java
class Solution {
    ListNode head;
    Random rand = new Random();
    public Solution(ListNode head) {
        this.head = head;
    }
    public int getRandom() {
        int count = 0, result = 0;
        ListNode curr = head;
        while(curr != null) {
            count++;
            if (rand.nextInt(count) == 0) result = curr.val;
            curr = curr.next;
        }
        return result;
    }
}
```

### 389. Find the Difference
- [Link](https://leetcode.com/problems/find-the-difference/)
- Tags: Hash Table, Bit Manipulation
- Stars: 1

#### 2019.8.20 stat
- time: 98.86%
- space: 9.38%

```java
class Solution {
    public char findTheDifference(String s, String t) {
        char[] stat = new char[26];
        for(char c: s.toCharArray()) stat[c-'a']++;
        for(char c: t.toCharArray()) stat[c-'a']--;
        for(int i=0; i<26; i++) if (stat[i] > 0) return (char)(i+'a');
        return 0;
    }
}
```

#### 2019.8.20 Bit manipulation
- time: 98.86%
- space: 43.75%
- interviewLevel

```java
class Solution {
    public char findTheDifference(String s, String t) {
        char result = 0;
        for(char c: s.toCharArray()) result ^= c;
        for(char c: t.toCharArray()) result ^= c;
        return result;
    }
}
```

### 342. Power of Four
- [Link](https://leetcode.com/problems/power-of-four/)
- Tags: Bit Manipulation
- Stars: 4

#### 2019.8.20 math and bit manipulation
- time: 100%
- space: 6.67%
- interviewLevel
- attention: This question is different from 389 (Find the Difference). 3 is a prime while 4 is not.
- attention: 4 is a special number that has a special relationship with bit manipulation.

`(num&(num-1)) == 0` is to ensure that the bitCount of `num` is 1.  
`(num&0xAAAAAAAA) == 0` is to make sure that the bit in `num` does not locate in an odd position.

```java
class Solution {
    public boolean isPowerOfFour(int num) {
        return num>0 && ((num&(num-1)) == 0) && (num&0xAAAAAAAA) == 0; // ...1010...
    }
}
```

### 303. Range Sum Query - Immutable
- [Link](https://leetcode.com/problems/range-sum-query-immutable/)
- Tags: Dynamic Programming
- Stars: 1

#### 2019.8.20 DP
- time: 100%
- space: 100%
- interviewLevel

```java
class NumArray {
    int[] accu;
    public NumArray(int[] nums) {
        accu = new int[nums.length];
        if (nums.length > 0) {
            accu[0] = nums[0];
            for(int i=1; i<nums.length; i++) accu[i] = accu[i-1] + nums[i];
        }
    }
    public int sumRange(int i, int j) {
        if (i == 0) return accu[j];
        return accu[j] - accu[i-1];
    }
}
```

### 392. Is Subsequence
- [Link](https://leetcode.com/problems/is-subsequence/)
- Tags: Binary Search, Dynamic Programming, Greedy
- Stars: 3

#### 2019.8.21 two pointers
- time: 49.41%
- space: 100%

```java
class Solution {
    public boolean isSubsequence(String s, String t) {
        int i=0, j=0, len1 = s.length(), len2 = t.length();
        while(i<len1 && j<len2) {
            if (s.charAt(i) == t.charAt(j)) {
                i++; j++;
            } else j++;
        }
        return i == len1;
    }
}
```

#### 2019.8.21 Follow-Up Question
- time: 24.11%
- space: 16%
- language: init an array of List: `List<Integer>[] arrOfList = new List[n]` and then init each list one by one.
- language: binary search for list: `Collections.binarySearch(list, target)`.

```java
class Solution {
    public boolean isSubsequence(String s, String t) {
        List<Integer>[] char2indices = new List[26];
        for(int i=0; i<26; i++) char2indices[i] = new ArrayList<>();
        int len1 = s.length(), len2 = t.length();
        for(int i=0; i<len2; i++) char2indices[t.charAt(i)-'a'].add(i);
        int start = 0;
        for(int i=0; i<len1; i++) {
            char c = s.charAt(i);
            List<Integer> indices = char2indices[c-'a'];
            int idx = Collections.binarySearch(indices, start);
            if (idx < 0) idx = - (idx + 1);
            if (idx == indices.size()) return false;
            start = indices.get(idx) + 1;
        }
        return true;
    }
}
```

### 338. Counting Bits
- [Link](https://leetcode.com/problems/counting-bits/)
- Tags: Dynamic Programming, Bit Manipulation
- Stars: 4

#### 2019.8.21 Math
- time: 99.74%
- space: 5.88%

```java
class Solution {
    public int[] countBits(int num) {
        int[] result = new int[num+1];
        if (num == 0) return result;
        result[1] = 1;
        int count = 2;
        while(count < num+1) {
            int len = count;
            for(int i=0; i<len && count < num+1; i++) 
                result[count++] = 1 + result[i];
        }
        return result;
    }
}
```

#### 2019.8.21 DP + bit
- time: 99.74%
- space: 5.88%
- interviewLevel

```java
class Solution {
    public int[] countBits(int num) {
        int[] result = new int[num+1];
        for(int i=1; i<num+1; i++) result[i] = result[i>>1] + (i&1);
        return result;
    }
}
```

Another version

```java
class Solution {
    public int[] countBits(int num) {
        int[] result = new int[num+1];
        for(int i=1; i<num+1; i++) result[i] = result[(i&(i-1))] + 1;
        return result;
    }
}
```

### 398. Random Pick Index
- [Link](https://leetcode.com/problems/random-pick-index/)
- Tags: Reservoir Sampling
- Stars: 4

#### 2019.8.21 Reservoir Sampling
- time: 38.30%
- space: 94.12%
- reference: https://leetcode.com/problems/random-pick-index/discuss/88072/simple-reservoir-sampling-solution

```java
class Solution {
    int[] nums;
    Random rand = new Random();
    public Solution(int[] nums) {
        this.nums = nums;
    }
    public int pick(int target) {
        int count = 0, result = -1;
        for(int i=0; i<nums.length; i++) {
            if (nums[i] != target) continue;
            if (rand.nextInt(++count) == 0) result = i;
        }
        return result;
    }
}
```

## Facebook

### 523. Continuous Subarray Sum
- [Link](https://leetcode.com/problems/continuous-subarray-sum/)
- Tags: Math, Dynamic Programming
- Stars: 4
- exploreFlag

#### 2019.9.14
- time: 49.26%
- space: 88.24%
- attention: k can be 0
```java
class Solution {
    public boolean checkSubarraySum(int[] nums, int k) {
        if (nums.length < 2) return false;
        if (k < 0) return checkSubarraySum(nums, -k);
        for(int i=1; i<nums.length; i++) nums[i] += nums[i-1];
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        for(int i=0; i<nums.length; i++) {
            int key = k == 0 ? nums[i] : nums[i]%k;
            if (map.containsKey(key)) {
                int idx = map.get(key);
                if (idx < i-1) return true;
            } else {
                map.put(key, i);
            }
        }
        return false;
    }
}
```

### 273. Integer to English Words
- [Link](https://leetcode.com/problems/integer-to-english-words/)
- Tags: Math, String
- Stars: 3

#### 2019.9.14
- time: 6.48%
- space: 6.38%
- attention: Don't forget `if (num == 0) return "Zero";`
- attention: Only when `copy > 0` can we append a base string.
```java
class Solution {
    public String numberToWords(int num) {
        if (num == 0) return "Zero";
        Stack<Integer> st = new Stack<>();
        int base = 1;
        while(num > 0) {
            st.add(num%1000);
            st.add(base);
            num /= 1000;
            base *= 1000;
        }
        StringBuilder sb = new StringBuilder();
        Map<Integer, String> map = new HashMap<>();
        map.put(1, "One");
        map.put(2, "Two");
        map.put(3, "Three");
        map.put(4, "Four");
        map.put(5, "Five");
        map.put(6, "Six");
        map.put(7, "Seven");
        map.put(8, "Eight");
        map.put(9, "Nine");
        map.put(10, "Ten");
        map.put(11, "Eleven");
        map.put(12, "Twelve");
        map.put(13, "Thirteen");
        map.put(14, "Fourteen");
        map.put(15, "Fifteen");
        map.put(16, "Sixteen");
        map.put(17, "Seventeen");
        map.put(18, "Eighteen");
        map.put(19, "Nineteen");
        map.put(20, "Twenty");
        map.put(30, "Thirty");
        map.put(40, "Forty");
        map.put(50, "Fifty");
        map.put(60, "Sixty");
        map.put(70, "Seventy");
        map.put(80, "Eighty");
        map.put(90, "Ninety");
        map.put(1000, "Thousand");
        map.put(1000000, "Million");
        map.put(1000000000, "Billion");
        while(!st.isEmpty()) {
            base = st.pop();
            int n = st.pop(), copy = n;
            if (n >= 100) {
                if (sb.length() > 0) sb.append(' ');
                sb.append(map.get(n/100));
                sb.append(' ');
                sb.append("Hundred");
                n %= 100;
            }
            if (n > 0) {
                if (sb.length() > 0) sb.append(' ');
                if (n <= 20) {
                    sb.append(map.get(n));
                } else {
                    sb.append(map.get(n/10*10));
                    n %= 10;
                    if (n > 0) {
                        sb.append(' ');
                        sb.append(map.get(n));
                    }
                }
            }
            if (base > 1 && copy > 0) {
                sb.append(' ');
                sb.append(map.get(base));
            }
        }
        return sb.toString();
    }
}
```

#### 2019.9.14 [大雪菜]
- time: 78.17%
- space: 100%
- reviewFlag
```java
class Solution {
    public final String[] small = new String[]{"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
    public final String[] decade = new String[]{"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
    public final String[] big = new String[]{"Billion", "Million", "Thousand", ""};
    
    public String numberToWords(int num) {
        if (num == 0) return "Zero";
        StringBuilder sb = new StringBuilder();
        for(int i=1000000000, j=0; i>0; i/=1000, j++) {
            if (num >= i) {
                getPart(num/i, sb);
                sb.append(big[j]);
                sb.append(' ');
                num %= i;
            }
        }
        while (sb.charAt(sb.length()-1) == ' ') sb.deleteCharAt(sb.length()-1);
        return sb.toString();
    }
    
    public void getPart(int n, StringBuilder sb) {
        if (n >= 100) {
            sb.append(small[n/100]);
            sb.append(' ');
            sb.append("Hundred");
            sb.append(' ');
            n %= 100;
        }
        if (n >= 20) {
            sb.append(decade[n/10]);
            sb.append(' ');
            n %= 10;
        }
        if (n > 0) {
            sb.append(small[n]);
            sb.append(' ');
        }
    }
}
```

### 560. Subarray Sum Equals K
- [Link](https://leetcode.com/problems/subarray-sum-equals-k/)
- Tags: Array, Hash Table
- Stars: 4
- reviewFlag

#### 2019.9.14
- time: 36.50%
- space: 98.91%
```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        int ret = 0, sum = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for(int i=0; i<nums.length; i++) {
            sum += nums[i];
            if (sum == k) ret++;
            ret += map.getOrDefault(sum - k, 0);
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return ret;
    }
}
```

### 973. K Closest Points to Origin
- [Link](https://leetcode.com/problems/k-closest-points-to-origin/)
- Tags: Divide and Conquer, Heap, Sort
- Stars: 3
- reviewFlag

#### 2019.9.6 quick select
- time: 99.75%
- space: 77.64%
- interviewLevel
```java
class Solution {
    public int[][] kClosest(int[][] points, int K) {
        int l = 0, r = points.length-1, j=0;
        while(true) {
            j = partition(points, l, r);
            if (j+1 > K) {
                r = j - 1;
            } else if (j+1 < K) {
                l = j + 1;
            } else break;
        }
        return Arrays.copyOfRange(points, 0, j+1);
    }
    public int partition(int[][] points, int l, int r) {
        int i=l, j=r+1, pivot = distance(points[l]);
        while(true) {
            while(distance(points[++i]) < pivot && i<r);
            while(pivot < distance(points[--j]) && l<j);
            if (i>=j) break;
            swap(points, i, j);
        }
        swap(points, l, j);
        return j;
    }
    public int distance(int[] p) {
        return p[0]*p[0] + p[1]*p[1];
    }
    public void swap(int[][] points, int i, int j) {
        if (i == j) return;
        int[] temp = points[i];
        points[i] = points[j];
        points[j] = temp;
    }
}
```

### 953. Verifying an Alien Dictionary
- [Link](https://leetcode.com/problems/verifying-an-alien-dictionary/)
- Tags: Hash Table
- Stars: 2

#### 2019.9.6
- time: 100%
- space: 100%
- attention: When a function has two similar inputs like `compare(T xxx, T xxx)`, don't use varname like `a` and `b`. Instead, use `xx1` and `xx2` to avoid some mistakes. 
```java
class Solution {
        int[] map;
    public boolean isAlienSorted(String[] words, String order) {
        map = new int[26];
        for(int i=0; i<26; i++) {
            map[order.charAt(i)-'a'] = i;
        }
        for(int i=0; i<words.length-1; i++) {
            if (compare(words[i], words[i+1]) > 0) return false;
        }
        return true;
    }
    public int compare(String a, String b) {
        int i=0, j=0, len1 = a.length(), len2 = b.length();
        while(i<len1 && j<len2) {
            int idx1 = map[a.charAt(i++)-'a'], idx2 = map[b.charAt(j++)-'a'];
            if (idx1 < idx2) return -1;
            else if (idx1 > idx2) return 1;
        }
        if (i < len1) return 1;
        else if (j < len2) return -1;
        return 0;
    }
}
```

# bili 视频

## 大雪菜 -- Tree

### 652. Find Duplicate Subtrees
- [Link](https://leetcode.com/problems/find-duplicate-subtrees/)
- Tags: Tree
- Stars: 4

#### 2019.9.12 encoding a tree recursively
- time: 77.06%
- space: 86.36%
```java
class Solution {
    List<TreeNode> ret = new ArrayList<>();
    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
        if(root == null) return ret;
        Map<String, Integer> map = new HashMap<>();
        addToMap(root, map);
        return ret;
    }
    public String addToMap(TreeNode root, Map<String, Integer> map) {
        if (root == null) return "#";
        String left = addToMap(root.left, map), right = addToMap(root.right, map);
        StringBuilder sb = new StringBuilder();
        sb.append(root.val);
        sb.append(',');
        sb.append(left);
        sb.append(',');
        sb.append(right);
        String code = sb.toString();
        int count = map.getOrDefault(code, 0) + 1;
        map.put(code, count);
        if (count == 2) ret.add(root);
        return code;
    }
}
```

### 428. Serialize and Deserialize N-ary Tree
- [Link](https://leetcode.com/problems/serialize-and-deserialize-n-ary-tree/)
- Tags: Tree
- Stars: 3

#### 2019.9.12 record the number of children
- time: 87.19%
- space: 27.78%
```java
class Codec {
    public String serialize(Node root) {
        if (root == null) return "";
        StringBuilder sb = new StringBuilder();
        sb.append(root.val);
        sb.append(',');
        sb.append(root.children.size());
        for(Node child: root.children) {
            sb.append(',');
            sb.append(serialize(child));
        }
        return sb.toString();
    }
    public Node deserialize(String s) {
        if (s.length() == 0) return null;
        String[] nodes = s.split(",");
        int[] index = new int[1];
        return deserialize(nodes, index);
    }
    public Node deserialize(String[] nodes, int[] index) {
        if (index[0] >= nodes.length) return null;
        int node = Integer.parseInt(nodes[index[0]++]);
        int N = Integer.parseInt(nodes[index[0]++]);
        Node root = new Node(node, new ArrayList<>());
        for(int i=0; i<N; i++) {
            root.children.add(deserialize(nodes, index));
        }
        return root;
    }
}
```

### 449. Serialize and Deserialize BST
- [Link](https://leetcode.com/problems/serialize-and-deserialize-bst/)
- Tags: Tree
- Stars: 3

#### 2019.9.12 Preorder in enough
- time: 92.23%
- space: 89.58%
```java
public class Codec {
    public String serialize(TreeNode root) {
        if (root == null) return "";
        String left = serialize(root.left), right = serialize(root.right);
        StringBuilder sb = new StringBuilder();
        sb.append(root.val);
        if (left.length() > 0) {
            sb.append(',');
            sb.append(left);
        }
        if (right.length() > 0) {
            sb.append(',');
            sb.append(right);
        }
        return sb.toString();
    }
    public TreeNode deserialize(String s) {
        if (s.length() == 0) return null;
        int[] index = new int[1];
        return deserialize(s.split(","), index, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
    public TreeNode deserialize(String[] nodes, int[] index, int min, int max) {
        if (index[0] == nodes.length) return null;
        int val = Integer.parseInt(nodes[index[0]]);
        if (val < min || val > max) return null;
        TreeNode root = new TreeNode(val);
        index[0]++;
        root.left = deserialize(nodes, index, min, val-1);
        root.right = deserialize(nodes, index, val+1, max);
        return root;
    }
}
```

### 297. Serialize and Deserialize Binary Tree
- [Link](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
- Tags: Tree, Design
- Stars: 3

#### 2019.9.12 Preorder + "#"
- time: 88.78%
- space: 30.47%
```java
public class Codec {
    public String serialize(TreeNode root) {
        if (root == null) return "#";
        StringBuilder sb = new StringBuilder();
        sb.append(root.val);
        sb.append(',');
        sb.append(serialize(root.left));
        sb.append(',');
        sb.append(serialize(root.right));
        return sb.toString();
    }
    public TreeNode deserialize(String s) {
        if (s.length() == 0) return null;
        String[] nodes = s.split(",");
        int[] index = new int[1];
        return deserialize(nodes, index);
    }
    public TreeNode deserialize(String[] nodes, int[] index) {
        if (index[0] == nodes.length) return null;
        String node = nodes[index[0]++];
        if (node.equals("#")) return null;
        TreeNode root = new TreeNode(Integer.parseInt(node));
        root.left = deserialize(nodes, index);
        root.right = deserialize(nodes, index);
        return root;
    }
}
```

## 大雪菜 -- 字符串

### 5. Longest Palindromic Substring
- [Link](https://leetcode.com/problems/longest-palindromic-substring/)
- Tags: String, Dynamic Programming
- Stars: 4
- exploreFlag

#### 2019.9.14 O(n^2) [大雪菜]
- time: 32.16%
- space: 52.42%
```java
class Solution {
    public String longestPalindrome(String s) {
        String ret = "";
        for(int i=0; i<s.length(); i++) {
            for(int j=i, k=i; j>=0 && k<s.length() && s.charAt(j) == s.charAt(k); j--, k++) {
                if (ret.length() < k-j+1) ret = s.substring(j, k+1);
            }
            for(int j=i, k=i+1; j>=0 && k<s.length() && s.charAt(j) == s.charAt(k); j--, k++) {
                if (ret.length() < k-j+1) ret = s.substring(j, k+1);
            }
        }
        return ret;
    }
}
```

## 大雪菜 -- 基本数据结构专题

### 706. Design HashMap
- [Link](https://leetcode.com/problems/design-hashmap/)
- Tags: Hash Table, Design
- Stars: 3

#### 2019.9.11
- time: 86.57%
- space: 97.30%
- attention: It's better to use a prime for the value of `len`.
```java
class MyHashMap {
    private int len = 20011;
    private Node[] map;
    public MyHashMap() {
        map = new Node[len];
    }
    public void put(int key, int value) {
        int h = Integer.hashCode(key)%len;
        if (map[h] == null) {
            map[h] = new Node(key, value);
        } else {
            Node node = map[h];
            while(node != null) {
                if (node.k == key) {
                    node.v = value;
                    return;
                } else {
                    node = node.next;
                }
            }
            node = map[h];
            map[h] = new Node(key, value);
            map[h].next = node;
        }
    }    
    public int get(int key) {
        int h = Integer.hashCode(key)%len;
        if (map[h] == null) return -1;
        Node node = map[h];
        while(node != null) {
            if (node.k == key) return node.v;
            node = node.next;
        }
        return -1;
    }
    public void remove(int key) {
        int h = Integer.hashCode(key)%len;
        if (map[h] == null) return;
        if (map[h].k == key) {
            map[h] = map[h].next;
            return;
        }
        Node prev = map[h], node = prev.next;
        while(node != null) {
            if (node.k == key) {
                prev.next = node.next;
                return;
            }
            prev = node;
            node = node.next;
        }
    }
    private class Node {
        int k, v;
        Node next;
        public Node(int key, int value) {
            k = key;
            v = value;
            next = null;
        }
    }
}
```


# Topics

## String

### 557. Reverse Words in a String III
- [Link](https://leetcode.com/problems/reverse-words-in-a-string-iii/)
- Tags: String
- Stars: 1

#### two pointers reverse word by word
```java
class Solution {
    public String reverseWords(String s) {
        int i=0;
        StringBuilder sb = new StringBuilder(s);
        while(i<sb.length()){
            while(i<sb.length() && sb.charAt(i) == ' ') i++;
            int j = i;
            while(j<sb.length() && sb.charAt(j) != ' ') j++;
            reverse(sb, i, j-1);
            i = j+1;
        }
        return sb.toString();
    }
    private void reverse(StringBuilder sb, int i, int j){
        while(i<j) swap(sb, i++, j--);
    }
    private void swap(StringBuilder sb, int i, int j){
        char c = sb.charAt(i);
        sb.setCharAt(i, sb.charAt(j));
        sb.setCharAt(j, c);
    }
}
```

#### Built-in functions
```java
class Solution {
    public String reverseWords(String s) {
        String[] strs = s.split(" ");
        for(int i=0; i<strs.length; i++){
            strs[i] = (new StringBuilder(strs[i])).reverse().toString();
        }
        return String.join(" ", Arrays.asList(strs));
    }
}
```

### 893. Groups of Special-Equivalent Strings
- [Link](https://leetcode.com/problems/groups-of-special-equivalent-strings/)
- Tags: String
- Stars: 1

#### HashArray
The key is to encode the string into something hashable and put it all into a HashSet.

Count number of characters for odd and even indices separately.  
Odd-indexed characters are counted in chars[:26], while even-indexed characters are counted in chars[26:]. 
```java
class Solution {
    public int numSpecialEquivGroups(String[] A) {
        HashSet<HashArray> set = new HashSet<>();
        for(String s : A){
            HashArray harr = new HashArray(s);
            set.add(harr);
        }
        return set.size();
    }
}
class HashArray {
    int[] chars = new int[52];
    public HashArray(String s){
        for(int i=0; i<s.length(); i+=2)
            chars[s.charAt(i)-'a']++;
        for(int i=1; i<s.length(); i+=2)
            chars[s.charAt(i)-'a'+26]++;
    }
    public boolean equals(Object o){
        for(int i=0; i<52; i++)
            if(this.chars[i] != ((HashArray)o).chars[i]) return false;
        return true;
    }
    public int hashCode(){
        return Arrays.hashCode(chars);
    }
}
```

### 824. Goat Latin
- [Link](https://leetcode.com/problems/goat-latin/)
- Tags: String
- Stars: 1

#### StringBuilder modify, no split
```java
class Solution {
    HashSet<Character> set = new HashSet<>(
        Arrays.asList(new Character[] {'a', 'e', 'i','o','u','A','E','I','O','U'}));
    public String toGoatLatin(String s) {
        StringBuilder sb = new StringBuilder(s);
        int i = 0, count = 0;
        while(true){
            while(i<sb.length() && sb.charAt(i) == ' ') i++;
            if(i == sb.length()) break;
            int j = i;
            while(j<sb.length() && sb.charAt(j) != ' ') j++;
            count++;
            
            if(beginsWithVowels(sb, i, j)) {
                sb.insert(j, "ma");
                j += 2;
            }
            else{
                char c = sb.charAt(i);
                sb.delete(i, i+1);
                sb.insert(j-1, c);
                sb.insert(j, "ma");
                j+=2;
            }
            for(int k=0; k<count; k++) sb.insert(j++, 'a');
            
            i = j;
        }
        return sb.toString();
    }
    private boolean beginsWithVowels(StringBuilder sb, int start, int end){
        return set.contains(sb.charAt(start));
    }
}
```

### 521. Longest Uncommon Subsequence I
- [Link](https://leetcode.com/problems/longest-uncommon-subsequence-i/)
- Tags: String
- Stars: 1

#### April Fool's Question
```java
class Solution {
    public int findLUSlength(String a, String b) {
        return a.equals(b) ? -1 : Math.max(a.length(), b.length());
    }
}
```

### 917. Reverse Only Letters
- [Link](https://leetcode.com/problems/reverse-only-letters/)
- Tags: String
- Stars: 1

#### skipping two pointers swap
```java
class Solution {
    public String reverseOnlyLetters(String S) {
        StringBuilder s = new StringBuilder(S);
        int i = 0, j = s.length()-1;
        while(true){
            while(i<j && !Character.isLetter(s.charAt(i))) i++;
            while(i<j && !Character.isLetter(s.charAt(j))) j--;
            if(i >= j) break;
            char c = s.charAt(i);
            s.setCharAt(i, s.charAt(j));
            s.setCharAt(j, c);
            i++; j--;
        }
        return s.toString();
    }
}
```

## Linked List

### 876. Middle of the Linked List
- [Link](https://leetcode.com/problems/middle-of-the-linked-list/)
- Tags: Linked List
- Stars: 1

#### slow-fast
```java
class Solution {
    public ListNode middleNode(ListNode head) {
        if(head == null) return null;
        ListNode slow = head, fast = head;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }
}
```


## Backtracking Questions
[Reference](https://leetcode.com/problems/permutations/discuss/18239/A-general-approach-to-backtracking-questions-in-Java-(Subsets-Permutations-Combination-Sum-Palindrome-Partioning))

**Backtrack == 发散式DFS**

### 78. Subsets
- [Link](https://leetcode.com/problems/subsets/)
- Tags: Array, Backtracking, Bit Manipulation
- Stars: 1

#### General Approach
```java
class Solution {
    List<List<Integer>> result;
    public List<List<Integer>> subsets(int[] nums) {
        result = new ArrayList<>();
        backtrack(new ArrayList<Integer>(), nums, 0);
        return result;
    }
    
    private void backtrack(List<Integer> currList, int[] nums, int start){
        result.add(new ArrayList<>(currList));
        for(int i=start; i<nums.length; i++){
            currList.add(nums[i]);
            backtrack(currList, nums, i+1);
            currList.remove(currList.size()-1);
        }
    }
}
```

#### My solution (Faster!)
```java
class Solution {
    List<List<Integer>> result;
    
    public List<List<Integer>> subsets(int[] nums) {
        result = new ArrayList<>();
        DFS(nums, 0, new ArrayList<Integer>());
        return result;
    }
    private void DFS(int[] nums, int k, List<Integer> currList){
        if(k==nums.length){
            result.add(currList);
            return ;
        }
        DFS(nums, k+1, new ArrayList<>(currList));
        currList.add(nums[k]);
        DFS(nums, k+1, currList);
    }
}
```

#### 2019.9.14 [大雪菜] binary numbers
- time: 41.21%
- space: 96.72%
- reviewFlag
```java
class Solution {
    List<List<Integer>> ret = new ArrayList<>();
    public List<List<Integer>> subsets(int[] nums) {
        for(int i=0; i<(1<<nums.length); i++) {
            List<Integer> path = new ArrayList<>();
            for(int j=0; j<nums.length; j++) {
                if ((i>>j & 1) == 1) {
                    path.add(nums[j]);
                }
            }
            ret.add(path);
        }
        return ret;
    }
}
```

### 90. Subsets II
- [Link](https://leetcode.com/problems/subsets-ii/)
- Tags: Array, Backtracking
- Stars: 1

#### General Approach
```java
class Solution {
    List<List<Integer>> result;
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        result = new ArrayList<>();
        Arrays.sort(nums);
        backtrack(new ArrayList<Integer>(), nums, 0);
        return result;
    }
    private void backtrack(List<Integer> currList, int[] nums, int start){
        result.add(new ArrayList<>(currList));
        for(int i=start; i<nums.length; i++){
            if(i==start || nums[i-1] != nums[i]){
                currList.add(nums[i]);
                backtrack(currList, nums, i+1);
                currList.remove(currList.size()-1);
            }
        }
    }
}
```

#### My Solution
```java
class Solution {
    List<List<Integer>> result;
    
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        result = new ArrayList<>();
        Arrays.sort(nums);
        backtrack(new ArrayList<Integer>(), nums, 0);
        return result;
    }
    
    private void backtrack(List<Integer> currList, int[] nums, int start){
        if(start==nums.length) {
            result.add(currList);
            return ;
        }
        int end = start;
        while(end<nums.length && nums[end] == nums[start])
            end++;
        for(int i = start+1; i<=end; i++){
            List<Integer> temp = new ArrayList<>(currList);
            for(int j=start; j<i; j++)
                temp.add(nums[j]);
            backtrack(temp, nums, end);
        }
        backtrack(currList, nums, end);
    }
}
```

#### 2019.7.8 backtrack
- time: 100%
- space: 99.36%
```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        backtrack(nums, new ArrayList<>(), 0);
        return result;
    }
    private void backtrack(int[] nums, List<Integer> curr, int start) {
        if (start >= nums.length) {
            result.add(new ArrayList<>(curr));
            return ;
        }
        // Case 1: not take `nums[start]` element
        int next = start;
        while(++next < nums.length && nums[next] == nums[start]); // Find the next different element
        backtrack(nums, curr, next);
        // Case 2: take `nums[start]` element
        curr.add(nums[start]);
        backtrack(nums, curr, start+1);
        curr.remove(curr.size()-1);
    }
}
```

#### 2019.9.14 [大雪菜]
- time: 100%
- space: 98.53%
```java
class Solution {
    List<List<Integer>> ret = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        dfs(nums, 0);
        return ret;
    }
    public void dfs(int[] nums, int u) {
        if (u == nums.length) {
            ret.add(new ArrayList<>(path));
            return;
        }
        int k=u;
        while(k<nums.length && nums[k] == nums[u]) k++;
        dfs(nums, k);
        for(int i=u; i<k; i++) {
            path.add(nums[u]);
            dfs(nums, k);
        }
        for(int i=u; i<k; i++) path.remove(path.size()-1);
    }
}
```

### 46. Permutations
- [Link](https://leetcode.com/problems/permutations/)
- Tags: Backtracking
- Stars: 1

#### My Backtracking Solution (not general but faster)
<span id="46-DP" />
This is a DP-like solution. 
For each iteration, you only consider the additional permutations that the k-th element brings about. 

```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        if(nums.length == 0) return result;
        result.add(new ArrayList<>());
        for(int i=0; i<nums.length; i++)
            backtrack(nums, i);
        return result;
    }
    private void backtrack(int[] nums, int start) {
        if(start == nums.length) return ;
        int len = result.size(), num = nums[start];
        for(int i=0; i<len; i++) {
            List<Integer> row = result.get(i);
            for(int j=0; j<row.size(); j++) {
                List<Integer> newRow = new ArrayList<>(row);
                newRow.add(j, num);
                result.add(newRow);
            }
            row.add(num);
        }
    }
}
```

#### Marking along the backtracking paths (general solution)
```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        backtrack(nums, new HashSet<>(), new ArrayList<>());
        return result;
    }
    private void backtrack(int[] nums, HashSet<Integer> mark, List<Integer> currList){
        if(currList.size() == nums.length){
            result.add(new ArrayList<>(currList));
            return ;
        }
        for(int i=0; i<nums.length; i++)
            if(!mark.contains(i)){
                mark.add(i);
                currList.add(nums[i]);
                backtrack(nums, mark, currList);
                currList.remove(currList.size()-1);
                mark.remove(i);
            }
    }
    
}
```

Updated 2019.7.28
```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        backtrack(nums, new boolean[nums.length], new ArrayList<>());
        return result;
    }
    public void backtrack(int[] nums, boolean[] used, List<Integer> currList) {
        if (currList.size() == nums.length) {
            result.add(new ArrayList<>(currList));
            return ;
        }
        for (int i=0; i<used.length; i++) {
            if (used[i]) continue;
            currList.add(nums[i]);
            used[i] = true;
            backtrack(nums, used, currList);
            used[i] = false;
            currList.remove(currList.size() - 1);
        }
    }
}
```

### 47. Permutations II
- [Link](https://leetcode.com/problems/permutations-ii/)
- Tags: Backtracking
- Stars: 4

#### Marking along the backtracking paths (general solution)
Notice that the DP-like solution in [46. Permutations](#46-DP) does not work here because of the presence of duplicates. 
e.g. If both `[1,3,3]` and `[3,1,3]` are in the result and you are going to insert a `1` in them, they can both get the same array `[1,3,1,3]`, which is not allowed. 
```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    HashSet<Integer> unique = new HashSet<>();
    public List<List<Integer>> permuteUnique(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for(int num: nums) {
            unique.add(num);
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        backtrack(nums, map, new ArrayList<>());
        return result;
    }
    private void backtrack(int[] nums, HashMap<Integer, Integer> map, List<Integer> currList){
        if(currList.size() == nums.length) {
            result.add(new ArrayList<>(currList));
            return ;
        }
        for(int num : unique){
            if(map.get(num)>0){
                map.put(num, map.get(num)-1);
                currList.add(num);
                backtrack(nums, map, currList);
                currList.remove(currList.size()-1);
                map.put(num, map.get(num)+1);
            }
        }
    }
}
```

updated 2019.7.28
```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    int len;
    public List<List<Integer>> permuteUnique(int[] nums) {
        HashMap<Integer, Integer> stat = getStat(nums);
        len = nums.length;
        backtrack(stat, new ArrayList<>());
        return result;
    }
    public void backtrack(HashMap<Integer, Integer> stat, List<Integer> currList) {
        if (currList.size() == len) {
            result.add(new ArrayList<>(currList));
            return ;
        }
        for (int key: stat.keySet()) {
            if (stat.get(key) <= 0) continue;
            stat.put(key, stat.get(key) - 1);
            currList.add(key);
            backtrack(stat, currList);
            currList.remove(currList.size() - 1);
            stat.put(key, stat.get(key) + 1);
        }
    }
    public HashMap<Integer, Integer> getStat(int[] nums) {
        HashMap<Integer, Integer> stat = new HashMap<>();
        for(int num: nums) stat.put(num, stat.getOrDefault(num, 0) + 1);
        return stat;
    }
}
```

#### Marking again! (faster general solution)
```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        backtrack(nums, new boolean[nums.length], new ArrayList<>());
        return result;
    }
    private void backtrack(int[] nums, boolean[] used, List<Integer> currList){
        if(currList.size() == nums.length){
            result.add(new ArrayList<>(currList));
            return ;
        }
        for(int i=0; i<nums.length; i++){
            if(used[i] || i>0 && nums[i-1] == nums[i] && !used[i-1]) continue;
            used[i] = true;
            currList.add(nums[i]);
            backtrack(nums, used, currList);
            currList.remove(currList.size()-1);
            used[i] = false;
        }
    }
}
```

#### 2019.9.14 [大雪菜] assign each number in nums to different possitions
- time: 56.76%
- space: 98.51%
- reviewFlag
```java
class Solution {
    List<List<Integer>> ret = new ArrayList<>();
    Integer[] path;
    boolean[] used;
    public List<List<Integer>> permuteUnique(int[] nums) {
        if (nums.length == 0) return ret;
        path = new Integer[nums.length];
        used = new boolean[nums.length];
        Arrays.sort(nums);
        dfs(nums, 0, 0);
        return ret;
    }
    public void dfs(int[] nums, int u, int start) {
        if (u == nums.length) {
            ret.add(new ArrayList<>(Arrays.asList(path)));
            return;
        }
        for(int i=start; i<nums.length; i++) {
            if (!used[i]) {
                used[i] = true;
                path[i] = nums[u];
                dfs(nums, u+1, u+1 < nums.length && nums[u+1] == nums[u] ? i+1 : 0);
                used[i] = false;
            }
        }
    }
}
```

#### 2019.9.14 [大雪菜] assign each position with different numbers from nums
- time: 56.76%
- space: 97.01%
- reviewFlag
```java
class Solution {
    List<List<Integer>> ret = new ArrayList<>();
    Integer[] path;
    boolean[] used;
    public List<List<Integer>> permuteUnique(int[] nums) {
        if (nums.length == 0) return ret;
        path = new Integer[nums.length];
        used = new boolean[nums.length];
        Arrays.sort(nums);
        dfs(nums, 0);
        return ret;
    }
    public void dfs(int[] nums, int u) {
        if (u == nums.length) {
            ret.add(new ArrayList<>(Arrays.asList(path)));
            return;
        }
        for(int i=0; i<nums.length; i++) {
            if (!used[i]) {
                used[i] = true;
                path[u] = nums[i];
                dfs(nums, u+1);
                used[i] = false;
                while(i+1<nums.length && nums[i+1] == nums[i]) i++;
            }
        }
    }
}
```

### 22. Generate Parentheses
- [Link](https://leetcode.com/problems/generate-parentheses/)
- Tags: String, Backtracking
- Stars: 1

[YouTube Video](https://www.youtube.com/watch?v=sz1qaKt0KGQ)

#### My Backtracking Solution
```java
class Solution {
    List<String> result;
    public List<String> generateParenthesis(int n) {
        result = new ArrayList<>();
        backtrack(new StringBuilder(), n, 0);
        return result;
    }
    private void backtrack(StringBuilder sb, int left, int right){
        if(left == 0 && right == 0){
            result.add(sb.toString());
            return ;
        }
        if(left>0){
            sb.append('(');
            backtrack(sb, left-1, right+1);
            sb.delete(sb.length()-1, sb.length());
        }
        if(right>0){
            sb.append(')');
            backtrack(sb, left, right-1);
            sb.delete(sb.length()-1, sb.length());
        }
    }
}
```

### 17. Letter Combinations of a Phone Number
- [Link](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)
- Tags: String, Backtracking
- Stars: 1

#### simple backtracking
```java
class Solution {
    List<String> result = new ArrayList<>();
    HashMap<Character, String> map = new HashMap<>();
    public List<String> letterCombinations(String digits) {
        if(digits == null || digits.length() == 0) return result;
        map.put('2', "abc");
        map.put('3', "def");
        map.put('4', "ghi");
        map.put('5', "jkl");
        map.put('6', "mno");
        map.put('7', "pqrs");
        map.put('8', "tuv");
        map.put('9', "wxyz");
        backtrack(digits, 0, new StringBuilder());
        return result;
    }
    private void backtrack(String digits, int start, StringBuilder currsb){
        if(start == digits.length()){
            result.add(currsb.toString());
            return ;
        }
        for(char c: map.get(digits.charAt(start)).toCharArray()){
            currsb.append(c);
            backtrack(digits, start+1, currsb);
            currsb.delete(currsb.length()-1, currsb.length());
        }
    }
}
```

#### 2019.9.14 [大雪菜]
- time: 63.60%
- space: 98.63%
- interviewLevel
```java
class Solution {
    List<String> ret = new ArrayList<>();
    final String[] map = new String[]{"","","abc","def","ghi","jkl", "mno", "pqrs", "tuv", "wxyz"};
    public List<String> letterCombinations(String digits) {
        if (digits.length() == 0) return ret;
        ret.add("");
        for(char digit: digits.toCharArray()) {
            List<String> newRet = new ArrayList<>();
            for(char c: map[digit-'0'].toCharArray()) {
                for(String prefix: ret) {
                    newRet.add(prefix + c);
                }
            }
            ret = newRet;
        }
        return ret;
    }
}
```

### 200. Number of Islands
- [Link](https://leetcode.com/problems/number-of-islands/)
- Tags: DFS, BFS, Union Find
- Stars: 1

#### DFS
```java
class Solution {
    public int numIslands(char[][] grid) {
        if(grid.length==0 || grid[0].length==0) return 0;
        int result = 0;
        for(int i=0; i<grid.length; i++)
            for(int j=0; j<grid[0].length; j++){
                if(grid[i][j] == '1'){
                    DFS(grid, i, j);
                    result++;
                }
            }
        return result;
    }
    private void DFS(char[][] grid, int i, int j){
        if(i<0 || j<0 || i>=grid.length || j>=grid[0].length || grid[i][j] != '1')
            return;
        grid[i][j] = 'x';
        DFS(grid, i-1, j);
        DFS(grid, i+1, j);
        DFS(grid, i, j+1);
        DFS(grid, i, j-1);
    }
}
```

### 131. Palindrome Partitioning
- [Link](https://leetcode.com/problems/palindrome-partitioning/)
- Tags: Backtracking
- Stars: 2
- reviewFlag

#### 2019.9.6 backtracking
- time: 97.28%
- space: 100%
```java
class Solution {
    String s;
    int len;
    List<List<String>> ret = new ArrayList<>();
    public List<List<String>> partition(String s) {
        this.s = s;
        len = s.length();
        backtrack(new ArrayList<>(), 0);
        return ret;
    }
    public void backtrack(List<String> currList, int start) {
        if (start == len) {
            ret.add(new ArrayList<>(currList));
            return ;
        }
        for(int i=start; i<len; i++) {
            if (isPalindrome(s, start, i)) {
                currList.add(s.substring(start, i+1));
                backtrack(currList, i+1);
                currList.remove(currList.size()-1);
            }
        }
    }
    public boolean isPalindrome(String s, int i, int j) {
        while(i<j) {
            if (s.charAt(i++) != s.charAt(j--)) return false;
        }
        return true;
    }
}
```

Optimized 2019.9.6 backtracking + memoization
- time: 100%
- space: 100%
```java
class Solution {
    String s;
    int len;
    List<List<String>> ret = new ArrayList<>();
    int[][] dp;
    public List<List<String>> partition(String s) {
        this.s = s;
        len = s.length();
        dp = new int[len][len];
        backtrack(new ArrayList<>(), 0);
        return ret;
    }
    public void backtrack(List<String> currList, int start) {
        if (start == len) {
            ret.add(new ArrayList<>(currList));
            return ;
        }
        for(int i=start; i<len; i++) {
            if (isPalindrome(s, start, i)) {
                currList.add(s.substring(start, i+1));
                backtrack(currList, i+1);
                currList.remove(currList.size()-1);
            }
        }
    }
    public boolean isPalindrome(String s, int l, int r) {
        if (dp[l][r] != 0) return dp[l][r] == 1;
        int i=l, j=r;
        while(i<j) {
            if (s.charAt(i++) != s.charAt(j--)) {
                dp[l][r] = -1;
                return false;
            }
        }
        dp[l][r] = 1;
        return true;
    }
}
```

#### Manacher's Algorithm
- cheatFlag
The Manacher method is copied from [a CSDN blog](https://blog.csdn.net/u014771464/article/details/79120964)
```java
class Solution {
    List<List<String>> result = new ArrayList<>();
    public List<List<String>> partition(String s) {
        int[] p = ManacherArray(s);
        backtrack(s, 0, p, new ArrayList<>());
        return result;
    }
    private void backtrack(String s, int start, int[] p, List<String> list){
        if(start == s.length()){
            result.add(list);
            return ;
        }
        int idx = 2*(start+1);
        for(int i=idx; i<p.length; i++){
            int len = p[i]-1;
            int left = i - len + 1;
            int diff = start - (left/2 - 1);
            if(diff >= 0){
                List<String> newList = new ArrayList<>(list);
                newList.add(s.substring(start, start+len-diff*2));
                backtrack(s, start+len-diff*2, p, newList);
            }
        }
    } 
    public int[] ManacherArray(String s) {
        // Insert '#'
        String t = "$#";
        for (int i = 0; i < s.length(); ++i) {
            t += s.charAt(i);
            t += "#";
        }
        t += "@";
        // Process t
        int[] p = new int[t.length()];;
        int mx = 0, id = 0, resLen = 0, resCenter = 0;
        for (int i = 1; i < t.length()-1; ++i) {
            p[i] = mx > i ? Math.min(p[2 * id - i], mx - i) : 1;
            while (((i - p[i])>=0) && 
                   ((i + p[i])<t.length()-1) && 
                   (t.charAt(i + p[i]) == t.charAt(i - p[i])))
                ++p[i];
            if (mx < i + p[i]) {
                mx = i + p[i];
                id = i;
            }
            if (resLen < p[i]) {
                resLen = p[i];
                resCenter = i;
            }
        }
        return p;
    }
}
```

### 39. Combination Sum
- [Link](https://leetcode.com/problems/combination-sum/)
- Tags: Array, Backtracking
- Stars: 2

#### general backtracking solution
```java
class Solution {
    int[] candidates;
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        this.candidates = candidates;
        backtrack(new ArrayList<>(), 0, target);
        return result;
    }
    private void backtrack(List<Integer> currList, int start, int target) {
        if(target == 0) {
            result.add(new ArrayList<>(currList));
            return ;
        }
        for(int i=start; i<candidates.length; i++) {
            if(candidates[i] > target) return ;
            currList.add(candidates[i]);
            backtrack(currList, i, target-candidates[i]);
            currList.remove(currList.size()-1);
        }
    }
}
```

### 40. Combination Sum II
- [Link](https://leetcode.com/problems/combination-sum-ii/)
- Tags: Array, Backtracking
- Stars: 2

#### general backtracking solution
```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        backtrack(candidates, 0, target, new ArrayList<>());
        return result;
    }
    private void backtrack(int[] candidates, int start, int target, List<Integer> currList){
        if(target == 0){
            result.add(new ArrayList<>(currList));
            return ;
        }
        if(start == candidates.length || target < candidates[start]) return ;
        int idx = start+1, num = candidates[start];
        while(idx < candidates.length && candidates[idx] == num) idx++; // get the index of the next distinct candidate
        // case 1: do not use `num`
        backtrack(candidates, idx, target, currList);
        // case 2: use `num`
        for(int i=start; i<idx; i++){
            currList.add(candidates[i]);
            target -= num;
            backtrack(candidates, idx, target, currList);
        }
        for(int i=0; i<idx-start; i++) currList.remove(currList.size()-1);
    }
}
```

#### genenral backtracking solution in a better way
```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        backtrack(candidates, 0, target, new ArrayList<>());
        return result;
    }
    private void backtrack(int[] candidates, int start, int target, List<Integer> currList){
        if(target == 0){
            result.add(new ArrayList<>(currList));
            return ;
        }
        if(start == candidates.length || target < candidates[start]) return ;
        for(int i=start; i<candidates.length; i++){
            if(i>start && candidates[i] == candidates[i-1]) continue;
            currList.add(candidates[i]);
            backtrack(candidates, i+1, target-candidates[i], currList);
            currList.remove(currList.size()-1);
        }
    }
}
```

### 79. Word Search
- [Link](https://leetcode.com/problems/word-search/)
- Tags: Array, Backtracking
- Stars: 2

#### simple backtracking solution beats 99.56% in time and 75.11% in space
```java
class Solution {
    char[][] board;
    String word;
    boolean[][] used;
    public boolean exist(char[][] board, String word) {
        if(word.length() == 0) return true;
        if(board.length == 0 || board[0].length == 0) return false;
        this.board = board;
        this.word = word;
        this.used = new boolean[board.length][board[0].length];
        for(int i=0; i<board.length; i++)
            for(int j=0; j<board[0].length; j++)
                if(recurr(i, j, 0)) return true;
        return false;
    }
    private boolean recurr(int i, int j, int start){
        if(start == word.length()) return true;
        if(i<0 || j<0 || i>=board.length || j>=board[0].length || used[i][j] || board[i][j] != word.charAt(start)) return false;
        used[i][j] = true;
        if(recurr(i+1, j, start+1) || recurr(i-1, j, start+1) || recurr(i, j+1, start+1) || recurr(i, j-1, start+1)) return true;
        used[i][j] = false;
        return false;
    }
}
```

#### 2019.8.11 backtrack O(n^2 * 4^k)
- time: 99.90%
- space: 97.96%
- thoughts: Generally, a matrix-related problem is a DP or DFS problem

```java
class Solution {
    public boolean exist(char[][] board, String word) {
        if (board.length == 0 || board[0].length == 0) return false;
        for(int i=0; i<board.length; i++) 
            for(int j=0; j<board[0].length; j++) 
                if (search(board, word, i, j, 0)) return true;
        return false;
    }
    public boolean search(char[][] board, String word, int i, int j, int start) {
        if (start == word.length()) return true;
        if (i<0 || i>=board.length || j<0 || j>=board[0].length) return false;
        char c = board[i][j];
        if (c == ' ' || c != word.charAt(start)) return false;
        board[i][j] = ' ';
        boolean ans = search(board, word, i-1, j, start+1) || 
            search(board, word, i+1, j, start+1) || 
            search(board, word, i, j-1, start+1) || 
            search(board, word, i, j+1, start+1);
        board[i][j] = c;
        return ans;
    }
}
```

## N Sums Questions

### 1. Two Sum
- [Link](https://leetcode.com/problems/two-sum/)
- Tags: Array, Hash Table
- Stars: 1

#### HashMap
```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for(int i=0; i<nums.length; i++){
            if(map.containsKey(target-nums[i]))
                return new int[] {map.get(target-nums[i]), i};
            map.put(nums[i], i);
        }
        throw new IllegalArgumentException("No two sum solution");
    }
}
```

### 454. 4Sum II
- [Link](https://leetcode.com/problems/4sum-ii/)
- Tags: Hash Table, Binary Search
- Stars: 2

#### HashMap + two sum
```java
class Solution {
    public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        int result = 0;
        HashMap<Integer, Integer> map1 = twoSum(A, B);
        HashMap<Integer, Integer> map2 = twoSum(C, D);
        for(Map.Entry<Integer, Integer> e1: map1.entrySet()){
            if(map2.containsKey(-e1.getKey())){
                result += e1.getValue() * map2.get(-e1.getKey());
            }
        }
        return result;
    }
    private HashMap<Integer, Integer> twoSum(int[] A, int[] B){
        HashMap<Integer, Integer> map = new HashMap<>();
        for(int a: A)
            for(int b: B)
                map.put(a+b, map.getOrDefault(a+b, 0) + 1);
        return map;
    }
}
```

### 15. 3Sum
- [Link](https://leetcode.com/problems/3sum/)
- Tags: Array, Two Pointers
- Stars: 2

#### HashSet, clear but slow
```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> threeSum(int[] nums) {
        if(nums.length == 0) return result;
        HashSet<Integer> set = new HashSet<>();
        for(int num: nums) set.add(num);
        Arrays.sort(nums);
        for(int i=0; i<nums.length-2; i++){
            if(i>0 && nums[i] == nums[i-1]) continue;
            for(int j=i+1; j<nums.length-1; j++){
                if(j>i+1 && nums[j] == nums[j-1]) continue;
                if(set.contains(-(nums[i]+nums[j])) && -(nums[i]+nums[j]) >= nums[j+1])
                    result.add(Arrays.asList((Integer)nums[i], (Integer)nums[j], (Integer)(-(nums[i]+nums[j]))));
            }
        }
        return result;
    }
}
```

#### jumping iteration + Two Sum two pointers
```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        
        Arrays.sort(nums);
        for(int i=0; i<nums.length-2; i++){
            if(i>0 && nums[i] == nums[i-1]) continue;
            int j=i+1, k=nums.length-1;
            while(j<k){
                if(nums[i]+nums[j]+nums[k] == 0) {
                    result.add(Arrays.asList((Integer)nums[i], (Integer)nums[j++], (Integer)nums[k--]));
                    while(j<k && nums[j] == nums[j-1]) j++;
                    while(j<k && nums[k] == nums[k+1]) k--;
                }
                else if(nums[i]+nums[j]+nums[k]>0) k--;
                else j++;
            }
        }
        return result;
    }
}
```

Updated 2019.8.4
```java
class Solution {
    List<List<Integer>> result = new ArrayList<>();
    public List<List<Integer>> threeSum(int[] nums) {
        if (nums.length == 0) return result;
        Arrays.sort(nums);
        int i=0;
        while (i<nums.length - 2) {
            int l = i+1, r = nums.length - 1;
            while (l < r) {
                int sum = nums[i] + nums[l] + nums[r];
                if (sum < 0) l++;
                else if (sum > 0) r--;
                else {
                    result.add(Arrays.asList(nums[i], nums[l], nums[r]));
                    l = moveToNext(nums, l, +1);
                    r = moveToNext(nums, r, -1);
                }
            }
            i = moveToNext(nums, i, +1);
        }
        return result;
    }
    public int moveToNext(int[] nums, int i, int incre) {
        if (i >= nums.length || i<0) return i;
        int num = nums[i];
        do {
            i += incre;
        } while (i>=0 && i<nums.length && nums[i] == num);
        return i;
    }
}
```

# Weekly Contests

## No. 152
### 5175. Can Make Palindrome from Substring
- [Link](https://leetcode.com/problems/can-make-palindrome-from-substring/)
- Tags: Array, String
- Stars: 3

#### 2019.9.1 
- time 65 ms
- space 119.9 MB

`count` is the number of characters that occurs odd times
```java
class Solution {
    public List<Boolean> canMakePaliQueries(String s, int[][] queries) {
        List<Boolean> ret = new ArrayList<>();
        int[][] stat = new int[s.length()][26];
        for(int i=0; i<s.length(); i++) {
            if (i > 0) stat[i] = Arrays.copyOf(stat[i-1], 26);
            char c = s.charAt(i);
            stat[i][c-'a']++;
        }
        for(int[] query: queries) {
            int count = 0;
            for(int i=0; i<26; i++) {
                if ((stat[query[1]][i] - (query[0] > 0 ? stat[query[0]-1][i] : 0)) % 2 == 0) continue;
                count++;
            }
            if (count - 2*query[2] > 1) ret.add(false);
            else ret.add(true);
        }
        return ret;
    }
}
```

### 5176. Number of Valid Words for Each Puzzle
- [Link](https://leetcode.com/problems/number-of-valid-words-for-each-puzzle/)
- Tags: Hash Table, Bit Manipulation
- Stars: 3

#### 2019.9.1 bit manipulation
- time 1536 ms
- space 50.6 MB
- attention: need to filter invalid words first!
```java
class Solution {
    public List<Integer> findNumOfValidWords(String[] words, String[] puzzles) {
        List<Integer> ret = new ArrayList<>();
        int[] stat = new int[words.length];
        int len = 0;
        for(int i=0; i<words.length; i++) {
            int temp = 0;
            for(char c: words[i].toCharArray()) temp |= (1<<(c-'a'));
            if (Integer.bitCount(temp) > 7) continue;
            stat[len++] = temp;
        }
        for(String puzzle: puzzles) {
            int count = 0, curr = 0;
            char first = puzzle.charAt(0);
            for(char c: puzzle.toCharArray()) curr |= (1<<(c-'a'));
            for(int i=0; i<len; i++) if (((stat[i] | curr) == curr) && (stat[i] & (1<<(first-'a'))) != 0) count++;
            ret.add(count);
        }
        return ret;
    }
}
```


## No. 96
### 881. Boats to Save People
- [Link](https://leetcode.com/problems/boats-to-save-people/)
- Tags: Two Pointers, Greedy
- Stars: 1

#### two pointers
```java
class Solution {
    public int numRescueBoats(int[] people, int limit) {
        Arrays.sort(people);
        int i=0, j=people.length-1;
        int result = 0;
        while(i<=j){
            if(i==j){
                result++;
                break;
            }
            if(people[i]+people[j] > limit){
                j--;
            }
            else{
                i++;
                j--;
            }
            result++;
        }
        return result;
    }
}
```

### 880. Decoded String at Index
- [Link](https://leetcode.com/problems/decoded-string-at-index/)
- Tags: Stack
- Stars: 3

#### Iterative
```java
class Solution {
    public String decodeAtIndex(String S, int K) {
        List<Tuple> list = getTuples(S);
        int pos = -1;
        for(int i=0; i<list.size(); i++)
            if(list.get(i).accu >= K) pos = i;
        int curr = K-1;
        while(pos>0){
            curr %= list.get(pos).curr;
            if(curr >= list.get(pos-1).accu) {
                curr -= list.get(pos-1).accu;
                return Character.toString(list.get(pos).str.charAt(curr));
            }
            pos--;
        }
        curr %= list.get(0).curr;
        return Character.toString(list.get(0).str.charAt(curr));
    }
    private List<Tuple> getTuples(String S){
        S = S + "1";
        List<Tuple> result = new ArrayList<>();
        int lastIdx = 0;
        for(int i=0; i<S.length(); i++){
            char c = S.charAt(i);
            if(!Character.isLetter(c)){
                if(lastIdx == i){
                    result.get(result.size()-1).repeat *= c-'0';
                    lastIdx = i+1;
                }
                else {
                    Tuple tup = new Tuple(S.substring(lastIdx, i), c-'0', i-lastIdx);
                    result.add(tup);
                    lastIdx = i+1;
                }
            }
        }
        result.get(0).accu = result.get(0).curr * result.get(0).repeat;
        for(int i=1; i<result.size(); i++){
            result.get(i).curr = (result.get(i-1).accu + result.get(i).curr);
            result.get(i).accu = result.get(i).curr * result.get(i).repeat;
        }
        return result;
    }
}
class Tuple{
    String str;
    int repeat;
    long accu; // accumulative length after repeat
    long curr; // accumulative length before repeat
    public Tuple(String s, int r, long a){
        str = s;
        repeat = r;
        curr = a;
    }
}
```

#### recursive
- attention: strLen might OVERFLOW!!!!!!! Thus, we must use long. 

```java
class Solution {
    public String decodeAtIndex(String S, int K) {
        long strLen = 0;
        for(int i=0; i<S.length(); i++){
            char c = S.charAt(i);
            if(Character.isLetter(c)){
                if(++strLen == K) return Character.toString(c);
            }
            else {
                int repeat = c-'0';
                if(strLen * repeat >= K) 
                    return decodeAtIndex(S.substring(0, i), (int)((K-1)%strLen+1));
                strLen *= repeat;
            }
        }
        return null;
    }
}
```

# TODO List

## skipped problems

- 208 Implement Trie
- 227 Basic Calculator II
- 324 Wiggle Sort II -- Explanation/Proof for the correctness, Virtual Indexing
- 5 Longest Palindromic Substring
- 53 Maximum Subarray -- divide and conquer
- 435 Non-overlapping Intervals (已经有###了)
- explore more solutions of 43. Multiply Strings
- explore more solutions of 673. Number of Longest Increasing Subsequence
- 440 K-th Smallest in Lexicographical Order

## recursive to non-recursive

[101. Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)  
[148. Sort List](https://leetcode.com/problems/sort-list/)  
[104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)  
[226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)  
[230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)  
[173. Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/)  


- [tree questions](https://leetcode.com/problems/validate-binary-search-tree/discuss/32112/Learn-one-iterative-inorder-traversal-apply-it-to-multiple-tree-questions-(Java-Solution))

## Math

- 202. Happy Number
https://leetcode.com/problems/happy-number/discuss/56918/All-you-need-to-know-about-testing-happy-number!

## Bit Manipulation

- (n-1)在位运算中的作用？








