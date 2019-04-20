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
- Stars: 2

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

#### Heavy Guardian (Moore Voting)
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
Attention that you `r-l` might overflow, so you have to use long integer.
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
        if(prices.length == 0)
            return 0;
        int maxProfit = 0, dp = 0;
        for(int i=1; i<prices.length; i++){
            dp = Math.max(0, dp + prices[i] - prices[i-1]);
            maxProfit = Math.max(maxProfit, dp);
        }
        return maxProfit;
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
- Stars: 3

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
```java
class Solution {
    public int rob(int[] nums) {
        if(nums.length == 0) return 0;
        if(nums.length == 1) return nums[0];
        return Math.max(rob(nums, 0, nums.length-1), rob(nums, 1, nums.length));
    }
    public int rob(int[] nums, int start, int end){
        if(start >= end) return 0;
        int[] dp = new int[end-start];
        dp[0] = nums[start];
        for(int i=start+1; i<end; i++){
            dp[i-start] = nums[i];
            if(i-2>=start) dp[i-start]+=dp[i-2-start];
            if(i-3>=start) dp[i-start] = Math.max(dp[i-start], dp[i-3-start]+nums[i]);
        }
        int result = dp[end-1-start];
        if(end-2>=start) result = Math.max(result, dp[end-2-start]);
        return result;
    }
}
```

### 337. House Robber III
- [Link](https://leetcode.com/problems/house-robber-iii/)
- Tags: Tree, DFS
- Stars: 3

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
Attention that `k` needs to be reduced to [0, nums.length).
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
- Stars: 1

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
- Stars: 1

#### swap step by step
```java
class Solution {
    private int[] arr;
    public Solution(int[] nums) {
        arr = nums;
    }
    /** Resets the array to its original configuration and return it. */
    public int[] reset() {
        return arr;
    }
    /** Returns a random shuffling of the array. */
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
<span id="378-binary-search"></span>
1. Attention: when `count == k`, `mid` might not exists in `matrix`, so we need to get the largest element that is less than or equal to `mid` in `matrix`. Therefore, we have `getMaxlte`.
2. There's a situation that might break the while loop, i.e., there are more than one elements that have the same value as the kth smallest. When this happens, r will goes below l, and it breaks the while loop. Therefore, we need to return `l` instead of an arbitrary number outside the while loop. 
3. The whole picture of this algorithm:
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
- Stars: 2


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
1. When implementing `HashArray.equals()`, the parameter `o` must be of type `Object`!!
2. Pay attention to the usage of `map.computeIfAbsent` and its return value. 
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

### 11. Container With Most Water
- [Link](https://leetcode.com/problems/container-with-most-water/)
- Tags: Array, Two Pointers
- Stars: 3

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

#### swap
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
        if(!map.containsKey(val))  return false;
        int idx = map.get(val);
        if(idx < list.size()-1){
            int lastone = list.get(list.size()-1);
            map.put(lastone, idx);
            list.set(idx, lastone);
        }
        list.remove(list.size()-1);
        map.remove(val);
        return true;
    }
    /** Get a random element from the set. */
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
- Stars: 1

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
- Stars: 3

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

### 207. Course Schedule
- [Link](https://leetcode.com/problems/course-schedule/)
- Tags: BFS, DFS, Graph, Topological Sort
- Stars: 1

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

### 19. Remove Nth Node From End of List
- [Link](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)
- Tags: Linked List, Two Pointers
- Stars: 3

#### one pass solution
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
- Stars: 3

#### sort
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

#### Method 3: use binary search for 3 times
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

### 150. Evaluate Reverse Polish Notation
- [Link](https://leetcode.com/problems/evaluate-reverse-polish-notation/)
- Tags: Stack
- Stars: 1

#### stack
1. `token.length()>1` is used to deal with negative numbers.
2. pay attention to the order of parameters in `compute` function
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
- Stars: 1

#### simple solution beats 91.83% in time and 96.99% in space
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

### 152. Maximum Product Subarray
- [Link](https://leetcode.com/problems/maximum-product-subarray/)
- Tags: Array, Dynamic Programming
- Stars: 3

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

### 179. Largest Number
- [Link](https://leetcode.com/problems/largest-number/)
- Tags: Sort
- Stars: 2

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
- Stars: 1

#### recursive
Attention that you need to take care of cases like `root.val == Integer.MIN_VALUE` and `root.val == Integer.MAX_VALUE`, because under these circumstances, the boundaries might overflow.
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
- Stars: 1

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

### 91. Decode Ways
- [Link](https://leetcode.com/problems/decode-ways/)
- Tags: String, Dynamic Programming
- Stars: 1

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

### 69. Sqrt(x)
- [Link](https://leetcode.com/problems/sqrtx/)
- Tags: Math, Binary Search
- Stars: 1

#### binary search
```java
class Solution {
    public int mySqrt(int x) {
        int l = 0, r = 46340;
        while(l+1<r) {
            int mid = l + ((r-l)>>1);
            if(mid*mid == x) return mid;
            else if(mid*mid > x) r = mid-1;
            else l = mid;
        }
        if(r*r <= x) return r;
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
- Stars: 3

#### sort and reverse, O(nlogn) time 
This solution can be optimized by virtual indexing! 

[Explanation/Proof for the correctness](https://leetcode.com/problems/wiggle-sort-ii/discuss/77678/3-lines-Python-with-Explanation-Proof)
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

### 128. Longest Consecutive Sequence
- [Link](https://leetcode.com/problems/longest-consecutive-sequence/)
- Tags: Array, Union Find
- Stars: 3

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
- Stars: 3

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

### 295. Find Median from Data Stream
- [Link](https://leetcode.com/problems/find-median-from-data-stream/)
- Tags: Heap, Design
- Stars: 2

#### lower bound binary search insertion
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
- Stars: 2

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
ATTENTION: `isSubtree` and `isEqual` are different DFS process. Do not try to integrate into a single function.
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
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0,1);
        for(int i=1; i<=n; i++){
            int temp = 0;
            for(int j=0; j<i; j++){
                temp += map.get(j) * map.get(i-1-j);
            }
            map.put(i, temp);
        }
        return map.get(n);
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

#### O(n) time
```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if(head == null) return null;
        ListNode curr = head;
        while(curr.next != null){
            if(curr.next.val == curr.val) curr.next = curr.next.next;
            else curr = curr.next;
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

#### lower bound binary search insertion
```java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int l = 0, r = nums.length;
        while(l<r){
            int mid = l + ((r-l)>>1);
            if(nums[mid] == target) return mid;
            else if(nums[mid] < target) l = mid+1;
            else r = mid;
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

### 47. Permutations II
- [Link](https://leetcode.com/problems/permutations-ii/)
- Tags: Backtracking
- Stars: 2

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

#### sub-optimal palindrome method
```java
class Solution {
    List<List<String>> result = new ArrayList<>();
    public List<List<String>> partition(String s) {
        backtrack(s, 0, new ArrayList<>());
        return result;
    }
    private void backtrack(String s, int start, List<String> list){
        if(start == s.length()){
            result.add(list);
            return ;
        }
        for(int i=start; i<s.length(); i++){
            if(isPalindrome(s, start, i)){
                List<String> newList = new ArrayList<>(list);
                newList.add(s.substring(start, i+1));
                backtrack(s, i+1, newList);
            }
        }
    }
    private boolean isPalindrome(String s, int i, int j){
        while(i<j){
            if(s.charAt(i++)!=s.charAt(j--))
                return false;
        }
        return true;
    }
}
```

#### Manacher's Algorithm 
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
- Stars: 1

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

# Weekly Contests

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
Attention: strLen might OVERFLOW!!!!!!! Thus, we must use long. 
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
[94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)  
[144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)  
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








