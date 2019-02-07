[TOC]
# LeetCode Questions

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
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
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
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
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
- Stars: 3

#### Heavy Guardian (Moore Voting)
```java
class Solution {
    public int majorityElement(int[] nums) {
        int majority = nums[0], count = 1;
        for(int i=1; i<nums.length; i++){
            if(count == 0){
                majority = nums[i];
                count = 1;
            }
            else {
                if(majority == nums[i])
                    count++;
                else
                    count--;
            }
        }
        return majority;
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

### 198. House Robber
- [Link](https://leetcode.com/problems/house-robber/)
- Tags: Dynamic Programming
- Stars: 1

#### DP iterative
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

### 66. Plus One
- [Link](https://leetcode.com/problems/plus-one/)
- Tags: Array, Math
- Stars: 1

#### 数组初始化
注意：默认初始化，数组元素相当于对象的成员变量，默认值跟成员变量的规则一样。**数字0**，布尔false，char\u0000，引用：null

本题不适合把`Arrays.asList()`转化为List, `.asList`方法不适用于基本数据类型（byte,short,int,long,float,double,boolean）
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

### 1. Two Sum
- [Link](https://leetcode.com/problems/two-sum/)
- Tags: Array, Hash Table
- Stars: 2

#### HashMap
```java
class Solution {
    HashMap<Integer, Integer> map;
    
    public int[] twoSum(int[] nums, int target) {
        map = new HashMap<Integer, Integer>();
        for(int i=0; i<nums.length; i++){
            if(map.containsKey(target-nums[i])){
                int[] ret = {map.get(target-nums[i]), i};
                return ret;
            }
            else {
                map.put(nums[i], i);
            }
        }
        return null;
    }
}
```

### 172. Factorial Trailing Zeroes
- [Link](https://leetcode.com/problems/factorial-trailing-zeroes/)
- Tags: Math
- Stars: 2

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
- Stars: 3

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
    // private void printList(ListNode head){
    //     while(head!=null){
    //         System.out.printf("%d ", head.val);
    //         head = head.next;
    //     }
    //     System.out.println();
    // }
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
```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA==null||headB==null)
            return null;
        ListNode p=headA, q=headB;
        boolean switchA=false, switchB=false;
        while(p!=q){
            if(p.next!=null)
                p = p.next;
            else if(!switchA){
                p = headB;
                switchA = true;
            }
            else return null;
            if(q.next!=null)
                q = q.next;
            else if(!switchB){
                q = headA;
                switchB = true;
            }
            else return null;
        }
        return p;
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

# TODO List

## recursive to non-recursive

- 101. Symmetric Tree

## Math

- 202. Happy Number
https://leetcode.com/problems/happy-number/discuss/56918/All-you-need-to-know-about-testing-happy-number!

## Bit Manipulation

- (n-1)在位运算中的作用？








