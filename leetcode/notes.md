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

<span id="378-binary-search"></span>
#### Binary Search
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

<span id="287-binary-search"></span>
#### Binary Search
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

<span id="287-two-pointers"></span>
#### slow-fast two pointers
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

<span id="142-two-pointers"></span>
#### slow-fast two pointers
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

# Topics

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

#### My Backtracking Solution
```java
class Solution {
    List<List<Integer>> result;
    public List<List<Integer>> permute(int[] nums) {
        result = new ArrayList<>();
        if(nums.length == 0) return result;
        List<Integer> firstList = new ArrayList<>();
        firstList.add(nums[0]);
        result.add(firstList);
        backtrack(nums, 1);
        return result;
    }
    private void backtrack(int[] nums, int k){
        if(k == nums.length)
            return ;
        int len = result.size();
        for(int i=0; i<len; i++){
            List<Integer> list = result.get(i);
            for(int j=0; j<list.size(); j++){
                List<Integer> temp = new ArrayList<>(list);
                temp.add(j, nums[k]);
                result.add(temp);
            }
            list.add(list.size(), nums[k]);
        }
        backtrack(nums, k+1);
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

## N Sums Questions

### 1. Two Sum
- [Link](https://leetcode.com/problems/two-sum/)
- Tags: Array, Hash Table
- Stars: 1

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

# TODO List

## recursive to non-recursive

- 101. Symmetric Tree
- 94. Binary Tree Inorder Traversal

## Math

- 202. Happy Number
https://leetcode.com/problems/happy-number/discuss/56918/All-you-need-to-know-about-testing-happy-number!

## Bit Manipulation

- (n-1)在位运算中的作用？








