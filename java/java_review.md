[TOC]

# Basic Java grammers

## Basic types, classes, and functions
### String
- def: `String s = "xxxxxx";`
- properties
```java
int len = s.length();
char c = s.charAt(2);
int idx = s.indexOf('a');
boolean bl = s.startsWith("duke");  // s.endsWith("university")
```
- operations
```java
String cat = "Hello" + " world!";
boolean bl = s1.equals(s2);
String subs = s.substring(4,7);
s.split("\\W") //Divides between “non-word” characters
int result = str1.compareTo(str2);
```
- <span style="background-color: yellow;">Implicitly uses `.toString()` in `System.out.println(anObject)`</span>String

### StringBuilder

**Mutable** sequence of characters

- properties:
	- append, insert, charAt, setCharAt, toString

```java
StringBuilder sb = new StringBuilder("Hello");
sb.append(" World");
sb.insert(5, " Around the ");
```

### char (Character)

The type char is primitive, single quotes.

Properties: isLowerCase(ch), isDigit(ch), toLowerCase(ch), toUpperCase(ch)

```java
char c = Character.toLowerCase(ch);
```

### Array

```java
int[] a = new int[256];
String[] sArray = new String[12];

for(int i=0; i<a.length; i++) 
	int val = a[k];
for(String s: sArray){
	System.out.println(s);
}
```

### I/O

`println` will automatically call `.toString()`
```java
System.out.println("xxx");
System.out.print("%d ", num);
```

## java.util
### java.util.Random
<span style="background-color: yellow;">don't know</span>
```java
Random rand = new Random();
int idx = rand.nextInt(10);

Math.random()
```
### java.util.ArrayList

Indexable collection, like array, but growable!

Methods: add(), size(), get(), set(), indexOf(), <span style="background-color: yellow;">contains()</span>

> 感觉类似python中的list和C++中的vector

```java
import java.util.ArrayList // import java.util.*;

ArrayList<String> words = new ArrayList<String>();  // ArrayList<Integer>
words.add("hello");
String s = words.get(1);
words.set(0, "goodbye");
System.out.println(words.size());
```

**Do not call `.remove()` during iteration**

### HashMap

Methods: put(), size(), get(), keySet(), containsKey()
```java
HashMap<String, Integer> map = new HashMap<String, Integer>();

map.put("Mike", 99);
map.put("Mike", map.get("Mike")+1);

for(String s: map.keySet()) {
	System.out.println(map.get(s));
}
```

### HashSet
Methods: .add(ele), .contains(ele)

```java
for(String s: st) {...}
```

### Comparator
method1: `Collections.sort(list,Comparator<T>);`

Collections.sort(list，Collections.reverseOrder())

method2: `list.sort(Comparator<T>)`

```java
Collections.sort(list, new Comparator<Student>() {
    @Override
    public int compare(Student o1, Student o2) {
        return o1.getId() - o2.getId();
    }
});
```


## conventions
### equality
- `==` means exactly same objects
- `.equals()` means same meaning, defined as `==` by default, therefore need redefine!

# Java Object Oriented Programming

### rules
- 每个public class必须放在单独的文件中，文件名与public class的名字相同
- 关于this： this is optional
- overload必须有不同的参数，仅仅有不同的return type是不够的
- 关于main
	- 每个program都必须有`public static void main(String[] args)`
	- 可以有多个class有main函数，但是只有一个可以被选择执行，
	```java
	javac *.java
	java <class-name>
	```
- 每个class的构造函数中的第一行都应该有一个`super(args)` （或`this(args)`），否则编译器会自动加进去

### inheritance, polymorphism
声明：
```java
public class Student extends Person {
	//...
}
```
Reference type (compile time) & Object type (runtime)
```java
Person p = new Student(); 
// this is ok!

int m = p.getScore();  
//This will cause an compile error! 
//Because Person class doesn't have .getScore() method, 
//and the compiler doesn't know it's a Student at compile time.

int m = ((Student)p).getScore()  
// Cast Person to Student. 
// Be careful, because the compiler will completely trust you!
```
Casting may cause problems, so we'd better do **<u>runtime type check</u>**! 
Use `instanceof` to check.
```java
if(s isinstanceof Student){
	int m = ((Student)p).getScore();
}
```

Also, pay attention to the difference between `super` and `this`!
The calls to `super.method1()` get bound to <u>compile time</u>, and `this.method1()` will use the actual type of the object at <u>runtime</u>.

#### Abstract class and Interface
If you want to define potentially required methods and common behavior, use <span style="color: blue;">Abstract class</span>.
If you just want to define a required method, use <span style="color: blue;">Interface</span>.

Class must be **abstract** if any methods are abstract!
```java
public abstract class Person {
	public abstract void monthlyStatement() {
		//...
	}
}
```

classes can inherit from multiple interfaces.
<span style="background-color: yellow;">What is interfaces? How does it works?</span>
Example of inheriting from an interface
```java
package java.lang;

public interface Comparable<E> {
	public abstract int compareTo(E o);
}

public class Person implements Comparable<Person> {
	private String name;
	//...

	@Override
	public int compareTo(Person o){
		return this.getName().compareTo(o.getName());
	}
}
```

### visibility
rule of thumb: **always use public or private**

Visibility | access from same class | from same package | from sub-class | from any class
----- | ----- | ----- | ----- | ----- 
public | v | v | v | v 
protected | v | v | v | x
package | v | v | x | x
private | v | x | x | x

### example
```java
public class Student {
	private String name;
	private int score;
	public Student(String s, int scr) {
		name = s;
		score = scr;
	}
}

Student mike = new Student("Mike", 100);
```

# 未整理
- 求最大值 `Math.max(a,b)`
- Arrays的sort函数 `Arrays.sort(xxx)`
- String and Integer conversion
```java
// String to Integer
Integer.parseInt(s);
// Integer to String
String.valueOf(num);
Integer.toString(num);
```
- <span style="background-color: yellow;">Difference between int and Interger???</span>
- `result += s.charAt(i)-'A'+1;`
- HashMap `map.getOrDefault(n, default)`
- 二维List的声明方法 `List<List<Integer>> = new ArrayList<List<Integer>>();`
- List<List<Integer>> allrows `allrows.add(new ArrayList<Integer>(row))` [reference](https://leetcode.com/problems/pascals-triangle/discuss/38141/My-concise-solution-in-Java)

从[LeetCode 118](https://leetcode.com/problems/pascals-triangle/)中学习 二维数组的声明，`list.add(0,1)`, `list.set(i, j)`, `allrows.add(new ArrayList<Integer>(row))`
```java
class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> output = new ArrayList<List<Integer>>();
        List<Integer> row = new ArrayList<Integer>();
        for(int i=0; i<numRows; i++){
            row.add(0, 1);
            for(int j=1; j<row.size()-1; j++){
                row.set(j, row.get(j)+row.get(j+1));
            }
            output.add(new ArrayList<Integer>(row));
        }
        return output;
    }
}
```
- 无符号右移用`>>>`，有符号用`>>`
- 注意`++`的使用技巧
[Source: 26. Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
```java
// nums[count] = nums[i];
// count++;
nums[count++] = nums[i];
```
- The use of `StringBuilder`
[Ref](https://blog.csdn.net/qq_33366229/article/details/78325689)

[Source: 38. Count and Say](https://leetcode.com/problems/count-and-say/)
```java
class Solution {
    public String countAndSay(int n) {
        if(n==1)
            return "1";
        String str = countAndSay(n-1);
        StringBuilder result = new StringBuilder();
        char last = str.charAt(0);
        int count = 1;
        for(int i=1; i<str.length(); i++){
            if(str.charAt(i) != last){
                // result = result + count + last;
                result.append(count).append(last);
                last = str.charAt(i);
                count = 1;
            }
            else count++;
        }
        result.append(count).append(last);
        return result.toString();
    }
}
```
- Use of `Stack<T>`
[Source: 20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)
```java
class Solution {
    public boolean isValid(String s) {
        Stack<Character> st = new Stack<Character>();
        for(int i=0; i<s.length(); i++){
            char c = s.charAt(i);
            if(c=='(') st.push(')');
            else if(c=='[') st.push(']');
            else if(c=='{') st.push('}');
            else{
                if(st.empty() || st.pop() !=c) return false;
            }
        }
        return st.empty();
    }
}
```
- `Arrays.toString(int[] nums)`, `Arrays.sort(int[] nums)`
- `str.toLowerCase()`, `Character.isLetterOrDigit(char c)`
- [在Java中，不存在Unsigned无符号数据类型，但可以轻而易举的完成Unsigned转换。](https://blog.csdn.net/qq_26386171/article/details/54564127)
- Java has the operator ">>>" to perform logical right shifts, but because the logical and arithmetic left-shift operations are identical, there is no "<<<" operator in Java. For those of you who are confused.
- [Java Array、List、Set互相转化](https://blog.csdn.net/u014532901/article/details/78820124)
- `Arrays.fill(f, false)`
- `ArrayList.subList(from, to).clear()` [Reference](https://www.cnblogs.com/ljdblog/p/6251387.html)
- [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
    - `HashMap<Integer, List<Integer>> map = new HashMap<>()` and `new ArrayList<>(list)` 后面的`<>`中的内容可省略.
    - `List<T>.addAll(List<T> list);`
    - `PriorityQueue<Map.Entry<Integer, Integer>> maxHeap = new PriorityQueue<>((a,b)->(b.getValue()-a.getValue()))`, `for(Map.Entry<Integer, Integer> entry: map.entrySet())` 
    - lambda function 
        - [Reference1](https://www.cnblogs.com/franson-2016/p/5593080.html)
        - [Reference2](https://www.jianshu.com/p/bde3699f37e5)
- `Arrays.copyOfRange(arr, i, j)`
- `nums.clone()` where `nums` is `int[]`
- `Collections.reverse(List<T> list)`
- `Java中的方法不可以有默认参数，只能通过重载来实现`
-  char在Java中是16位的，因为Java用的是Unicode。
- `Collections.swap(List<?> list, int i, int j)`
- `Arrays.binarySearch(int[] arr, int from, int to, int target)`
- Arrays has no function of indexOf