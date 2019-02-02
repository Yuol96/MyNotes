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