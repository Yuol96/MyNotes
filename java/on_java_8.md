[TOC]

## What is an Object?
### Collection
- Java uses dynamic memory allocation, exclusively. 

### Exception Handling
Exceptions cannot be ignored, so ti's guaranteed to be dealt with at some point. 

Exception handling is not a OO feature, although exception is normally represented by an object. 

## Objects Everywhere

### 对象操纵
####Special case: Primitive Types
They are created without new and are placed on the stack.
Much more efficient!

Attention: char is a 16-bit type.  
(Unicode) min = Unicode 0, max = Unicode $2^{16}-1$ 

#### High-precision numbers
`BigInteger` and `BigDecimal`

#### Arrays in java
Java 的设计主要目标之一是安全性  
在 Java 中，数组使用前需要被初始化  
并且不能访问数组长度以外数据  
这种长度检查的代价是每个阵列都有少量的内存开销以及在运行时验证索引的额外时间

#### Scoping
在nested的作用域中使用相同的变量名是不被允许的
```java
{
	int x = 1;
	{
		int x = 10; // Illegal!!
	}
}
```

### Creating new data types: class

#### Default values for primitive members
Type | Default Values
---- | ----
boolean | false
char | \u0000
byte | 0
short | 0
int | 0
long | 0L
float | 0.0f
long | 0.0d

这种默认值的赋予并不适用于局部变量 —— 那些不属于类的属性的变量。
e.g. in the global we write
```java
int x;
```
这里的变量 x 不会自动初始化为0，因而在使用变量 x 之前，程序员有责任主动地为其赋值（和 C 、C++ 一致）。如果我们忘记了这一步，在 JAVA 中将会提示我们“编译时错误，该变量尚未被初始化”。

### Writing a java Program

#### name visibility
Java uses reversed URLs.

这种方式似乎为我们在编写 Java 程序中的某个问题打开了大门。空目录填充了深层次结构，它们不仅用于表示反向 URL，还用于捕获其他信息。这些长路径基本上用于存储有关目录中的内容的数据。如果你希望以最初设计的方式使用目录，这种方法可以从“令人沮丧”到“令人抓狂”，对于产生的 Java 代码，你基本上不得不使用专门为此设计的 IDE 来管理代码。例如 NetBeans，Eclipse 或 IntelliJ IDEA。实际上，这些 IDE 都为我们管理和创建深度空目录层次结构。

#### using other components
必须通过使用 import 关键字来告诉 Java 编译器具体要使用的类。

#### First Java Program
- `java.lang` is always automatically included in every Java file. 
- `java.util.Date`
- `java.lang.System` has a static **out** object, which is a **PrintStream** type.   
`System.out.print`, `System.out.printf`, and `System.out.println`

#### Coding Style
- **camel-casing**: Capitalize the first letter of a <u>**class name**</u> as well as the first letter of all separate words.  
As for methods and variable names, don't campitalize the first letter of the identifier. 

## Operators
- `Integer.toBinaryString(0x2f)`  output the binary string
- `>>>` and `>>>=`
- `while()`中的条件不会自动转化为boolean，如果不是boolean会报错。
- boolean does not allow any casting at all.  
Class types do not allow casting except for casting within the same family. 
- When converting `float` or `double` to an integer value, it always truncates the number instead of using `java.lang.Math.round()`
- **promotion**
	- 当对char，byte，shor做运算时，这些数值会被自动转化成int再运算，运算结果需要显示cast才能变回smaller type.
	- In general, the largest data type in an expression is the one that determines the size of the result of that expression. 
- Java does not have `sizeof()` because java doesn't need it. All the data types are the same size on all machines. 

## Control Flow

**The comma operator**  
Used in for loop. You can have a number of statements separated by commas.

**For-In Syntax**  
Used for Strings, Arrays, and any iterable Object.  

## House Keeping

When passed a value/variable x into a function `func(T t)`  
- if x is a constant value, x are treated as int  
- if x == char and T != char, x is promoted to int.

Cannot call non-static methods from inside static methods. also cannnot use `this`.

`finalize()`









