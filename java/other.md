### Mac OS X 下查看和设置JAVA_HOME
[ref](https://blog.csdn.net/done58/article/details/51138057)

安装JDK后，可以查看Java版本：
```bash
$ java -version
# java version "1.8.0_161"
# Java(TM) SE Runtime Environment (build 1.8.0_161-b12)
# Java HotSpot(TM) 64-Bit Server VM (build 25.161-b12, mixed mode)
```

**如何查看Java安装路径？**

无法通过以下方法得到真实的安装路径
```bash
$ which java
# /usr/bin/java
$ which javac
# /usr/bin/javac
```

根据[this article](https://developer.apple.com/library/mac/qa/qa1170/_index.html):
> Many Java applications need to know the location of a $JAVA_HOME directory. The $JAVA_HOME on Mac OS X should be found using the /usr/libexec/java_home command line tool on Mac OS X 10.5 or later. On older Mac OS X versions where the tool does not exist, use the fixed path "/Library/Java/Home". The /usr/libexec/java_home tool dynamically finds the top Java version specified in Java Preferences for the current user. This path allows access to the bin subdirectory where command line tools such as java, javac, etc. exist as on other platforms. The tool /usr/libexec/java_home allows you to specify a particular CPU architecture and Java platform version when locating a $JAVA_HOME.
>
> Another advantage of dynamically finding this path, as opposed to hardcoding the fixed endpoint, is that it is updated when a new version of Java is downloaded via Software Update or installed with a newer version of Mac OS X. For this reason, it is important that developers do not install files in the JDKs inside of /System, since the changes will be lost with subsequent updates by newer versions of Java.
>
> To obtain the path to the currently executing $JAVA_HOME, use the java.home System property.

我们来验证一下:
```bash
$ ls -l /usr/libexec/java_home 
# lrwxr-xr-x  1 root  wheel  79  8 16  2017 /usr/libexec/java_home -> /System/Library/Frameworks/JavaVM.framework/Versions/Current/Commands/java_home

$ /usr/libexec/java_home 
# /Library/Java/JavaVirtualMachines/jdk1.8.0_161.jdk/Contents/Home
```

因此`$JAVA_HOME`可以设置为:
```bash
export JAVA_HOME=$(/usr/libexec/java_home)
```

### PATH and CLASSPATH
[Ref](https://docs.oracle.com/javase/tutorial/essential/environment/paths.html)

![structure after installing JDK](https://docs.oracle.com/javase/tutorial/figures/essential/environment-directories.gif)

The bin directory in the JDF directory contains both the compiler and the launcher.

#### Checking the CLASSPATH variable (All platforms)
The CLASSPATH variable is one way to tell applications, including the JDK tools, where to look for user classes. (Classes that are part of the JRE, JDK platform, and extensions should be defined through other means, such as the bootstrap class path or the extensions directory.)

The preferred way to specify the class path is by using the -cp command line switch. This allows the CLASSPATH to be set individually for each application without affecting other applications. Setting the CLASSPATH can be tricky and should be performed with care.

The default value of the class path is ".", meaning that only the current directory is searched. Specifying either the CLASSPATH variable or the -cp command line switch overrides this value.



# Concurrent Programming in Java
## Concurrency in the Java Tutorials
[Ref](https://docs.oracle.com/javase/tutorial/essential/concurrency/index.html)

### Processes and Treads
Most implementations of the Java virtual machine run as a single process. A Java application can create additional processes using a `ProcessBuilder` object. Multiprocess applications are beyond the scope of this lesson.

Both processes and threads provide an execution environment, but creating a new thread requires fewer resources than creating a new process. 
Threads share the process's resources, including memory and open files. This makes for efficient, but potentially problematic, communication.

Multithreaded execution is an essential feature of the Java platform. Every application has at least one thread — or several, if you count "system" threads that do things like memory management and signal handling. But from the application programmer's point of view, you start with just one thread, called the main thread.

### Thread Objects
#### Defining and Starting a Thread
An instance of Thread must be Runnable,
```java
public class Task implements Runnable {
	public void run() {
		//tasks to do
	}
	public static void main(String[] args){
		Tread t1 = new Tread(new Task());
		t1.start();
	}
}
```
or inherits from `Thread`, since the `Thread` class itself implements Runnable.
```java
public class Task extends Thread {
	public void run() {
		//tasks to do
	}
	public static void main(String[] args){
		Thread t1 = new Task();
		t1.start();
	}
}
```
Which of these idioms should you use? 
The first idiom, which employs a Runnable object, **is more general**, because the Runnable object can subclass a class other than Thread. The second idiom is easier to use in simple applications, but is limited by the fact that your task class must be a descendant of Thread. 

#### Pausing Execution with Sleep
`Thread.sleep(4000)` causes the current thread to suspend execution for a specified period. 
1. efficient means of making processor time available to the other threads
2. also used for pacing

sleep times are not guaranteed to be precise, because: 
1. they are limited by the facilities provided by the underlying OS. 
2. Also, the sleep period can be terminated by interrupts


### Synchronization



## Week 1
### Threads
Things need to do:
1. create a thread
2. start the thread
3. join a thread

When operating threads, be careful about deadlocks.

Java included the notions of threads (as instances of the `java.lang.Thread` class) in its language definition right from the start. 

```java
t1 = new Tread(); // Thread t1 not started yet
t1.start();  // starts the thread
t1.join(); // wait for t1 to finish
```

### Locks
Data race: 2 threads wants to write the same object.  
Solution: `synchronized` statements.
-  require and release operations are implicit!
These operations are automatically performed by the Java runtime environment when entering and exiting the scope of a `synchronized` statements.

Limited Resource: e.g. in a buffer in and out example, only k entries at maximum allowed.
Solution: Wait and Notify




