## 算法基础课

### 790. 数的三次方根
- [Link](https://www.acwing.com/problem/content/description/792/)
- Tags: 模板题
- Stars: 3

#### 2019.9.23
- attention: Remember to deal with the negative input.
- notes: 一般eps要比要求的的数量级小2，以避免浮点数计算的不精确性
```java
class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        double n = sc.nextDouble();
        int sign = 1;
        if (n < 0) {
            sign = -1;
            n = -n;
        }
        double l = 0, r = n, eps = 1e-8;
        while(r-l > eps) {
            double mid = (r+l) / 2;
            if (mid*mid*mid >= n) r = mid;
            else l = mid;
        }
        System.out.printf("%.6f", sign*l);
    }
}
```

### 791. 高精度加法
- [Link](https://www.acwing.com/problem/content/submission/793/)
- Tags: 模板题
- Stars: 3

#### 2019.9.23
```java
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String a = sc.nextLine(), b = sc.nextLine();
        List<Integer> A = new ArrayList<>(), B = new ArrayList<>();
        for(int i=a.length() - 1; i>=0; i--) A.add(a.charAt(i) - '0');
        for(int i=b.length() - 1; i>=0; i--) B.add(b.charAt(i) - '0');
        
        List<Integer> C = add(A, B);
        
        StringBuilder sb = new StringBuilder();
        for(int i=C.size()-1; i>=0; i-- ) sb.append(C.get(i));
        System.out.println(sb.toString());
    }
    
    public static List<Integer> add(List<Integer> A, List<Integer> B) {
        List<Integer> C = new ArrayList<>();
        int t = 0;
        for(int i=0; i<A.size() || i<B.size(); i++) {
            if (i<A.size()) t += A.get(i);
            if (i<B.size()) t += B.get(i);
            C.add(t%10);
            t /= 10;
        }
        if (t>0) C.add(t);
        return C;
    }
}
```

### 792. 高精度减法
- [Link](https://www.acwing.com/problem/content/794/)
- Tags: 模板题
- Stars: 3

#### 2019.9.23
- attention: cmp的实现中，需要倒着判断
- attention: 需要去除leading zeros
- attention: `while(C.size() > 1 && C.get(C.size() - 1) == 0)`中必须有`C.size() > 1`的条件，保证C==0时有正确输出
- notes: 必须使A>B，所以要加上cmp的判断
- notes: `C.add((t+10)%10)`的写法相当于把`t>=0`和 `t<0`两种情况合二为一了
```java
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String a = sc.nextLine(), b = sc.nextLine();
        List<Integer> A = new ArrayList<>(), B = new ArrayList<>();
        for(int i=a.length()-1; i>=0; i--) A.add(a.charAt(i) - '0');
        for(int i=b.length()-1; i>=0; i--) B.add(b.charAt(i) - '0');
        
        List<Integer> C = null;
        if (cmp(A, B) >= 0) {
            C = sub(A, B);
        } else {
            C = sub(B, A);
            System.out.print('-');
        }
        StringBuilder sb = new StringBuilder();
        for(int i=C.size()-1; i>=0; i-- ) sb.append(C.get(i));
        System.out.println(sb.toString());
    }
    
    public static int cmp(List<Integer> A, List<Integer> B) {
        if (A.size() != B.size()) return A.size() - B.size();
        for(int i=A.size()-1; i>=0; i--) if (A.get(i) != B.get(i)) return A.get(i) - B.get(i);
        return 0;
    }
    
    public static List<Integer> sub(List<Integer> A, List<Integer> B) {
        List<Integer> C = new ArrayList<>();
        for(int i=0, t=0; i<A.size(); i++) {
            t = A.get(i) - t;
            if (i<B.size()) t -= B.get(i);
            C.add((t+10)%10);
            t = t < 0 ? 1 : 0;
        }
        while(C.size() > 1 && C.get(C.size() - 1) == 0) C.remove(C.size() - 1);
        return C;
    }
}
```

### 793. 高精度乘法
- [Link](https://www.acwing.com/problem/content/795/)
- Tags: 模板题
- Stars: 3

#### 2019.9.23
- notes: `for(int i=0, t=0; i<A.size() || t > 0; i++)` 是个很不错的写法
```java
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String a = sc.nextLine();
        int b = sc.nextInt();
        List<Integer> A = new ArrayList<>();
        for(int i=a.length()-1; i>=0; i--) A.add(a.charAt(i) - '0');
        
        List<Integer> C = mul(A, b);
        
        StringBuilder sb = new StringBuilder();
        for(int i=C.size()-1; i>=0; i-- ) sb.append(C.get(i));
        System.out.println(sb.toString());
    }
    
    public static List<Integer> mul(List<Integer> A, int b) {
        List<Integer> C = new ArrayList<>();
        for(int i=0, t=0; i<A.size() || t > 0; i++) {
            if (i<A.size()) t += A.get(i) * b;
            C.add(t%10);
            t /= 10;
        }
        return C;
    }
}
```

### 794. 高精度除法
- [Link](https://www.acwing.com/problem/content/796/)
- Tags: 模板题
- Stars: 3

#### 2019.9.23
- notes: `r[0] * 10 + A.get(i)` 是核心操作，且注意需要将`C`反转
```java
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String a = sc.nextLine();
        int b = sc.nextInt();
        List<Integer> A = new ArrayList<>();
        for(int i=a.length()-1; i>=0; i--) A.add(a.charAt(i) - '0');
        
        int[] r = new int[1];
        List<Integer> C = div(A, b, r);
        
        StringBuilder sb = new StringBuilder();
        for(int i=C.size()-1; i>=0; i-- ) sb.append(C.get(i));
        System.out.println(sb.toString());
        System.out.println(r[0]);
    }
    
    public static List<Integer> div(List<Integer> A, int b, int[] r) {
        List<Integer> C = new ArrayList<>();
        r[0] = 0;
        for(int i=A.size() - 1; i>=0; i--) {
            r[0] = r[0] * 10 + A.get(i);
            C.add(r[0] / b);
            r[0] %= b;
        }
        Collections.reverse(C);
        while(C.size() > 1 && C.get(C.size()-1) == 0) C.remove(C.size() - 1);
        return C;
    }
}
```

### 796. 子矩阵的和
- [Link](https://www.acwing.com/problem/content/798/)
- Tags: 模板题
- Stars: 3

#### 2019.9.24
- notes: use 1-based index
```java
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt(), m = sc.nextInt(), q = sc.nextInt();
        int[][] mat = new int[n+1][m+1], S = new int[n+1][m+1];
        for(int i=1; i<=n; i++) for(int j=1; j<=m; j++) mat[i][j] = sc.nextInt();
        
        for(int i=1; i<=n; i++) for(int j=1; j<=m; j++)
            S[i][j] = S[i-1][j] + S[i][j-1] - S[i-1][j-1] + mat[i][j];
            
        while(q>0) {
            q--;
            int x1 = sc.nextInt(), y1 = sc.nextInt(), x2 = sc.nextInt(), y2 = sc.nextInt();
            System.out.println(S[x2][y2] - S[x1-1][y2] - S[x2][y1-1] + S[x1-1][y1-1]);
        }
    }
}
```

### 797. 差分
- [Link](https://www.acwing.com/problem/content/799/)
- Tags: 模板题
- Stars: 3

#### 2019.9.24
```java
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt(), m = sc.nextInt();
        int[] a = new int[n];
        for(int i=0; i<n; i++) a[i] = sc.nextInt();
        
        int[] b = new int[n];
        for(int i=0; i<n; i++) insert(b, i, i, a[i]);
        while(m > 0) {
            m--;
            int l = sc.nextInt(), r = sc.nextInt(), c = sc.nextInt();
            l--; r--;
            insert(b, l, r, c);
        }
        for(int i=1; i<n; i++) b[i] += b[i-1];
        
        StringBuilder sb = new StringBuilder();
        for(int num: b) sb.append(num).append(' ');
        System.out.println(sb.toString());
    }
    
    public static void insert(int[] b, int l, int r, int c) {
        b[l] += c;
        if (r+1 < b.length) b[r+1] -= c;
    }
}
```

### 798. 差分矩阵
- [Link](https://www.acwing.com/problem/content/800/)
- Tags: 模板题
- Stars: 3

#### 2019.9.24
```java
public class Main{
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt(), m = sc.nextInt(), q = sc.nextInt();
        int[][] a = new int[n+10][m+10];
        for(int i=1; i<=n; i++) for(int j=1; j<=m; j++) {
            a[i][j] = sc.nextInt();
        }
        
        int[][] b = new int[n+10][m+10];
        for(int i=1; i<=n; i++) for(int j=1; j<=m; j++) insert(b, i, j, i, j, a[i][j]);
        while(q > 0) {
            q--;
            int x1 = sc.nextInt(), y1 = sc.nextInt(), x2 = sc.nextInt(), y2 = sc.nextInt(), c = sc.nextInt();
            insert(b, x1, y1, x2, y2, c);
        }
        for(int i=1; i<=n; i++) for(int j=1; j<=m; j++) 
            b[i][j] += b[i-1][j] + b[i][j-1] - b[i-1][j-1];
        
        for(int i=1; i<=n; i++) {
            StringBuilder sb = new StringBuilder();
            for(int j=1; j<=m; j++) sb.append(b[i][j]).append(' ');
            System.out.println(sb.toString());
        }
    }
    
    public static void insert(int[][] b, int x1, int y1, int x2, int y2, int c) {
        b[x1][y1] += c;
        b[x2+1][y1] -= c;
        b[x1][y2+1] -= c;
        b[x2+1][y2+1] += c;
    }
}
```

### 788. 逆序对的数量
- [Link](https://www.acwing.com/problem/content/790/)
- Tags: 模板题
- Stars: 3

#### 2019.9.24
- attention: 数据范围n<=1e5, 因此返回值最大可能是`n*(n-1)/2` 大于`Integer.MAX_VALUE`，因此需要用`long`类型
```java
public class Main {
    public static void main (String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[] nums = new int[n];
        for(int i=0; i<n; i++) nums[i] = sc.nextInt();
        
        System.out.println(countReverse(nums, 0, n-1, Arrays.copyOf(nums, n)));
    }
    
    public static long countReverse(int[] nums, int l, int r, int[] copy) {
        if (l >= r) return 0;
        int mid = l + r >> 1;
        long ret = countReverse(nums, l, mid, copy) + countReverse(nums, mid+1, r, copy);
        
        int i=l, j=mid+1, p = l;
        while(i<=mid && j<=r) {
            if (nums[i] <= nums[j]) copy[p++] = nums[i++];
            else {
                ret += mid-i + 1;
                copy[p++] = nums[j++];
            }
        }
        while(i <= mid) copy[p++] = nums[i++];
        while(j <= r) copy[p++] = nums[j++];
        for(int k=l; k<=r; k++) nums[k] = copy[k];
        return ret;
    }
}
```

### 799. 最长连续不重复子序列
- [Link](https://www.acwing.com/problem/content/801/)
- Tags: 模板题
- Stars: 3

#### 2019.9.24
```java
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[] nums = new int[n];
        for(int i=0; i<n; i++) nums[i] = sc.nextInt();
        
        Set<Integer> set = new HashSet<>();
        int i = 0, j = 0, ret = 0;
        while(j < n) {
            if (set.add(nums[j])) {
                j++;
            } else {
                ret = Math.max(ret, j-i);
                while(nums[i] != nums[j]) set.remove(nums[i++]);
                set.remove(nums[i++]);
            }
        }
        ret = Math.max(ret, j-i);
        System.out.println(ret);
    }
}
```

### 801. 二进制中1的个数
- [Link](https://www.acwing.com/problem/content/803/)
- Tags: 模板题
- Stars: 3

#### 2019.9.24
- notes: `lowbit` function returns the lowest bit number of input `x`;
```java
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        StringBuilder sb = new StringBuilder();
        while(n > 0) {
            n--;
            int num = sc.nextInt(), count = 0;
            while(num > 0) {
                num -= lowbit(num); 
                count++;
            }
            sb.append(count).append(' ');
        }
        System.out.println(sb.toString());
    }
    
    public static int lowbit(int x) {
        return x&(-x);
    }
}
```

### 802. 区间和
- [Link](https://www.acwing.com/problem/content/description/804/)
- Tags: 模板题
- Stars: 3

#### 2019.9.25
- notes: 把query和insert时用到的所有坐标轴上的坐标放到map里，用hashMap可以去重。
```java
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt(), m = sc.nextInt();
        Map<Integer, Integer> map = new HashMap<>();
        List<Integer> list = new ArrayList<>();
        List<Pair> query = new ArrayList<>();
        for(int i=0; i<n; i++) {
            int x = sc.nextInt(), c = sc.nextInt();
            map.put(x, map.getOrDefault(x, 0) + c);
        }
        for(int i=0; i<m; i++) {
            int l = sc.nextInt(), r = sc.nextInt();
            if (!map.containsKey(l)) map.put(l, 0);
            if (!map.containsKey(r)) map.put(r, 0);
            query.add(new Pair(l, r));
        }
        for(int key: map.keySet()) list.add(key);
        
        // 排序并进行离散化，并计算前缀和
        Collections.sort(list);
        int[] nums = new int[map.size()+1];
        for(int i=1; i<nums.length; i++) {
            int key = list.get(i-1);
            nums[i] = map.get(key) + nums[i-1];
        }
        
        StringBuilder sb = new StringBuilder();
        for(Pair q: query) {
            int l = Collections.binarySearch(list, q.l), r = Collections.binarySearch(list, q.r);
            l++; r++;
            sb.append(nums[r] - nums[l-1]).append('\n');
        }
        System.out.print(sb.toString());
    }
    
    public static class Pair {
        int l, r;
        public Pair(int a, int b) {
            l = a;
            r = b;
        }
    }
}
```