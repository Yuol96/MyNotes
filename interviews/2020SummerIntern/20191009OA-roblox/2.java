import java.io.*;
import java.math.*;
import java.security.*;
import java.text.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;
import java.util.regex.*;
import java.util.stream.*;
import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toList;



class Result {

    /*
     * Complete the 'palindrome' function below.
     *
     * The function is expected to return an INTEGER.
     * The function accepts STRING s as parameter.
     */
    final static int P = 131;
    static long[] h, p;

    public static long get(int l, int r) {
        return h[r] - h[l-1] * p[r-l+1];
    }

    public static int palindrome(String s) {
    // Write your code here
        int n = s.length();
        h = new long[n+1];
        p = new long[n+1];
        p[0] = 1;
        for(int i=1; i<=n; i++) {
            p[i] = p[i-1] * P;
            h[i] = h[i-1] * P + s.charAt(i-1);
        }

        Set<Long> set = new HashSet<>();
        for(int i=0; i<n; i++) {
            for(int j=i, k=i; j>=0 && k<s.length() && s.charAt(j) == s.charAt(k); j--, k++) {
                set.add(get(j+1, k+1));
            }
            for(int j=i, k=i+1; j>=0 && k<s.length() && s.charAt(j) == s.charAt(k); j--, k++) {
                set.add(get(j+1, k+1));
            }
        }
        return set.size();
    }

}

public class Solution {
    public static void main(String[] args) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(System.getenv("OUTPUT_PATH")));

        String s = bufferedReader.readLine();

        int result = Result.palindrome(s);

        bufferedWriter.write(String.valueOf(result));
        bufferedWriter.newLine();

        bufferedReader.close();
        bufferedWriter.close();
    }
}
