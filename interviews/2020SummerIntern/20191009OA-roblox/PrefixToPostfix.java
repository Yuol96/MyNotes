/**
 * Source: https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=557191&highlight=roblox
 */

import java.util.*;

public class PrefixToPostfix {
    public static List<String> prefixToPostfix(List<String> prefixes) {
        List<String> ret = new ArrayList<>(prefixes.size());
        for(String prefix: prefixes) {
            ret.add(convert(prefix));
        }
        return ret;
    }

    public static String convert(String prefix) {
        int len = prefix.length();
        char[] chrs = new char[len];
        int p = 0;
        char[] stk = new char[len+10];
        int[] cnt = new int[len+10];
        int tt = 0;

        for(char c: prefix.toCharArray()) {
            if (Character.isDigit(c) || Character.isLetter(c)) {
                // number
                chrs[p++] = c;
                cnt[tt]++;
            } else {
                // operator
                stk[++tt] = c;
                cnt[tt] = 0;
            }
            while (tt > 0 && cnt[tt] == 2) {
                chrs[p++] = stk[tt];
                cnt[--tt]++;
            }
        }

        return new String(chrs);
    }

    public static void main(String[] args) {
        List<String> list = Arrays.asList(
            "+1**23/14",
            "*34",
            "+1*23",
            "+12",
            "*+AB-CD",
            "*-A/BC-/AKL"
        );
        List<String> result = prefixToPostfix(list);
        System.out.println(result.toString());
    }
}

/**
 * Another Method: https://www.geeksforgeeks.org/prefix-postfix-conversion/
 */

 // JavaProgram to convert prefix to postfix 
// import java.util.*; 
  
// class GFG 
// { 
  
// // funtion to check if character  
// // is operator or not 
// static boolean isOperator(char x)  
// { 
//     switch (x)  
//     { 
//         case '+': 
//         case '-': 
//         case '/': 
//         case '*': 
//         return true; 
//     } 
//     return false; 
// } 
  
// // Convert prefix to Postfix expression 
// static String preToPost(String pre_exp) 
// { 
  
//     Stack<String> s= new Stack<String>(); 
  
//     // length of expression 
//     int length = pre_exp.length(); 
  
//     // reading from right to left 
//     for (int i = length - 1; i >= 0; i--)  
//     { 
  
//         // check if symbol is operator 
//         if (isOperator(pre_exp.charAt(i)))  
//         { 
  
//             // pop two operands from stack 
//             String op1 = s.peek(); s.pop(); 
//             String op2 = s.peek(); s.pop(); 
  
//             // concat the operands and operator 
//             String temp = op1 + op2 + pre_exp.charAt(i); 
  
//             // Push String temp back to stack 
//             s.push(temp); 
//         } 
  
//         // if symbol is an operand 
//         else
//         { 
//             // push the operand to the stack 
//             s.push( pre_exp.charAt(i)+""); 
//         } 
//     } 
  
//     // stack contains only the Postfix expression 
//     return s.peek(); 
// } 
  
// // Driver Code 
// public static void main(String args[])  
// { 
//     String pre_exp = "*-A/BC-/AKL"; 
//     System.out.println("Postfix : " + preToPost(pre_exp)); 
// } 
// } 