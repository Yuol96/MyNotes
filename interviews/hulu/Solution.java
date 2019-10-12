import java.util.*;

public class Solution {
    // 题目是给一个只有小写字母的String s，求一个最小的子串，使得子串中包含所有s中出现过的字符
    // 思路：滑动窗口
    public static String solve(String s){
        HashSet<Character> set = new HashSet<>();
        for(char c : s.toCharArray()) set.add(c);
        HashMap<Character, Integer> map = new HashMap<>();
        for(int i=0; i<set.size()-1; i++){
            map.put(s.charAt(i), map.getOrDefault(s.charAt(i), 0)+1);
        }
        int i = 0, j = set.size()-2;
        int result = s.length(), idx = 0;
        while(true){
            // System.out.printf("%s i=%d j=%d\n", map.toString(), i, j);
            while(j+1<s.length() && !satisfy(map, set)) {
                j++;
                map.put(s.charAt(j), map.getOrDefault(s.charAt(j), 0)+1);    
            }
            if(!satisfy(map,set)) break;
            while(map.getOrDefault(s.charAt(i), 0) > 1){
                map.put(s.charAt(i), map.get(s.charAt(i))-1);
                i++;
            }
            if(result >= j-i+1) {
                result = j-i+1;
                idx = i;
                map.put(s.charAt(i), map.get(s.charAt(i))-1);
                i++;
            }
            // j++;
            // map.put(s.charAt(j), map.getOrDefault(s.charAt(j), 0)+1);   
        }
        return s.substring(idx, idx+result);
    }
    private static boolean satisfy(HashMap<Character, Integer> map, HashSet<Character> set){
        for(char c : set){
            if(map.getOrDefault(c, 0) == 0) return false;
        }
        return true;
    }
    
    public static void main(String[] args){
        System.out.println(solve("ababcad")); // bcad
        System.out.println(solve("aaaaaaa")); // a
        System.out.println(solve("abcdefg")); // abcdefg
        System.out.println(solve("abcddddefgfedcabbbbad")); // abcdefg
    }
}