/**
 * Source: https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=557191&highlight=roblox
 */
import java.util.*;

public class MostFreqSubstring {
    public static int getMaxOccurrences(String s, int minLength, int maxLength, int maxUnique) {
        int sLen = s.length();
        int[] cnt = new int[26];
        int count = 0, ret = 0;
        Map<String, Integer> map = new HashMap<>();
        for(int len=minLength; len<=maxLength; len++) {
            Arrays.fill(cnt, 0);
            count = 0;
            for(int i=0; i<sLen; i++) {
                cnt[s.charAt(i) - 'a']++;
                if (cnt[s.charAt(i) - 'a'] == 1) count++;
                if (i >= len) {
                    cnt[s.charAt(i-len) - 'a']--;
                    if (cnt[s.charAt(i-len) - 'a'] == 0) count--;
                }
                if (i >= len-1 && count < maxUnique) {
                    String temp = s.substring(i-len+1, i+1);
                    map.put(temp, map.getOrDefault(temp, 0) + 1);
                    // System.out.println(temp);
                    ret = Math.max(ret, map.get(temp));
                }
            }
        }
        return ret;
    }

    public static void main(String[] args) {
        String[] inputs = new String[]{
            "abcde,2,4,26",
            "ababab,2,3,4"
        };
        for(String inp: inputs) {
            String[] params = inp.split(",");
            System.out.println(getMaxOccurrences(
                params[0], 
                Integer.parseInt(params[1]),
                Integer.parseInt(params[2]),
                Integer.parseInt(params[3])
            ));
        }
    }
}