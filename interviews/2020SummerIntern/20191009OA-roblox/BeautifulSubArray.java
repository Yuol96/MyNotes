/**
 * Source: https://www.1point3acres.com/bbs/thread-557611-1-1.html
 * Reference: https://www.geeksforgeeks.org/number-subarrays-m-odd-numbers/
 */
import java.util.*;

public class BeautifulSubArray {
    // public static int beautifulSubArray(int[] nums, int m) {
    //     int n = nums.length;
    //     int i = 0, j = 0, count = 0, ret = 0;
    //     while(true) {
    //         while(j < n && count < m) {
    //             if ((nums[j]&1) == 1) {
    //                 count++;
    //             }
    //             j++;
    //         }
    //         if (count < m) break;
    //         int incre = 1;
    //         while(j < n && (nums[j]%2 == 0)) {
    //             j++;
    //             incre++;
    //         }
    //         while(count == m) {
    //             // for(int k=i; k<j; k++) System.out.printf("%d ", nums[k]);
    //             // System.out.println("");
    //             ret+=incre;
    //             if (i >= j) break;
    //             if ((nums[i]&1) == 1) {
    //                 count--;
    //             }
    //             i++;
    //         }
    //     }
    //     return ret;
    // }

    public static int beautifulSubArray(int[] nums, int m) {
        
    }

    public static void main(String[] args) {
        System.out.println(beautifulSubArray(new int[]{2, 5, 6, 9}, 2)); // 2
        System.out.println(beautifulSubArray(new int[]{2, 2, 5, 6, 9, 2, 11}, 2)); // 8
    }
}