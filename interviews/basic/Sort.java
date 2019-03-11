import java.util.*;

public class Sort {
    public static void QuickSort(int[] nums, int l, int r){
        if(l>=r) return ;
        int j = partition(nums, l, r);
        QuickSort(nums, l, j-1);
        QuickSort(nums, j+1, r);
    }
    public static int partition(int[] nums, int l, int r){
        int i=l, j=r+1;
        while(true){
            while(nums[++i] < nums[l] && i<r);
            while(nums[l] < nums[--j] && j>l);
            if(i>=j) break;
            swap(nums, i, j);
        }
        swap(nums, l, j);
        return j;
    }
    public static void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    
    public static void main(String[] args){
        Random rd = new Random();
        int t = 200;
        while(t-- > 0){
            int len = rd.nextInt(1000);
            int[] nums = new int[len];
            for(int i=0; i<len; i++)
                nums[i] = rd.nextInt(1000);
            int[] copynums = Arrays.copyOfRange(nums, 0, len);
            Arrays.sort(copynums);
            QuickSort(nums, 0, len-1);
            for(int i=0; i<len; i++)
                if(nums[i] != copynums[i]) {
                    System.out.println("Wrong Answer!");
                    return ;
                }
        }
        System.out.println("Accepted!");
    }
}