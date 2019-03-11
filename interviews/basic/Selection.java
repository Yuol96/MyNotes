import java.util.*;

public class Selection {
    public static int QuickSelection(int[] nums, int k){
        int l=0, r=nums.length-1;
        while(l<r){
            int j = partition(nums, l, r);
            if(j == k-1) return nums[j];
            else if(j > k-1) r = j-1;
            else l = j+1;
        }
        return nums[k-1];
    }
    public static int partition(int[] nums, int l, int r){
        int i=l, j=r+1;
        while(true){
            while(nums[++i] < nums[l] && i<r) ;
            while(nums[l] < nums[--j] && j>l) ;
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
            int len = rd.nextInt(1000)+1;
            int k = rd.nextInt(len)+1;
            int[] nums = new int[len];
            for(int i=0; i<len; i++)
                nums[i] = rd.nextInt(1000);
            int[] copynums = Arrays.copyOfRange(nums, 0, len);
            Arrays.sort(copynums);
            if(copynums[k-1] != QuickSelection(nums, k)){
                System.out.println("Wrong Answer!");
                return;
            }
        }
        System.out.println("Accepted!");
    }
}