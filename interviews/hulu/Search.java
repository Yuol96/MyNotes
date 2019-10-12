import java.util.*;

public class Search {
    public static int binarySearch(int[] nums, int target){
        int l=0, r=nums.length-1;
        while(l<r){
            int mid = l + ((r-l)>>1);
            if(nums[mid] == target) r = mid;
            else if (nums[mid] > target) r = mid-1;
            else l = mid+1;
        }
        if(nums[l] == target) return l;
        return -1;
    }
    
    public static void main(String[] args){
        Random rd = new Random();
        int t = 10000;
        while(t-- > 0){
            int len = rd.nextInt(1000)+1;
            int target = rd.nextInt(1000);
            int[] nums = new int[len];
            for(int i=0; i<len; i++)
                nums[i] = rd.nextInt(1000);
            Arrays.sort(nums);
            int answer = -1;
            for(int i=0; i<len; i++)
                if(nums[i] == target) {
                    answer = i;
                    break;
                }
            if(answer != binarySearch(nums, target)){
                System.out.println("wrong answer");
                return ;
            }
        }
        System.out.println("Accepted");
    }
}