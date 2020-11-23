Class Solution{
    public int[] searchRange(int[] nums, int target) {
        int[] res = {-1,-1};
        int pos1 = findpos(nums,target,true);
        int pos2 = findpos(nums,target,false);
        if(nums.length==0||pos1==-1||pos2==-1){
            return res;
        }
        else if(nums[pos1]==target&&nums[pos2]==target){
            res[0] = pos1;
            res[1] = pos2;
        }
        return res;
    }
    public int findpos(int[] nums, int target, boolean flag){
        int left = 0, right = nums.length - 1;
        int mid = (right-left)/2 + left;
        while(left<=right){
            mid = (right-left)/2 + left;
            if(nums[mid]<target){
                left = mid+1;
            }
            else if(nums[mid]>target){
                right = mid-1;
            }
            else return mid;
        }
        if(nums[mid]==target){
            if(flag){
                while(mid-1>=0&&nums[mid-1]==target){
                    mid = mid-1;
                }
            }
            else{
                while(mid+1<=nums.length-1&&nums[mid+1]==target){
                    mid = mid+1;
                }
            }
            return mid;
        }
        else return -1;
    }
    //能找到相同数字的最左和大于target的第一个
    public int xtremeInsertionIndex(int[] nums, int target, boolean left){
        int l = 0, r = nums.length-1;
        int mid = (r-l)/2+l;
        while(l<r){
            mid = (r-l)/2+l;
            if(nums[mid]>target || (left&&(nums[mid]==target))){
                 r = mid;
            }
            else{
                l = mid + 1;
            }
        }
        return l;
    }
    public int[] searchRange(int[] nums, int target) {
        int[] targetRange = {-1, -1};

        int leftIdx = extremeInsertionIndex(nums, target, true);

        // assert that `leftIdx` is within the array bounds and that `target`
        // is actually in `nums`.
        if (leftIdx == nums.length || nums[leftIdx] != target) {
            return targetRange;
        }

        targetRange[0] = leftIdx;
        targetRange[1] = extremeInsertionIndex(nums, target, false)-1;

        return targetRange;
    }
    public int binarysearch(int[] nums, double target){
        int lo = 0, hi = nums.length-1;
        while(lo<hi){
            int mid = (hi-lo)/2+lo;
            if(nums[mid]<target){
                lo = mid+1;
            }
            else if(nums[mid]>target){
                hi = mid-1;
            }
        }
        return lo;
    }
}