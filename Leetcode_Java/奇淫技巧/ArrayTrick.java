
//从另一个数组截一段到这
Arrays.copyOfRange(in,0,rootIndex));
Arrays.copyOfRange(in,rootIndex+1,in.length));

//直接指定内容
int[] targetRange = {-1, -1}
 
public int[][] merge(int[][] intervals) {
    if (intervals.length <= 1)
        return intervals;

    // Sort by ascending starting point
    Arrays.sort(intervals, (i1, i2) -> Integer.compare(i1[0], i2[0]));

    List<int[]> result = new ArrayList<>();
    int[] newInterval = intervals[0];
    result.add(newInterval);
    for (int[] interval : intervals) {
        if (interval[0] <= newInterval[1]) // Overlapping intervals, move the end if needed
            newInterval[1] = Math.max(newInterval[1], interval[1]);
            //java 的存储机制
        else {                             // Disjoint intervals, add the new interval to the list
            newInterval = interval;
            result.add(newInterval);
        }
    }

    return result.toArray(new int[result.size()][]);
}

int[] dp = new int[n + 1];
Arrays.fill(dp, Integer.MAX_VALUE);
	