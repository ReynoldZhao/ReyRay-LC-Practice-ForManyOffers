import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.lang.Integer;
import java.lang.Character;
import java.lang.String;
import java.lang.Math;
import java.util.PriorityQueue;
import java.lang.StringBuffer;
import java.lang.Iterable;

public class Solution {
    public int MoreThanHalfNum_Solution(int [] array) {
        HashMap<Integer,Integer> map = new HashMap<Integer,Integer>();
         
        for(int i=0;i<array.length;i++){
             
            if(!map.containsKey(array[i])){
               map.put(array[i],1);
            }else{
                int count = map.get(array[i]);
                map.put(array[i],++count);
            }
        }
        Iterator iter = map.entrySet().iterator();
        while (iter.hasNext()) {
            Map.Entry entry = (Map.Entry)iter.next();
            Integer key = (Integer)entry.getKey();
            Integet value = (Integer)entry.getValue();
            if (value > array.length/2) return key;
        }
        return 0;
}