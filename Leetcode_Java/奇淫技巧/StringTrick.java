class Solution {
    public String replaceSpaces(String S, int length) {
        return S.substring(0,length).replace(" ","%20");
    }
}
//字符串替换函数

public String replaceSpace(StringBuffer str) {
    str.length();
    str.charAt(i)
    str.setLength(newLength);
}

public String ReverseSentence(String str) {
    if(str.trim().equals("")||str==null){
        return str;
    }
    String[] a = str.spilt(" ");
    StringBuffer o = new StringBuffer();
    int i;
    for (int i=a.length;i>0;i--){
        o.append(str[i]);
        if(i>0)
            o.append(" ");
    }
    return o.toString();
}

//java 的 string 想获得某个用charAt, 新增用stringbuilder
public String countAndSay(int n) {
    if(n==1) return "1";
    String s = countAndSay(n-1);
    StringBuilder res = new StringBuilder();
    for(int j = 0;j<s.length();j++){
        Character temp = s.charAt(j);
        int count = 1;
        while(j+1<s.length()&&s.charAt(j+1)==temp) {
            j++;
            count++;
        }
        res.append(count+""+temp);
    }
    return res.toString();
}

String sort = "badcfehg"; // 字母顺序
String str = "aabbccddeeffgghh"; // 待排序字符串
 
List arr = new ArrayList();
for (char ch : str.toCharArray())
    arr.add(ch);
 
// 排序
arr.sort((a, b) -> {
    int idx1 = sort.indexOf((char) a);
    int idx2 = sort.indexOf((char) b);
    return idx1 - idx2;
});
StringBuilder sb = new StringBuilder();
for(Object ch : arr){
    sb.append(ch);
}
System.out.println(sb);

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

//string 排序
public class StringSort {
 
    public static void main(String[] args) {
        String string = "to good too no string and now";
        String sortString = sortChar(string);
        System.out.println(sortString);
    }
 
    private static String sortChar(String str) {
        // 1.将字符串转化成数组
        char[] chs = stringToArray(str);
        // 2.对数组进行排序
        sort(chs);
        // 3.将数组转成字符串
        return toString(chs);
    }
 
    private static String toString(char[] chs) {
        return new String(chs);
    }
 
    // 对字符数组进行升序排序
    private static void sort(char[] chs) {
        Arrays.sort(chs);
    }
 
    // 将字符串转化成为数组
    private static char[] stringToArray(String string) {
        return string.toCharArray();
    }

    public class Solution {
        public List<List<String>> groupAnagrams(String[] strs) {
            List<List<String>> res = new ArrayList<List<String>>();
            if (strs == null || strs.length == 0) return res;
            Map<String, List<String>> map = new HashMap<String, List<String>>();
            for(String str:strs){
                char[] ch = str.toCharArray();
                Arrays.sort(ch);
                String s = String.valueOf(ch);
                if(!map.containsKey(s)){
                    map.put(s, new ArrayList<String>());
                }
                map.get(s).add(str);
            }
            return new ArrayList<List<String>>(map.values());
    }
}

//JAVA 比较两个字符串内容相等 别用== 用 equals！！！

String[] ipv4 = IP.split("\\.",-1);

if(IP.chars().filter(ch -> ch == '.').count() == 3){
    for(String s : ipv4) if(isIPv4(s)) continue;else return "Neither"; return "IPv4";
}

public static boolean isIPv4 (String s){
    try{ return String.valueOf(Integer.valueOf(s)).equals(s) && Integer.parseInt(s) >= 0 && Integer.parseInt(s) <= 255;}
    catch (NumberFormatException e){return false;}
}


