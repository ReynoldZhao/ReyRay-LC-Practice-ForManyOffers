
//T1432 map default用法 第一个和最后一个使用方法，坐标+数值
public int longestSubarray(int[] A, int limit) {
    int i = 0, j;
    TreeMap<Integer, Integer> m = new TreeMap<>();
    for (j = 0; j < A.length; j++) {
        m.put(A[j], 1 + m.getOrDefault(A[j], 0));
        if (m.lastEntry().getKey() - m.firstEntry().getKey() > limit) {
            m.put(A[i], m.get(A[i]) - 1);
            if (m.get(A[i]) == 0)
                m.remove(A[i]);
            i++;
        }
    }
    return j - i;
}

//java 的map没有初始化 要用getOrDefault方法初始化
private Map<Integer, List<Integer>> createGraph(int[][] edges) {
    Map<Integer, List<Integer>> graph = new HashMap<>();
  
    for(int i = 0; i < edges.length; i++) {
        List<Integer> list = graph.getOrDefault(edges[i][0], new ArrayList<>()); // Adjecency list representation.
        list.add(edges[i][1]);
        graph.put(edges[i][0], list);
        
        list = graph.getOrDefault(edges[i][1], new ArrayList<>()); // Adjecency list representation.
        list.add(edges[i][0]);
        graph.put(edges[i][1], list);
    }
  
    return graph;
}
//看这巧妙的getordefault
private int dfs(Map<Integer, List<Integer>> graph, int node, List<Boolean> hasApple, int myCost, Map<Integer, Boolean> visited) {
    Boolean v = visited.getOrDefault(node, false);
    if (v) {
        return 0;
    }
    visited.put(node, true);  
    int childrenCost = 0;
    for(int n : graph.getOrDefault(node, new ArrayList<>())) {
        childrenCost += dfs(graph, n, hasApple, 2, visited);
    }  
    if (childrenCost == 0 && hasApple.get(node) == false) {
      return 0;
    }
    return childrenCost + myCost;
}