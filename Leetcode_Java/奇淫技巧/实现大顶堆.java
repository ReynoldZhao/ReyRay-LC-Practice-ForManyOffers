
PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(k, new Comparator<Integer>() {
 
    @Override
    public int compare(Integer o1, Integer o2) {
        return o2.compareTo(o1);
    }
});