import java.util.Arrays;
import java.util.HashMap;
import java.lang.Integer;
import java.lang.Character;
import java.lang.String;
import java.lang.Math;
import java.util.PriorityQueue;
import java.util.concurrent.locks.ReentrantReadWriteLock.ReadLock;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Comparator;
import java.util.LinkedList;
import java.lang.StringBuilder;

public class solution{
    public static void swap(int a, int b){
        a = a^b;
        b = a^b;
        a = a^b;
    }
    public static void bubblesort(int[] arr){
        if(arr.length<2||arr==null){
            return ;
        }
        for(int e = arr.length-1; e>0; e--){
            for(int i = 0; i<e; i++){
                if(arr[i]>arr[i+1]){
                    swap(arr[i], arr[i+1]);
                }
            }
        }
    }
    public static void selectionsort(int[] arr){
        if(arr.length<2||arr==null)
            return ;
        for(int i = 0; i<arr.length ; i++){
            int minindex = i;
            for(int j = i+1; j<arr.length; j++){
                minindex = arr[j] < arr[minindex] ? j:minindex;
            }
            swap(arr[j], arr[minindex]);
        }
    }
    public static void insertationsort(int[] arr){
        if(arr.length<2||arr==null)
            return ;
        for(int i = 1; i<arr.length; i++){
            for(int j = i-1; j>=0 && arr[j+1]<arr[j]; j--){
                swap(arr[j+1], arr[j])
            }
        }
    }

    public static void heapsort(int[] arr){
        if(arr.length<2||arr==null)
            return ;
               
    }

    public static void quicksort(int[] arr){
        if(arr.size()<2||arr.size()==0){
            return;
        }
        quicksort(arr, 0, arr.size()-1);
    }
    public static void quicksort(int[] arr, int left, int right){
        if(left<right){
            int pivot = partition(int[] arr, int left, int right);
            quicksort(arr, left, pivot-1);
            quicksort(arr, pivot+1, right);
        }
    }

    public int partition(int[] arr, int l, int r){
        int pivot = arr[r];
        while(l<r){
            while(arr[left]<pivot) l++;
            while(arr[right]>pivot) r--;   
            if(l<r)
                swap(arr, l, r);
        }
        swap(arr, l, arr.lenght-1);
        return l;
    }

    public int threepartition(int[] arr, int l, int r){
        int left = l-1;
        //the left side of left are all below pivot
        int right = r;
        int move = l;
        int pivot = arr[r];
        while(move<right){
            if(arr[move]<pivot){
                left++;
                swap(arr, left, move);
                move++;
            }
            else if(arr[move]>pivot){
                right--;
                swap(arr, move, right);
                //l doesn't ++ because we don't know the previous number on position right
            }
            else{
                move++;
                //equal;
            }
        }
        swap(arr, right, r);
        return new int[] {less+1, more};
        //less+1和pivot相等, more
    }

    public void mergesort(int[] arr){
        if(arr.lenght==0||arr.length<2){
            return ;
        }
        mergesort(arr, 0, arr.length-1);
    }
    public void mergesort(int[] arr, int l, int r){
        if(l==r){
            return;
        }
        int mid =  l + (r-l)>>1;
        mergesort(arr, l, mid);
        mergesort(arr, mid+1, r);
        merge(arr, l, mid, r);
    }

    public void merge(int[] arr, int l, int mid, int r){
        int[] help = new int[r - l + 1];
        int p1 = l, p2 = mid + 1;
        int p = 0;
        while(p1<=mid&&p2<=r){
            help[p++] = arr[p1]<arr[p2]?arr[p1++]:arr[p2++];
        }
        while(p1<=mid){
            help[p++] = arr[p1++];
        }
        while(p2<=r){
            help[p++] = arr[p2++];
        }
        for(int i=0;i<help.length;i++){
            arr[l+i] = help[0];
        }
        return ;
    }
}



   
