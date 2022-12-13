import java.util.Arrays;
import java.util.HashMap;
import java.lang.Integer;
import java.lang.Character;
import java.lang.String;
import java.lang.reflect.Array;
import java.lang.Math;
import java.util.PriorityQueue;
import java.util.Stack;
import javax.swing.tree.TreeNode;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class Singleton {
    private static volatile Singleton singleton;

    private Singleton() {}

    public static Singleton getInstance(){
        if (singleton == null) {
            synchronized (Singleton.class) {
                if (singleton == null) {
                    singleton = new Singleton();
                }
            }
        }
        return singleton;
    }
}