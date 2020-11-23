import java.util.HashSet;; 
import java.lang.String;   
    public boolean isValidSudoku(char[][] board) {
        boolean[][] rowflag = new boolean[9][9];
        boolean[][] colflag = new boolean[9][9];
        boolean[][] blockflag = new boolean[9][9];
        for(int i=0;i<9;i++){
            for(int j=0;j<9;j++){
                if(board[i][j]==".") continue;
                int c = board[i][j] - "1";
                if(rowflag[i][c]==true||colflag[c][j]==true||blockflag[3*(i/3)+j/3][c] = true) return false;
                rowflag[i][c] = true;
                colflag[c][j] = true;
                blockflag[3*(i/3)+j/3][c] = true;
            }
        }
        return true;
    }

    public boolean isValidSudoku2(char[][] board) {
        HashSet<String> map = new HashSet<String>();
        for(int i=0;i<9;i++){
            for(int j=0;j<9;j++){
                if(board[i][j]) == '.') continue;
                String t = "(" + String.valueOf(board[i][j])+")";
                String row = String.valueOf(i) + t;
                String col = t + String.valueOf(j);
                String block = String.valueOf(i/3)+t+String.valueOf(j/3);
                if(map.contains(row)||map.contains(col)||map.contains(block)) return false;
                map.add(row);
                map.add(col);
                map.add(block);
            }
        }
        return true;
    }