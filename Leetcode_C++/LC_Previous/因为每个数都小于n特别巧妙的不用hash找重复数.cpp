#include<iostream>
#include<vector>
#include<string> 
#include<algorithm>
#include<queue> 
#include<map>
#include<set> 
#include<utility>
#include<string.h> 
using namespace std;

int find_dup( int numbers[], int length) {
    for ( int i= 0 ; i<length; i++) {
        int index = numbers[i];
        if (index >= length) {
            index -= length;
        }   
        if (numbers[index] >= length) { 
            return index;
        }   
        numbers[index] = numbers[index] + length;
    }   
    return - 1 ; 
}
