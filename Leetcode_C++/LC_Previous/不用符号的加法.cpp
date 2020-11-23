class Solution {
public:
    int Add(int num1, int num2)
    {	
    	int temp;
		while(num2!=0){
			temp = num1;
			num1 = num1^num2;
			num2 = (temp&num2)<<1;
		}
		return num1;
    }
};
