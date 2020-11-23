    int countPrimes(int n) {
        int res = 0;
        vector<bool> prime(n, true);
        for (int i = 2; i < n; ++i) {
            if (!prime[i]) continue;
            ++res;
            for (int j = 2; i * j < n; ++j) {
                prime[i * j] = false;
            }
        }
        return res;
    }
bool isPrime2(int n){
    bool yes=true;
    for(int i=2;i<=sqrt(n);i++){
        if(n%i==0){
            yes=false;
            break;
        }
    }
    return yes;
}