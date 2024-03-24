#include<iostream>
#include<cstdio>
using namespace std;

float a[30],b[30];
int n;
int main(){
    n=5;
    for(int i=1;i<=n;i++){
        cin>>a[i]>>b[i];
    }
    cout<<"X = [";
    for(int i=1;i<=n;i++){
        if(i!=n) cout<<a[i]<<',';
        else cout<<a[i]<<"]\n";
    }
    cout<<"X1 = [";
    for(int i=1;i<=n;i++){
        if(i!=n) cout<<'['<<a[i]<<']'<<',';
        else cout<<'['<<a[i]<<']'<<"]\n";
    }
    cout<<"Y = [";
    for(int i=1;i<=n;i++){
        if(i!=n) cout<<b[i]<<',';
        else cout<<b[i]<<"]\n";
    }
    return 0;
}