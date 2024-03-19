#include<stdlib.h>
#include<stdio.h>

#define ARRAY_SIZE_1 250000
#define ARRAY_SIZE_2 500000
#define ARRAY_SIZE_3 750000
#define ARRAY_SIZE_4 1000000
#define ARRAY_SIZE ARRAY_SIZE_4

#define SRAND_SEED 3

void merge(long int arr[], int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    long int L[n1], R[n2];

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];


    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}


void mergeSort(long int arr[], int l, int r)
{
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

int main()
{
    long int array[ARRAY_SIZE];
    int iteration;

    srand(SRAND_SEED);
    for (iteration=0; iteration<ARRAY_SIZE; iteration++)
        array[iteration] = rand();

    mergeSort(array, 0, ARRAY_SIZE-1);
    return 0; 
}
