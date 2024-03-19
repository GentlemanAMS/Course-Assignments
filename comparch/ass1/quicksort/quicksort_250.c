#include<stdlib.h>
#include<stdio.h>

#define ARRAY_SIZE_1 250000
#define ARRAY_SIZE_2 500000
#define ARRAY_SIZE_3 750000
#define ARRAY_SIZE_4 1000000
#define ARRAY_SIZE ARRAY_SIZE_1

#define SRAND_SEED 3

int partition(long int array[], int low, int high)
{
    long int pivot = array[high];
    int iter1 = low - 1;
    long int temp;
    for (int iter2 = low; iter2 <= high-1; iter2++)
    {
        if (array[iter2] < pivot)
        {
            iter1++;
            temp = array[iter1];
            array[iter1] = array[iter2];
            array[iter2] = temp;
        }
    }
    temp = array[iter1+1];
    array[iter1+1] = array[high];
    array[high] = temp;

    return iter1+1;
}

void quicksort(long int array[], int low, int high)
{
    long int pivot;
    if(low < high)
    {
        pivot = partition(array, low, high);
        quicksort(array, low, pivot-1);
        quicksort(array, pivot+1, high);
    }
}

int main()
{
    long int array[ARRAY_SIZE];
    int iteration;

    srand(SRAND_SEED);
    for (iteration=0; iteration<ARRAY_SIZE; iteration++)
        array[iteration] = rand();

    quicksort(array, 0, ARRAY_SIZE-1);
    return 0; 
}
