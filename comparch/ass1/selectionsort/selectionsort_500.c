#include<stdio.h>
#include<stdlib.h>

#define ARRAY_SIZE_1 250000
#define ARRAY_SIZE_2 500000
#define ARRAY_SIZE_3 750000
#define ARRAY_SIZE_4 1000000
#define ARRAY_SIZE ARRAY_SIZE_2

#define SRAND_SEED 3

void selectionsort(int array[], int array_size)
{
    int iter1, iter2;
    int min_value;
    int min_index;
    for(iter1 = 0; iter1<array_size; iter1++)
    {
        min_index = iter1;
        min_value = array[iter1];
        for(iter2 = iter1+1; iter2<array_size; iter2++)
        {
            if(min_value > array[iter2])
            {
                min_value = array[iter2];
                min_index = iter2;
            }
        }
        array[min_index]  = array[iter1];
        array[iter1] = min_value;
    }
}

int main()
{
    int array[ARRAY_SIZE];
    int iteration;

    srand(SRAND_SEED);
    for (iteration=0; iteration<ARRAY_SIZE; iteration++)
        array[iteration] = rand();

    selectionsort(array, ARRAY_SIZE);
    return 0; 
}
