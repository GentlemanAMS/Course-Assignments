#include<stdlib.h>
#include<stdio.h>

#define ARRAY_SIZE_1 250000
#define ARRAY_SIZE_2 500000
#define ARRAY_SIZE_3 750000
#define ARRAY_SIZE_4 1000000
#define ARRAY_SIZE ARRAY_SIZE_1

#define SRAND_SEED 3

void bubblesort(int array[], int array_size)
{
    int iter1, iter2;
    int temp;
    for(iter1 = 0; iter1<array_size; iter1++)
    {
        for(iter2 = 0; iter2<array_size-iter1-1; iter2++)
        {
            if(array[iter2] > array[iter2+1])
            {
                temp = array[iter2];
                array[iter2] = array[iter2+1];
                array[iter2+1]  = temp;
            }
        }
    }
}

int main()
{
    int array[ARRAY_SIZE];
    int iteration;

    srand(SRAND_SEED);
    for (iteration=0; iteration<ARRAY_SIZE; iteration++)
        array[iteration] = rand();

    bubblesort(array, ARRAY_SIZE);
    return 0; 
}
