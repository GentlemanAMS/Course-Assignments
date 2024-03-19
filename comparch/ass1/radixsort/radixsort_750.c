#include<stdlib.h>
#include<stdio.h>

#define ARRAY_SIZE_1 250000
#define ARRAY_SIZE_2 500000
#define ARRAY_SIZE_3 750000
#define ARRAY_SIZE_4 1000000
#define ARRAY_SIZE ARRAY_SIZE_3

#define SRAND_SEED 3


void countsort(int array[], int placeholder)
{
    int output[ARRAY_SIZE];
    int iter;
    int count[10] = {0};

    for (iter = 0; iter < ARRAY_SIZE; iter++)
        count[(array[iter]/placeholder)%10]++;

    for (iter = 0; iter < 10; iter++)
        count[iter] += count[iter - 1];

    for (iter = ARRAY_SIZE-1; iter>=0; iter--)
    {
        output[count[(array[iter]/placeholder)%10]-1] = array[iter];
        count[(array[iter]/placeholder)%10]--;
    }

    for (iter = 0; iter < ARRAY_SIZE; iter++)
        array[iter] = output[iter];
}

void radixsort(int array[])
{
    int maximum = array[0];
    int iter;

    for (iter = 1; iter < ARRAY_SIZE; iter++)
        if (array[iter] > maximum)
            maximum = array[iter];

    int placeholder = 1;
    for (placeholder = 1; maximum / placeholder; placeholder*=10)
        countsort(array, placeholder);
}

int main()
{
    int array[ARRAY_SIZE];
    int iteration;

    srand(SRAND_SEED);
    for (iteration=0; iteration<ARRAY_SIZE; iteration++)
        array[iteration] = rand();

    radixsort(array);

    return 0; 
}
