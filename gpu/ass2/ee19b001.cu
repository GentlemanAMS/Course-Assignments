#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

#define TILING_SIZE 32


__global__ void comput_prod_AB_CD(int* d_matrixA, int* d_matrixB, int* d_matrixC, int* d_matrixD, int p, int q, int r, int* d_matrixE)
{
    __shared__ int Atile[TILING_SIZE][TILING_SIZE];
    __shared__ int Btile[TILING_SIZE][TILING_SIZE];
    __shared__ int Ctile[TILING_SIZE][TILING_SIZE];
    __shared__ int Dtile[TILING_SIZE][TILING_SIZE + 1]; //Posibility of Bank-conflicts is removed

    int Xvalue = 0;
    int Yvalue = 0;

    // Corresponds to which row of A. must be less than p
    int A_y = threadIdx.y + blockIdx.y * blockDim.y;

    // Corresponds to which column of B. must be less than r
    int B_x = threadIdx.x + blockIdx.x * blockDim.x;

    // Corresponds to which row of C. must be less than p
    int C_y = threadIdx.y + blockIdx.y * blockDim.y;

    // Corresponds to which column of D.T. must be less than r 
    // Note: 'threadIdX.y', This is because of D transpose
    int D_x = threadIdx.y + blockIdx.x * blockDim.x;



    for ( int tile_ii = 0; 
          tile_ii < ceil(float(q)/TILING_SIZE);
          tile_ii++ )
    {

        /*
         * Atile[threadIdX.x][threadIdX.y] = A[A_x][A_y] where
         * A_x = tile_ii * TILING_SIZE + threadIdX.x
         * A_y = threadIdX.y + blockIdX.y * blockDim.y
         * Problem is it fails coalescing. This is why we exchanged
         * X and Y axes
         * Atile[threadIdX.y][threadIdX.x] = A[A_y][A_x]
        */
        // Corresponds to which column of A. must be less than q
        int A_x = tile_ii * TILING_SIZE + threadIdx.x;

        if (A_x < q && A_y < p)
            Atile[threadIdx.y][threadIdx.x] = d_matrixA[A_y * q + A_x];
        else
            Atile[threadIdx.y][threadIdx.x] = 0;


        /*
         * Ctile[threadIdX.x][threadIdX.y] = C[C_x][C_y] where
         * C_x = tile_ii * TILING_SIZE + threadIdX.x
         * C_y = threadIdX.y + blockIdX.y * blockDim.y
         * Problem is it fails coalescing. This is why we exchanged
         * X and Y axes
         * Ctile[threadIdX.y][threadIdX.x] = C[C_y][C_x]
        */
        // Corresponds to which column of C. must be less than q

        int C_x = tile_ii * TILING_SIZE + threadIdx.x;

        if (C_x < q && C_y < p)
            Ctile[threadIdx.y][threadIdx.x] = d_matrixC[C_y * q + C_x];
        else
            Ctile[threadIdx.y][threadIdx.x] = 0;

        /*
         * Btile[threadIdX.x][threadIdX.y] = B[B_x][B_y] where
         * B_y = tile_ii * TILING_SIZE + threadIdX.x
         * B_x = threadIdX.y + blockIdX.y * blockDim.y
         * Btile[threadIdX.x][threadIdX.y] = B[B_x][B_y]
         * Note: Both B[B_x][B_y]and Btile accesses are not coalesced 
         * Btile[threadIdx.y][threadIdx.x] = B[B_y][B_Bx]
        */
        // Corresponds to which row of B. must be less than q
        int B_y = tile_ii * TILING_SIZE + threadIdx.y;
        
        if (B_y < q && B_x < r)
            Btile[threadIdx.y][threadIdx.x] = d_matrixB[B_y * r + B_x];
        else
            Btile[threadIdx.y][threadIdx.x] = 0;

        int D_y = C_x;
        
        if (D_y < q && D_x < r)
            Dtile[threadIdx.y][threadIdx.x] = d_matrixD[D_x * q + D_y];
        else
            Dtile[threadIdx.y][threadIdx.x] = 0;

        //Sync threads before multiplication
        __syncthreads();


        /*
         * Calculating value
         */
        for(int k = 0; k < TILING_SIZE; k++)
        {
            Xvalue += Atile[threadIdx.y][k] * Btile[k][threadIdx.x]; 
            Yvalue += Ctile[threadIdx.y][k] * Dtile[threadIdx.x][k];
        }
        __syncthreads();
    }

    if(A_y < p && B_x < r)
        d_matrixE[A_y * r + B_x] = Xvalue + Yvalue;

}


// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */


	int *d_matrixX;
	int *d_matrixY;
	cudaMalloc(&d_matrixY, p * r * sizeof(int));
	cudaMalloc(&d_matrixX, p * r * sizeof(int));

	#define TILING_SIZE 32
	dim3 block_dimensions(TILING_SIZE, TILING_SIZE, 1);
	dim3 grid_dimensions((ceil((float)r/TILING_SIZE)),(ceil((float)p/TILING_SIZE)),1);

	comput_prod_AB_CD<<<grid_dimensions, block_dimensions>>>(d_matrixA, d_matrixB, d_matrixC, d_matrixD, p, q, r, d_matrixE);

	cudaFree(d_matrixX);
	cudaFree(d_matrixY);
	
	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	
