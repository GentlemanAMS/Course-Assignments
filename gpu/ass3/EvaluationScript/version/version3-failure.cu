/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/

//Find Layer 0 boundary
__global__ void layer_0_layering(int* d_apr, int* d_layer, int V)
{
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < V-1)
    {
        if ((d_apr[threadid] == 0) && d_apr[threadid + 1] != 0)
            d_layer[0] = threadid;
    }
    else if (threadid == V-1)
    {
        if(d_apr[threadid] == 0)
            d_layer[0] = threadid;
    }
}
//Layer 0 activate
//Calculate Layer 1 Energy
__global__ void layer_0_activate_layer_1_energy(int* d_aid, int* d_csrList, int* d_offset, int* d_layer, int* d_energy, int* d_max_vertex)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (d_layer[0]) + 1)
    {
        d_aid[tid] = 1;
        int vertex = tid;
        d_max_vertex[vertex] = d_layer[0];
        for (int ii = (d_offset[vertex]); ii <= (d_offset[vertex + 1]-1); ii++)
        {
            atomicAdd(&d_energy[d_csrList[ii]], 1);
            if (d_max_vertex[vertex] < d_csrList[ii])
                d_max_vertex[vertex] = d_csrList[ii];
        }
    }
}
//Calculate Layer 1 Boundary - Called by single thread.
__global__ void layer_1_boundary(int* d_layer, int* d_max_vertex)
{
    d_layer[1] = d_layer[0];
    for (int ii=0; ii <= d_layer[0]; ii++)
        if(d_layer[1] < d_max_vertex[ii])
            d_layer[1] = d_max_vertex[ii];
}
















//Activate threads in the layer 'layer'

//d_layer will store the layer limits
//energy of vertices in layer 'layer+1' depend on vertices in layer 'layer': dlayer[layer-1]+1 to dlayer[layer]
//threadid for each vertex in layer 'layer': Number of threads: d_layer[layer] - d_layer[layer-1]+1
//reception vertex list for each vertex is given by d_csrlist[d_offset[tid] .... d_offset[tid + 1]-1]
//Store the energy and at the same time find the maximum vertex number connected to each vertex.

__global__ void energy_activate(int* d_apr, int* d_csrList, int* d_offset, int* d_layer, int* d_aid, int* d_energy, int* d_max_vertex, int layer)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vertex = tid + (d_layer[layer-1]+1);

    if (tid < (d_layer[layer]) - (d_layer[layer-1]+1) + 1)
    {
        if((d_energy[vertex] >= d_apr[vertex]))
            d_aid[vertex] = 1;
        else
            d_aid[vertex] = 0;

        d_max_vertex[vertex] = d_layer[layer];
        for (int ii = (d_offset[vertex]); ii <= (d_offset[vertex + 1]-1); ii++)
        {
            atomicAdd(&d_energy[d_csrList[ii]], d_aid[vertex]);
            if (d_max_vertex[vertex] < d_csrList[ii])
                d_max_vertex[vertex] = d_csrList[ii];
        }
    }
}

__global__ void vertex_deactivate(int* d_csrList, int* d_offset, int* d_layer, int* d_aid, int* d_energy, int layer)
{
    int left_margin = (d_layer[layer-1]+1) + 1;
    int right_margin = (d_layer[layer]) - 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < right_margin - left_margin + 1)
    {
        int vertex = tid + left_margin;
        if((d_aid[vertex] != 0) && (d_aid[vertex - 1] == 0) && (d_aid[vertex + 1] == 0))
        {
            d_aid[vertex] = 0;
            for (int ii=(d_offset[vertex]); ii<=(d_offset[vertex+1]-1); ii++)
                atomicSub(&d_energy[d_csrList[ii]], 1);
        }
    }
}

//Only one thread to be launched
//Find the boundary layer of layer l+1
__global__ void next_layer_boundary(int* d_layer, int* d_max_vertex, int layer)
{
    d_layer[layer+1] = d_layer[layer];
    for (int ii=(d_layer[layer-1]+1); ii <= (d_layer[layer]); ii++)
        if(d_layer[layer+1] < d_max_vertex[ii])
            d_layer[layer+1] = d_max_vertex[ii];
}






__global__ void vertex_activate(int* d_apr, int* d_layer, int* d_aid, int* d_energy, int layer)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (d_layer[layer]) - (d_layer[layer-1]+1) + 1)
    {
        int vertex = tid + (d_layer[layer-1]+1);
        if((d_energy[vertex] >= d_apr[vertex]))
            d_aid[vertex] = 1;
        else
            d_aid[vertex] = 0;
    }    
}

__global__ void last_layer_deactivate(int* d_layer, int* d_aid, int L)
{
    int left_margin = (d_layer[L-2]+1) + 1;
    int right_margin = (d_layer[L-1]) - 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < right_margin - left_margin + 1)
    {
        int vertex = tid + left_margin;
        if((d_aid[vertex] != 0) && (d_aid[vertex - 1] == 0) && (d_aid[vertex + 1] == 0))
            d_aid[vertex] = 0;
    }
}


//'L' threads to be launched
__global__ void find_number_active(int* d_aid, int* d_activeVertex, int* d_layer, int L)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < L && tid!=0)
    {
        for (int ii=(d_layer[tid-1]+1); ii <=d_layer[tid]; ii++)
            d_activeVertex[tid] = d_activeVertex[tid] + d_aid[ii];
    }
    d_activeVertex[0] = d_layer[0] + 1;
}



    
/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
	cudaMalloc(&d_activeVertex, L*sizeof(int));


/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/

cudaMemset(d_aid, 0, V*sizeof(int));

int *d_layer;
cudaMalloc(&d_layer, (L)*sizeof(int));

int *d_energy;
cudaMalloc(&d_energy, V*sizeof(int));

int *d_max_vertex;
cudaMalloc(&d_max_vertex, V*sizeof(int));



int h_left_limit;
int h_right_limit;
int no_of_threadblocks;
#define NO_OF_THREADS 1024



h_left_limit = 0; 
h_right_limit = V-1;
no_of_threadblocks = ceil(((float)(h_right_limit - h_left_limit + 1))/NO_OF_THREADS);
layer_0_layering                 <<< no_of_threadblocks, NO_OF_THREADS>>> (d_apr, d_layer, V);
cudaDeviceSynchronize();

h_left_limit = 0; 
cudaMemcpy(&h_right_limit, &d_layer[0], sizeof(int), cudaMemcpyDeviceToHost);
no_of_threadblocks = ceil(((float)(h_right_limit - h_left_limit + 1))/NO_OF_THREADS);
layer_0_activate_layer_1_energy  <<< no_of_threadblocks, NO_OF_THREADS>>> (d_aid, d_csrList, d_offset, d_layer, d_energy, d_max_vertex);
cudaDeviceSynchronize();

layer_1_boundary <<< 1, 1 >>> (d_layer, d_max_vertex);
cudaDeviceSynchronize();



for(int ii=1; ii<(L-1); ii++)
{

    cudaMemcpy(&h_left_limit, &d_layer[ii-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_right_limit, &d_layer[ii], sizeof(int), cudaMemcpyDeviceToHost);
    h_left_limit = h_left_limit + 1;
    no_of_threadblocks = ceil(((float)(h_right_limit - h_left_limit + 1))/NO_OF_THREADS);

    energy_activate    <<< no_of_threadblocks, NO_OF_THREADS>>> (d_apr, d_csrList, d_offset, d_layer, d_aid, d_energy, d_max_vertex, ii);
    vertex_deactivate  <<< no_of_threadblocks, NO_OF_THREADS>>> (d_csrList, d_offset, d_layer, d_aid, d_energy, ii);
    next_layer_boundary<<< 1, 1 >>>                     (d_layer, d_max_vertex, ii);

    cudaDeviceSynchronize();
}

cudaMemcpy(&h_left_limit, &d_layer[L-2], sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(&h_right_limit, &d_layer[L-1], sizeof(int), cudaMemcpyDeviceToHost);
h_left_limit = h_left_limit + 1;
no_of_threadblocks = ceil(((float)(h_right_limit - h_left_limit + 1))/NO_OF_THREADS);

vertex_activate    <<< no_of_threadblocks, NO_OF_THREADS>>> (d_apr, d_layer, d_aid, d_energy, L-1);
last_layer_deactivate <<< no_of_threadblocks, NO_OF_THREADS>>> (d_layer, d_aid, L);

h_left_limit = 0; 
h_right_limit = L-1;
no_of_threadblocks = ceil(((float)(h_right_limit - h_left_limit + 1))/NO_OF_THREADS);
find_number_active <<< no_of_threadblocks, NO_OF_THREADS>>> (d_aid, d_activeVertex, d_layer, L);

cudaMemcpy(h_activeVertex, d_activeVertex, L*sizeof(int), cudaMemcpyDeviceToHost);


// int *h_temp_aid;
// h_temp_aid = (int*)malloc(V*sizeof(int));
// cudaMemcpy(h_temp_aid, d_aid, sizeof(int)*V, cudaMemcpyDeviceToHost);


// int *h_layer;
// h_layer = (int*)malloc(L*sizeof(int));
// cudaMemcpy(h_layer, d_layer, sizeof(int)*L, cudaMemcpyDeviceToHost);

// int *h_energy;
// h_energy = (int*)malloc(V*sizeof(int));
// cudaMemcpy(h_energy, d_energy, sizeof(int)*V, cudaMemcpyDeviceToHost);

// int *h_max_vertex;
// h_max_vertex = (int*)malloc(V*sizeof(int));
// cudaMemcpy(h_max_vertex, d_max_vertex, sizeof(int)*V, cudaMemcpyDeviceToHost);


/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
// printResult(h_temp_aid, V, outFIle);
// printResult(h_layer, L, outFIle);
// printResult(h_offset, V, outFIle);
// printResult(h_max_vertex, V, outFIle);


if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
