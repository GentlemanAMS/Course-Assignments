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


__global__ void helper(int levelno, int* nodeinlevel){
   nodeinlevel[levelno]+=nodeinlevel[levelno-1];
}

__global__ void activation(int *offset, int *csrList, int *apr, int *aid, int *activeVertices, int level, int *levelArray, bool *activenode, int V, int L, int *nodeInLevel, int *visited)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    //int activeNodes;
    int node_active =0;
    
    if(tid<V && level == 0  && apr[tid] ==0 )
    {
        atomicAdd(&nodeInLevel[level], 1);
        atomicAdd(&activeVertices[level], 1);
        activenode[tid] =1; 
        node_active =1;

        for(int i=offset[tid]; i<offset[tid+1]; i++)
        {
            levelArray[csrList[i]] = level+1;
            if(!atomicCAS(&visited[csrList[i]], 0, 1)) atomicAdd(&nodeInLevel[level+1], 1);
            //atomicCAS(&levelArray[csrList[i]], level+1, level+1);
            atomicAdd(&aid[csrList[i]], 1);
        }
    }

    
    else
    {
        int start = nodeInLevel[level -1];
        int   end = nodeInLevel[level];
        int levelsize = end -start;
    if(tid<levelsize && levelArray[tid+start] == level )
    {
        if(aid[start + tid] >=apr[start + tid])
        {
            atomicAdd(&activeVertices[level], 1);
            node_active =1;
            activenode[start + tid] =1; 
        }


        
        for(int i=offset[start + tid]; i<offset[start +tid+1] && level<L-1; i++)
        {
            if(!atomicCAS(&visited[csrList[i]], 0, 1)) atomicAdd(&nodeInLevel[level+1], 1);
            levelArray[csrList[i]] = level+1;
            //atomicCAS(&levelArray[csrList[i]], level+1, level+1);
            if(node_active)atomicAdd(&aid[csrList[i]], 1);
        }
    }
    }
    


}

__global__ void deactivation(int *offset, int *csrList, int *apr, int *aid, int *activeVertices, int level, int *levelArray, bool *activenode, int V, int L, int *nodeInLevel)
{
    int start = nodeInLevel[level-1];
    int end = nodeInLevel[level];
    int levelsize = end-start;   
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if(tid>0 && tid <(levelsize-1) && levelArray[tid+start] == level && levelArray[tid-1+start] == level && levelArray[tid+1+start] == level && activenode[start + tid] && (!(activenode[start +tid-1]) && !(activenode[start + tid+1])))
    {
        activenode[start + tid] = 0;
        atomicSub(&activeVertices[level], 1);
        for(int i=offset[start + tid]; i<offset[start + tid+1] && level < L-1; i++)
    {
        atomicSub(&aid[csrList[i]], 1);
    }
    }
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
    int *d_aid; // acive in-degree ar   ray
    int *d_levelArray;
    bool *d_activenodes;
    int *d_nodeInLevel;
    int *d_visited;

    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));
    cudaMalloc(&d_levelArray, V*sizeof(int));
    cudaMalloc(&d_activenodes, V*sizeof(bool));
    cudaMalloc(&d_nodeInLevel, L*sizeof(int));
    cudaMalloc(&d_visited, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    int *h_nodeInLevel;
    h_nodeInLevel = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    //memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
	cudaMalloc(&d_activeVertex, L*sizeof(int));
    cudaMemset(d_activeVertex, 0, L*sizeof(bool));

/***Important***/

cudaMemset(d_aid, 0, V*sizeof(int));// Initialize d_aid array to zero for each vertex
cudaMemset(d_levelArray, 0, V*sizeof(int));
cudaMemset(d_activenodes, 0, V*sizeof(bool));
cudaMemset(d_nodeInLevel, 0, L*sizeof(int));
cudaMemset(d_visited, 0, V*sizeof(int));
// Make sure to use comments



/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/
//int *offset, int *csrList, int *apr, int *aid, int *activeVertices, int level, int *levelArray, bool *activenode, int V
//int *offset, int *csrList, int *apr, int *aid, int *activeVertices, int level, int *levelArray, bool *activenode
for(int i=0; i<L ; i++)
{
    if(i!=0){
        helper<<<1,1>>>(i,d_nodeInLevel);
       }
    activation<<<16,1024>>>(d_offset, d_csrList, d_apr, d_aid, d_activeVertex, i, d_levelArray, d_activenodes, V,L, d_nodeInLevel, d_visited);
    if(i!=0)deactivation<<<16,1024>>>(d_offset, d_csrList, d_apr, d_aid, d_activeVertex, i, d_levelArray, d_activenodes, V,L , d_nodeInLevel);
    //cudaMemcpy(h_nodeInLevel, d_nodeInLevel, L*sizeof(int), cudaMemcpyDeviceToHost);
}
cudaDeviceSynchronize();





    
 
//    for(int i=0; i<V+1; i++)
//    {
//     printf(" %d ",i );
//    }
//    printf("\n");
//    for(int i=0; i<V+1; i++)
//    {
//     printf(" %d ",h_offset[i] );
//    }
//    printf("\n");
//    for(int i=0; i<E; i++)
//    {
//     printf(" %d ",i );
//    }
//    printf("\n");
//    for(int i=0; i<E; i++)
//    {
//     printf(" %d ",h_csrList[i] );
//    }
//    printf("\n");

    
    

     

/********************************END OF CODE AREA**********************************/
cudaMemcpy(h_activeVertex, d_activeVertex, L*sizeof(int), cudaMemcpyDeviceToHost);

cudaFree(d_offset);
cudaFree(d_csrList);
cudaFree(d_apr);
cudaFree(d_aid);
cudaFree(d_levelArray);
cudaFree(d_activenodes);
cudaFree(d_activeVertex);

double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
