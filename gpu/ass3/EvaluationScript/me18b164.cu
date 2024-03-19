/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 * Submitted by: Ramita Jawahar ME18B164
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
  
__global__ void update_levelzero(int *d_activeVertex, int *d_level_offsets)
  { 
      int val = d_level_offsets[1] - d_level_offsets[0];
      d_activeVertex[0] = val;
  }


__global__ void compute_AID_kernel(int* edges, bool*list_isactive,int edges_starting_index,int* AID,int edges_at_level, bool *d_edge_source_is_active)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<edges_at_level)
    {
        tid += edges_starting_index;

        if(d_edge_source_is_active[tid])
        {
                atomicAdd(&AID[edges[tid]], 1);
        }
    }
}

__global__ void check_stability_kernel(int* APR, int* AID, bool* list_isactive,int vertex_starting_index,int vertices_at_level, bool* d_edge_source_is_active, int* offset)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<vertices_at_level) 
    {
        tid += vertex_starting_index;
        if(AID[tid] < APR[tid])
            {
                list_isactive[tid] = false; 
                for(int i =offset[tid]; i<offset[tid+1];++i)
                    d_edge_source_is_active[i] = false;
            }

    }
}

__global__ void update_and_store_stability_kernel(bool* list_isactive,int vertex_starting_index,int vertices_at_level,int* d_activeVertex,int level, int vertex_ending_index,bool* d_edge_source_is_active,int *offset)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if(tid<vertices_at_level)
        {
            tid += vertex_starting_index;
            if(tid<vertex_ending_index && tid > vertex_starting_index) 
            {
                if(list_isactive[tid-1]==false && list_isactive[tid+1]==false)
                    {
                        list_isactive[tid]=false;
                     
                        for(int i =offset[tid]; i<offset[tid+1];++i)
                              d_edge_source_is_active[i] = false;
                    }
                      
            }

            if(list_isactive[tid]==true)
                atomicAdd(&d_activeVertex[level],1);
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

    //set d_aid to 0 
    cudaMemset(d_aid, 0, V * sizeof(int));

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));
    // variable for result, storing number of active vertices at each level, on device

    int *d_activeVertex;
	cudaMalloc(&d_activeVertex, L*sizeof(int));
    cudaMemcpy(d_activeVertex, h_activeVertex, L*sizeof(int),cudaMemcpyHostToDevice);

double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/

    /* Host variables*/
    int h_level_offsets[L]; // To store index of Vertex starting at each Level  
    bool h_visited[V] = {false}; // Used to find level of each node
    

    /* Device variables*/
    int *d_level_offsets; // gpu copy of hh_level_offsets
    bool *d_isactive; // stores whether the node is active or not
    bool *d_edge_source_is_active; // stores whether the destination node of the edge is active or not. Useful for direct access.

    cudaMalloc(&d_isactive,V * sizeof(bool));
    cudaMalloc(&d_edge_source_is_active,E * sizeof(bool));
    cudaMalloc(&d_level_offsets, (L+1) * sizeof(int));

    cudaMemset(d_isactive, true, V * sizeof(bool));
    cudaMemset(d_edge_source_is_active,true, E * sizeof(bool));
   
    /* Deducing Level of Nodes & updating h_level_offsets*/


    // pushing 0th node for level 0 
    h_level_offsets[0] = 0;

    //finding the starting element of level 1 throught tid
    int it = 0;
    while(h_apr[it] == 0)
    {
      h_visited[it] = true;
      it++;
    }
    h_level_offsets[1] = it;

    //find all nodes in level l+1 by iterating through edgelist from
    //h_csrList[h_offset[starting]] to h_csrList[h_offset[ending]]

    for(int l = 1; l < L-1;++l)
    {
        int edge_start = h_offset[h_level_offsets[l-1]];
        int edge_end   = h_offset[h_level_offsets[l]];

        for(int edge_index = edge_start; edge_index<edge_end;++edge_index)
            h_visited[h_csrList[edge_index]] = true;
    

        it = h_level_offsets[l];
        while(h_visited[it] == true) 
            { it++;}

        h_level_offsets[l+1] = it;
    }
    h_level_offsets[L] = V; 
    
    cudaMemcpy(d_level_offsets, h_level_offsets, (L+1)*sizeof(int), cudaMemcpyHostToDevice);

    /* Starting the Game */

    //Updating Active nodes in level = 0
    update_levelzero<<<1,1>>>(d_activeVertex, d_level_offsets);
  
    // Updating Active nodes in other levels
    for(int l= 1;l<L;++l)
  {
    /* 
        Step 1: Computing AID of nodes in level = l.
        Iterate through the edgelist for nodes in level= l-1.  
    */
     
      int vertex_starting_index = h_level_offsets[l-1]; 
      int vertex_ending_index = h_level_offsets[l]-1; 
      int edges_at_level = h_offset[vertex_ending_index+1] - h_offset[vertex_starting_index];
      int edges_starting_index = h_offset[vertex_starting_index];

      //kernel launch to compute AID of vertices at level = l
      int block_size = 1024;
      int grid_size = (edges_at_level + block_size - 1) / block_size;
      compute_AID_kernel<<<grid_size, block_size>>>(d_csrList,d_isactive,edges_starting_index,d_aid,edges_at_level,d_edge_source_is_active);
      cudaDeviceSynchronize();

    
    /* Step 2: Check Stability critera for nodes in level = l.  */
     
      vertex_starting_index = h_level_offsets[l];
      vertex_ending_index = h_level_offsets[l+1]-1;
      int vertices_at_level = vertex_ending_index - vertex_starting_index + 1;

      //kernel launch to check stability of vertices at level = l
      grid_size = (vertices_at_level + block_size - 1) / block_size;
      check_stability_kernel<<<grid_size, block_size>>>(d_apr,d_aid,d_isactive,vertex_starting_index,vertices_at_level,d_edge_source_is_active,d_offset);
      cudaDeviceSynchronize();

      /* Step 3: Update instability critera and finally store total sum of active nodes in level = l in result array d_activeVertex */
      //kernel launch to update instability of vertices at level = l, and store the count of stable vertices in d_activeVertex
      update_and_store_stability_kernel<<<grid_size, block_size>>>(d_isactive,vertex_starting_index,vertices_at_level, d_activeVertex, l,vertex_ending_index,d_edge_source_is_active,d_offset);     
      cudaDeviceSynchronize();
  }

  //Copy result to host
  cudaMemcpy(h_activeVertex, d_activeVertex, L*sizeof(int), cudaMemcpyDeviceToHost);


/********************************END OF CODE AREA**********************************/
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
