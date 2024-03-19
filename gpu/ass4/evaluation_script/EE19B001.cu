#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define max_N 100000
#define max_R 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;



/**
 * Structure for accessing the request
 */
struct REQUEST{
  int request_id;
  int centre_id;
  int facility_id;
  int start_slot_number;
  int number_of_slots;
};

__global__ void process_facility_request(
  int* successful_requests,
  int* capacity_of_facilities,                //Capacity of facilities in an array
  int* number_of_facilities_in_centre,        //Number of facilities in this centre
  int* number_of_requests,                    //Number of requests in this facility
  int  max_number_of_requests_all_facility,   //Maximum number of requests in all facilities
  REQUEST* requests_queue                     //Requests stored in an array
)
{
  int facility_id = threadIdx.x;
  int centre_id = blockIdx.x;

  //Check whether facility_id is actually valid or not
  if (threadIdx.x < number_of_facilities_in_centre[centre_id]){

    int index = centre_id * max_P + facility_id;
    int max_capacity_of_this_facility = capacity_of_facilities[index];
    
    int capacity[24];
    for (int ii = 0; ii < 24; ii++)
      capacity[ii] = max_capacity_of_this_facility;

    /**
     * @brief 
     * Consider REQUEST* request as an array of request[max_N*max_P][max_number_of_requests_all_facility]
     * The request for this facility will be in reqeust[index][0] to request[index][number_of_requests[index]]
     * where index = centre_id * max_P + facility_id;
     * 
     * 
     * So we must access from:
     * index*max_number_of_requests_all_facility to index*max_number_of_requests_all_facility + number_of_requests[index]
     */

    int left_margin  = index*max_number_of_requests_all_facility;
    int right_margin = index*max_number_of_requests_all_facility + number_of_requests[index];

    int successful_requests_this_facility = 0;


    for (int ii = left_margin; ii < right_margin; ii++){

      REQUEST temp_request;
      temp_request.centre_id = requests_queue[ii].centre_id;
      temp_request.facility_id = requests_queue[ii].facility_id;
      temp_request.request_id = requests_queue[ii].request_id;
      temp_request.number_of_slots = requests_queue[ii].number_of_slots;
      temp_request.start_slot_number = requests_queue[ii].start_slot_number;

      bool check_availability = true;
      for (int jj = temp_request.start_slot_number - 1; 
               jj < temp_request.start_slot_number + temp_request.number_of_slots - 1;
               jj++ ){
        
        if (capacity[jj] <= 0)
          check_availability = false;
      }

      if (check_availability == true)
      {
        for (int jj = temp_request.start_slot_number - 1; 
                 jj < temp_request.start_slot_number + temp_request.number_of_slots - 1;
                 jj++ ){        
        
          capacity[jj]--;
        }
        successful_requests_this_facility++;
      }
      check_availability = true;
    }
    atomicAdd(&successful_requests[centre_id], successful_requests_this_facility);
  }
}



__global__ void enqueuing(
  REQUEST* requests_queue,                        //Request Queue
  REQUEST* requests_list,                         //Request List
  int*     number_of_requests,                    //Number of requests for the facility
  int      max_number_of_requests_all_facility,   //Maximum number of requests of all facility
  int      R,                                      //Maximum reqeusts
  int      N
  )
{
  
  //Initializing
  for (int ii = 0; ii < N*max_P; ii++){
    number_of_requests[ii] = 0;
  }

  REQUEST temp_request; 
  int index;

  for (int ii = 0; ii < R; ii++){
    // printf("\nZZZ\n");
    temp_request.centre_id         = requests_list[ii].centre_id;
    temp_request.facility_id       = requests_list[ii].facility_id;
    temp_request.request_id        = requests_list[ii].request_id;
    temp_request.start_slot_number = requests_list[ii].start_slot_number;
    temp_request.number_of_slots   = requests_list[ii].number_of_slots;

    index = temp_request.centre_id * max_P + temp_request.facility_id;

    requests_queue[index*max_number_of_requests_all_facility + number_of_requests[index]].centre_id          = requests_list[ii].centre_id;
    requests_queue[index*max_number_of_requests_all_facility + number_of_requests[index]].facility_id        = requests_list[ii].facility_id;
    requests_queue[index*max_number_of_requests_all_facility + number_of_requests[index]].request_id         = requests_list[ii].request_id;
    requests_queue[index*max_number_of_requests_all_facility + number_of_requests[index]].start_slot_number  = requests_list[ii].start_slot_number;
    requests_queue[index*max_number_of_requests_all_facility + number_of_requests[index]].number_of_slots    = requests_list[ii].number_of_slots;

    number_of_requests[index]++;
  }
}












//***********************************************


int main(int argc,char **argv)
{
	// variable declarations...
    int N;
    int *centre;
    int *number_of_facilities_in_centre;
    int *capacity_of_facilities;
    int *fac_ids;
    int *succ_reqs; 
    int *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre                          = (int*)malloc(N * sizeof (int));           // Computer  centre numbers
    fac_ids                         = (int*)malloc(max_P * N * sizeof (int));   // Facility room numbers of each computer centre
    number_of_facilities_in_centre  = (int*)malloc(N * sizeof (int));           // Number of facilities in each computer centre
    capacity_of_facilities          = (int*)malloc(max_P * N * sizeof (int));   // stores capacities of each facility for every computer centre 


    int success=0;                             // total successful requests
    int fail = 0;                              // total failed requests
    tot_reqs  = (int *)malloc(N*sizeof(int));  // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int));  // total successful requests for each centre

    // Input the computer centres data
    for(int i=0; i < N ;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &number_of_facilities_in_centre[i] );
      
      for(int j=0; j<number_of_facilities_in_centre[i]; j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[i * max_P + j] );
      }
      for(int j=0; j<number_of_facilities_in_centre[i]; j++)
      {
        fscanf( inputfilepointer, "%d", &capacity_of_facilities[i * max_P + j]);
      }
    }

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request
    
    // Allocate memory on CPU 
	int R;
	fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }
		





    //*********************************
    // Call the kernels here
    //********************************


    REQUEST requests_list[max_R];
    int* number_of_requests;
    number_of_requests = (int*) malloc( N * max_P * sizeof(int));
    memset(number_of_requests, 0, N*max_P*sizeof(int));
    
    for (int ii = 0 ; ii < R; ii++ ){

      requests_list[ii].centre_id         = req_cen[ii];
      requests_list[ii].facility_id       = req_fac[ii];
      requests_list[ii].request_id        = req_id[ii];
      requests_list[ii].start_slot_number = req_start[ii];
      requests_list[ii].number_of_slots   = req_slots[ii];
    
      number_of_requests[requests_list[ii].centre_id * max_P + requests_list[ii].facility_id]++;
    }

    int max_number_of_requests_all_facility = 0;
    for (int ii = 0; ii < (N*max_P); ii++){
      if (number_of_requests[ii] > max_number_of_requests_all_facility)
          max_number_of_requests_all_facility = number_of_requests[ii];
    }


    REQUEST* d_requests_list;
    cudaMalloc(&d_requests_list,  R*sizeof(REQUEST));
    cudaMemcpy(d_requests_list, requests_list, R*sizeof(REQUEST), cudaMemcpyHostToDevice);


    REQUEST* d_requests_queue;
    cudaMalloc(&d_requests_queue, max_number_of_requests_all_facility*N*max_P*sizeof(REQUEST));

    int* d_number_of_requests;
    cudaMalloc(&d_number_of_requests, N*max_P*sizeof(int));
    cudaMemset(d_number_of_requests, 0, N*max_P*sizeof(int));

    cudaDeviceSynchronize();
    enqueuing<<<1,1>>>(d_requests_queue, d_requests_list, d_number_of_requests, max_number_of_requests_all_facility, R, N);
    cudaDeviceSynchronize();


    int* d_capacity_of_facilities;
    cudaMalloc(&d_capacity_of_facilities, max_P * N * sizeof (int));
    cudaMemcpy(d_capacity_of_facilities, capacity_of_facilities, max_P * N * sizeof(int), cudaMemcpyHostToDevice);

    int* d_number_of_facilities_in_centre;
    cudaMalloc(&d_number_of_facilities_in_centre, N * sizeof (int));
    cudaMemcpy(d_number_of_facilities_in_centre, number_of_facilities_in_centre, N * sizeof(int), cudaMemcpyHostToDevice);

    int* d_successful_requests;
    cudaMalloc(&d_successful_requests, N*sizeof(int));
    cudaMemset(d_successful_requests, 0, N*sizeof(int));
    
    cudaDeviceSynchronize();
    process_facility_request<<<N, max_P>>>(d_successful_requests, d_capacity_of_facilities, d_number_of_facilities_in_centre, d_number_of_requests, max_number_of_requests_all_facility, d_requests_queue);
    cudaDeviceSynchronize();


    cudaMemcpy(succ_reqs, d_successful_requests, N*sizeof(int), cudaMemcpyDeviceToHost);


    for (int i = 0; i < N; i++)
    {
      success += succ_reqs[i];
      fail += tot_reqs[i]-succ_reqs[i];
    }
    

    // Output

    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}