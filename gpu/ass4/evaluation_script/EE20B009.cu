#include <iostream>
#include <stdio.h>
#include <bits/stdc++.h>
#include <cuda.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

struct request
{
	int req_id;
	int req_cen;
	int req_fac;
	int req_start;
	int req_slots;
};

//***********************************************
// Write down the kernels here
__global__ void Process_Requests(request *req, int *facility, int *capacity, int *succ_reqs, int R)
{
	__shared__ int cap[max_P][24];
	int centre = blockIdx.x;
	int fac = threadIdx.x;
	for (int i = 0; i < 24; i++)
		cap[threadIdx.x][i] = capacity[blockIdx.x * max_P + threadIdx.x];

	int first = -1, last = -1;

	// Binary search through the array of requests to get the index of the first request corresponding to 'centre' and 'fac'
	unsigned start = 0, end = R, mid;
	do
	{
		mid = start + (end - start) / 2;
		if ((req[mid].req_cen == centre) and (req[mid].req_fac == fac) and (mid == 0 or req[mid - 1].req_fac != fac or req[mid - 1].req_cen != centre))
		{
			first = mid;
			break;
		}
		else if ((req[mid].req_cen < centre) or (req[mid].req_cen == centre and req[mid].req_fac < fac))
			start = mid + 1;
		else
			end = mid;
	} while (start < end);

	// Binary search through the array of requests to get the index of the last request corresponding to 'centre' and 'fac'
	start = first;
	end = R;
	do
	{
		mid = start + (end - start) / 2;
		if ((req[mid].req_cen == centre) and (req[mid].req_fac == fac) and (mid == end or req[mid + 1].req_fac != fac or req[mid + 1].req_cen != centre))
		{
			last = mid;
			break;
		}
		else if ((req[mid].req_cen < centre) or (req[mid].req_cen == centre and req[mid].req_fac < fac) or (req[mid].req_cen == centre and req[mid].req_fac == fac and req[mid + 1].req_cen == centre and req[mid + 1].req_fac == fac))
			start = mid + 1;
		else
			end = mid;
	} while (start < end);

	bool check = true;
	if (first != -1 and last != -1)
	{
		for (int i = first; i <= last; i++)
		{
			for (int j = req[i].req_start - 1; j < req[i].req_start + req[i].req_slots - 1; j++)
				check = check and (cap[threadIdx.x][j] > 0);

			if (check)
			{
				for (int j = req[i].req_start - 1; j < req[i].req_start + req[i].req_slots - 1; j++)
					cap[threadIdx.x][j]--;

				atomicInc((unsigned *)&succ_reqs[blockIdx.x], INT_MAX);
			}
			check = true;
		}
	}
}

//***********************************************

bool compare(request a, request b)
{
	if (a.req_cen != b.req_cen)
		return (a.req_cen < b.req_cen);

	if (a.req_fac != b.req_fac)
		return (a.req_fac < b.req_fac);

	return (a.req_id < b.req_id);
}

int main(int argc, char **argv)
{
	// variable declarations...
	int N, *centre, *facility, *capacity, *fac_ids, *succ_reqs, *tot_reqs;

	FILE *inputfilepointer;

	// File Opening for read
	char *inputfilename = argv[1];
	inputfilepointer = fopen(inputfilename, "r");

	if (inputfilepointer == NULL)
	{
		printf("input.txt file failed to open.");
		return 0;
	}

	fscanf(inputfilepointer, "%d", &N); // N is number of centres

	// Allocate memory on cpu
	centre = (int *)malloc(N * sizeof(int));		   // Computer  centre numbers
	facility = (int *)malloc(N * sizeof(int));		   // Number of facilities in each computer centre
	fac_ids = (int *)malloc(max_P * N * sizeof(int));  // Facility room numbers of each computer centre
	capacity = (int *)malloc(max_P * N * sizeof(int)); // stores capacities of each facility for every computer centre

	int success = 0;							// total successful requests
	int fail = 0;								// total failed requests
	tot_reqs = (int *)malloc(N * sizeof(int));	// total requests for each centre
	succ_reqs = (int *)malloc(N * sizeof(int)); // total successful requests for each centre

	// Input the computer centres data
	int k1 = 0, k2 = 0;
	for (int i = 0; i < N; i++)
	{
		fscanf(inputfilepointer, "%d", &centre[i]);
		fscanf(inputfilepointer, "%d", &facility[i]);

		for (int j = 0; j < facility[i]; j++)
		{
			fscanf(inputfilepointer, "%d", &fac_ids[i * max_P + j]);
			k1++;
		}
		for (int j = 0; j < facility[i]; j++)
		{
			fscanf(inputfilepointer, "%d", &capacity[i * max_P + j]);
			k2++;
		}
	}
	// Copying to GPU
	int *d_facility, *d_capacity;
	cudaMalloc(&d_facility, N * sizeof(int));
	cudaMalloc(&d_capacity, max_P * N * sizeof(int));

	cudaMemcpy(d_facility, facility, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_capacity, capacity, max_P * N * sizeof(int), cudaMemcpyHostToDevice);

	// variable declarations
	int *req_id, *req_cen, *req_fac, *req_start, *req_slots; // Number of slots requested for every request

	// Allocate memory on CPU
	int R;
	fscanf(inputfilepointer, "%d", &R);			// Total requests
	req_id = (int *)malloc(R * sizeof(int));	// Request ids
	req_cen = (int *)malloc(R * sizeof(int));	// Requested computer centre
	req_fac = (int *)malloc(R * sizeof(int));	// Requested facility
	req_start = (int *)malloc(R * sizeof(int)); // Start slot of every request
	req_slots = (int *)malloc(R * sizeof(int)); // Number of slots requested for every request

	// Input the user request data
	for (int j = 0; j < R; j++)
	{
		fscanf(inputfilepointer, "%d", &req_id[j]);
		fscanf(inputfilepointer, "%d", &req_cen[j]);
		fscanf(inputfilepointer, "%d", &req_fac[j]);
		fscanf(inputfilepointer, "%d", &req_start[j]);
		fscanf(inputfilepointer, "%d", &req_slots[j]);
		tot_reqs[req_cen[j]] += 1;
	}

	// Storage variables
	int *d_tot_reqs;
	cudaMalloc(&d_tot_reqs, N * sizeof(int));
	cudaMemcpy(d_tot_reqs, tot_reqs, N * sizeof(int), cudaMemcpyHostToDevice);

	int *d_succ_reqs;
	cudaMalloc(&d_succ_reqs, N * sizeof(int));
	cudaMemset(d_succ_reqs, 0, N * sizeof(int));

	// Array of Structures
	request *req_list;
	req_list = (request *)malloc(R * sizeof(request));

	// Populating the request list
	for (int i = 0; i < R; i++)
	{
		req_list[i].req_id = req_id[i];
		req_list[i].req_cen = req_cen[i];
		req_list[i].req_fac = req_fac[i];
		req_list[i].req_start = req_start[i];
		req_list[i].req_slots = req_slots[i];
	}

	// Sorting the request list based on requested centre
	sort(req_list, req_list + R, compare);

	request *d_req_list;
	cudaMalloc(&d_req_list, R * sizeof(request));
	cudaMemcpy(d_req_list, req_list, R * sizeof(request), cudaMemcpyHostToDevice);

	//*********************************
	// Call the kernels here
	Process_Requests<<<N, max_P>>>(d_req_list, d_facility, d_capacity, d_succ_reqs, R);
	cudaDeviceSynchronize();
	//*********************************

	cudaMemcpy(succ_reqs, d_succ_reqs, N * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++)
		success += succ_reqs[i];

	fail = R - success;

	// Output
	char *outputfilename = argv[2];
	FILE *outputfilepointer;
	outputfilepointer = fopen(outputfilename, "w");

	fprintf(outputfilepointer, "%d %d\n", success, fail);
	for (int j = 0; j < N; j++)
		fprintf(outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j] - succ_reqs[j]);

	fclose(inputfilepointer);
	fclose(outputfilepointer);
	return 0;
}