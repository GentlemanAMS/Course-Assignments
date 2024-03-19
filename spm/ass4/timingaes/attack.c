/******************************************************************************
 * File	Name			: attack.c
 * Organization                 : Indian Institute of Technology Kharagpur
 * Project Involved		: First Round Attack on AES
 * Author		    	: Arun Krishna AMS
 * Date of Creation		: 15/Dec/2012
 * Date of freezing		: 
 * Log Information regading 
 * maintanance			:
 * Synopsis			: 
 ******************************************************************************/
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <aes.h>

#include "params.h"


unsigned long long int ttime[16][16];         
unsigned int tcount[16][16];        
double deviations[16][16];          

unsigned int i_secretkey[16]; 

unsigned int guessing_entropy;
float guessing_entropy_average;

unsigned int finddeviant(unsigned int c)
{
	int i;

	/* Compute average timing for c */
	unsigned long long int ttimesum = 0;
	unsigned int tcountsum = 0;

	double tavg[16][16];       

	for(i=0; i<16; ++i){
		tavg[c][i] = ttime[c][i] / (double)tcount[c][i];	
		ttimesum += ttime[c][i];
		tcountsum += tcount[c][i];
	}	
	double tavgavg = ttimesum/(double)tcountsum;

	/* Compute deviations from the average time */
	double deviations_temp[16][16];          
	for(i=0; i<16; ++i){
		deviations[c][i] = fabs(tavg[c][i] - tavgavg);
		deviations_temp[c][i] = fabs(tavg[c][i] - tavgavg);
	}	

	/* Find the maximum deviation, this is the possible leakage */
	int maxi = 0;
	double	maxdeviation = deviations[c][0];
	for(i=1; i<16; ++i){
		if(maxdeviation < deviations[c][i]){
			maxdeviation = deviations[c][i];
			maxi = i;
		}
	}

	unsigned int deviations_index_sorted[16][16];
	for(i=0; i<16; i++)
		deviations_index_sorted[c][i] = i;
	
	for (i=0; i<15; i++){
		for (int j=0; j<15-i; j++){
			if(deviations_temp[c][j] < deviations_temp[c][j+1]){
				double temp_d = deviations_temp[c][j];
				deviations_temp[c][j] = deviations_temp[c][j+1];
				deviations_temp[c][j+1] = temp_d;
				int temp_i = deviations_index_sorted[c][j];
				deviations_index_sorted[c][j] = deviations_index_sorted[c][j+1];
				deviations_index_sorted[c][j+1] = temp_i;
			}
		}
	}
	
	for (i=0; i<16; i++){
		if(deviations_index_sorted[c][i]==i_secretkey[c]/16){
			guessing_entropy = guessing_entropy + i+1;
		}
	}
	
	return maxi;
}


static inline void lfence() {
  asm volatile("lfence");
}

static inline void mfence() {
  asm volatile("mfence");
}

static inline void clflush(void *v) {
  asm volatile ("clflush 0(%0)": : "r" (v):);
}

static inline unsigned long long rdtsc(void)
{
	unsigned hi, lo;
	__asm__ __volatile__ ("xorl %%eax,%%eax\n cpuid \n" ::: "%eax", "%ebx", "%ecx", "%edx"); // flush pipeline
	__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
	__asm__ __volatile__ ("xorl %%eax,%%eax\n cpuid \n" ::: "%eax", "%ebx", "%ecx", "%edx"); // flush pipeline
	return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


#define TIME_THRESHOLD     600    // for 4 tables of 1024 bytes

unsigned char pt[16];               /* Holds the Plaintext */
unsigned char ct[16];               /* Holds the ciphertext */
AES_KEY expanded;


void attack_round1(unsigned long iterations)
{
	int ii=0, i;
	unsigned int start, end, timing;

	guessing_entropy = 0;

	while(ii++ <= (iterations)){
		/* Set a random plaintext */
		for(i=0; i<16; ++i) pt[i] = random() & 0xff;
		/* Fix a few plaintext bits of some plaintext bytes */
		pt[0] = pt[0] & 0x0f;
		pt[1] = pt[1] & 0x0f;
		pt[2] = pt[2] & 0x0f;
		pt[3] = pt[3] & 0x0f;

		/* clean the cache memory of any AES data */
		cleancache();	

		/* Make the encryption */
		start = timestamp();
		AES_encrypt(pt, ct, &expanded);
		end = timestamp();

		timing = end - start;

		if(ii > 1000 && timing < TIME_THRESHOLD){      
			/* Record the timings */
			for(i=4; i<16; ++i){
				ttime[i][pt[i] >> 4] += timing;
				tcount[i][pt[i] >> 4] += 1;	
			}	
		}
	}

	for(int i = 0; i < 16; i ++) finddeviant(i);


	FILE *f;
	f = fopen("Guessing_Entropy", "a");
	fprintf(f, "\nKey: ");
	for(int i=0; i<16; i++)
		fprintf(f," %x ", i_secretkey[i]);
	fprintf(f, " Guessing Entropy: %d", guessing_entropy);
	fclose(f);

	printf("\nKey: ");
	for(int i=0; i<16; i++)
		printf(" %x ", i_secretkey[i]);
	printf(" Guessing Entropy: %d", guessing_entropy);

	guessing_entropy_average = guessing_entropy_average + guessing_entropy;
}

unsigned int no_of_keys;
void attack_round_all_keys(const unsigned char *filename, unsigned long int iterations)
{
	unsigned char uc_secretkey[16]; 

	FILE *f;
	if((f = fopen(filename, "r")) == NULL){
		printf("Cannot open key file\n");
		exit(-1);
	}
	no_of_keys = 0;
	guessing_entropy_average = 0;

	unsigned int temp_c;
	int i = 0;
	while (fscanf(f, "%x", &temp_c) == 1){
		i_secretkey[i] = temp_c;
		uc_secretkey[i] = (unsigned char) i_secretkey[i];
		i++;
		if(i == 16){
			AES_set_encrypt_key(uc_secretkey, 128, &expanded);
			i = 0;
			no_of_keys++;
			attack_round1(iterations);
		}
	}
	fclose(f);
}

int main(int argc, char **argv)
{
	// Deleting log file
	FILE *f; f = fopen("Guessing_Entropy", "w"); fclose(f);

	unsigned long int iterations;


	iterations = 1 << 8;
	f = fopen("Guessing_Entropy", "a");
	fprintf(f, "\n\nIterations: %ld\n\n", iterations);
	fclose(f);
	attack_round_all_keys("key100", iterations);
	guessing_entropy_average = guessing_entropy_average / no_of_keys;
	printf("\n\n Iterations = %ld     Average Guessing Entropy = %.3f\n\n", iterations, guessing_entropy_average);


	iterations = 1 << 10;
	f = fopen("Guessing_Entropy", "a");
	fprintf(f, "\n\nIterations: %ld\n\n", iterations);
	fclose(f);
	attack_round_all_keys("key100", iterations);
	guessing_entropy_average = guessing_entropy_average / no_of_keys;
	printf("\n\n Iterations = %ld     Average Guessing Entropy = %.3f\n\n", iterations, guessing_entropy_average);


	iterations = 1 << 12;
	f = fopen("Guessing_Entropy", "a");
	fprintf(f, "\n\nIterations: %ld\n\n", iterations);
	fclose(f);
	attack_round_all_keys("key100", iterations);
	guessing_entropy_average = guessing_entropy_average / no_of_keys;
	printf("\n\n Iterations = %ld     Average Guessing Entropy = %.3f\n\n", iterations, guessing_entropy_average);

	iterations = 1 << 14;
	f = fopen("Guessing_Entropy", "a");
	fprintf(f, "\n\nIterations: %ld\n\n", iterations);
	fclose(f);
	attack_round_all_keys("key100", iterations);
	guessing_entropy_average = guessing_entropy_average / no_of_keys;
	printf("\n\n Iterations = %ld     Average Guessing Entropy = %.3f\n\n", iterations, guessing_entropy_average);

	iterations = 1 << 16;
	f = fopen("Guessing_Entropy", "a");
	fprintf(f, "\n\nIterations: %ld\n\n", iterations);
	fclose(f);
	attack_round_all_keys("key100", iterations);
	guessing_entropy_average = guessing_entropy_average / no_of_keys;
	printf("\n\n Iterations = %ld     Average Guessing Entropy = %.3f\n\n", iterations, guessing_entropy_average);

	iterations = 1 << 18;
	f = fopen("Guessing_Entropy", "a");
	fprintf(f, "\n\nIterations: %ld\n\n", iterations);
	fclose(f);
	attack_round_all_keys("key100", iterations);
	guessing_entropy_average = guessing_entropy_average / no_of_keys;
	printf("\n\n Iterations = %ld     Average Guessing Entropy = %.3f\n\n", iterations, guessing_entropy_average);


	iterations = 1 << 20;
	f = fopen("Guessing_Entropy", "a");
	fprintf(f, "\n\nIterations: %ld\n\n", iterations);
	fclose(f);
	attack_round_all_keys("key100", iterations);
	guessing_entropy_average = guessing_entropy_average / no_of_keys;
	printf("\n\n Iterations = %ld     Average Guessing Entropy = %.3f\n\n", iterations, guessing_entropy_average);

}