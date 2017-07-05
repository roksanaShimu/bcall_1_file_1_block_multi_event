/*
 * Viterbi_TcBk_v1_multi_events.cuh
 *
 *  Created on: Jun 20, 2017
 *      Author: roksana
 */

#ifndef VITERBI_TCBK_V1_1_BLOCK_MULTI_EVENTS_CUH_
#define VITERBI_TCBK_V1_1_BLOCK_MULTI_EVENTS_CUH_






#include<iostream>
#include<cuda.h>
#include <ctime>
#include <math.h>
#include <cuda_runtime.h>
#include <vector>
#include <thrust/device_vector.h>

#define __BCALL_I__
#include "bcall-i.h"
#include "bcall.h"
#include "Kmer.cuh"
#include "Predefine_Values.cuh"
#include "Device_Memory_Allocation.cuh"
#include "Viterbi_TcBk_v1_functions.cuh"

using namespace std;





__global__ void kernel(int32_t *devPtr_pm, int32_t *devPtr_log_pr, int32_t *alpha, uint16_t *beta, int32_t p_max, int32_t *d_amax, unsigned int *d_jmax, int32_t *d_mutex, int ith_event, int32_t *d_temp_sum_pmax, int start_event, int end_event){

	unsigned id = threadIdx.x + blockIdx.x * threads_per_block;   //id is j from the serial code

	unsigned alpha_read=0; unsigned alpha_write=1; 	
	int32_t ln_pe_list[number_of_iteration];
	__syncthreads();

unsigned counter=0;
for(int j=start_event; j<=end_event;j++){	
	for (int k=0;k<number_of_iteration;k++){
		unsigned temp_id=id+k*threads_per_block;
		if( temp_id < N_STATES){
			//TD DO: try to move alpha to shared memory so that it can save time accessing global memory
			int32_t ln_pt=dev_ln_ptransition(devPtr_log_pr, temp_id, alpha, beta, alpha_read, counter);


			ln_pe_list[k] = dev_ln_pemission(devPtr_pm, p_max,  ln_pt, temp_id, counter);


			
			//if(ith_event==1) if(threadIdx.x==0) printf("k=%d and ln_pt=%d \n",k,ln_pt);
		
	

		}
	}

	/*if(ith_event==1){
		if(threadIdx.x==0){
			for (int k=0;k<number_of_iteration;k++){
				printf("k=%d and ln_pe=%d \n",k,ln_pe_list[k]);

			}
		}
	}*/

	__syncthreads();

	for (int k=0;k<number_of_iteration;k++){
		unsigned temp_id=id+k*threads_per_block;
		alpha[temp_id+alpha_write*N_STATES] = ln_pe_list[k];

	}

	__syncthreads();

	int32_t max_ln_pe=ln_pe_list[0];   	//in thread now
	unsigned max_id=id;		//in thread now
	for(int k=1; k<number_of_iteration; k++ ){
		if (max_ln_pe<ln_pe_list[k]){
			 max_ln_pe=ln_pe_list[k];
			 max_id=id+k*threads_per_block;
		}
	}


	/*if(ith_event==1){
		if(threadIdx.x==0){
			printf("max_ln_pe_in_thread=%d and max_temp_id_in_thread=%d \n",max_ln_pe,max_id);
		}
	}*/
	__syncthreads();



	//find max ln_pe and the corresponding id
	__shared__ int cache[threads_per_block]; __shared__ unsigned shared_max_id_available_in_thread;
	cache[threadIdx.x]=max_ln_pe;
	unsigned max_id_available_in_thread=threadIdx.x;
	__syncthreads();


	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			if(cache[threadIdx.x]< cache[threadIdx.x + i]){
				cache[threadIdx.x]=cache[threadIdx.x + i];
				max_id_available_in_thread=threadIdx.x + i;

			}
		}

		__syncthreads();
		i /= 2;
	}
	__syncthreads();
	
	p_max=cache[0];

	if (threadIdx.x==0){
		*d_amax=cache[0];
		shared_max_id_available_in_thread=max_id_available_in_thread;
		*d_temp_sum_pmax=*d_temp_sum_pmax+cache[0];
	}
	__syncthreads();

	if (threadIdx.x==shared_max_id_available_in_thread){
		*d_jmax=max_id_available_in_thread;
	}
	__syncthreads();


alpha_read=!alpha_read; alpha_write=!alpha_write;
counter++;
}
	

}








void preprocess_for_parallel(struct bcall_tcbk * fix, int32_t alpha[N_STATES], int32_t p_max, int i, int32_t * devPtr_pm, int32_t * devPtr_log_pr, int32_t * dev_alpha,  uint16_t *dev_beta, uint16_t *h_beta, int32_t *h_amax, int32_t *d_amax, unsigned int *h_jmax, unsigned int *d_jmax, int32_t *d_mutex, int32_t *h_temp_sum_pmax, int32_t *d_temp_sum_pmax  ){

	struct fix_event * e;

	int j, k;


	int start_event=i; int end_event;
	if(start_event== (fix->n_events-1) ){
		end_event=start_event;	
	}else if((start_event+number_of_events_per_kernel-1)<fix->n_events){
		end_event=start_event+number_of_events_per_kernel-1;
	}else{
		end_event=fix->n_events-start_event-1;
	}

	const int number_of_events=end_event-start_event+1;
	//cout<<"number_of_events="<<number_of_events<<endl;

	//event
	int32_t event_host[4* number_of_events];

	k=0;
	for(j=start_event; j<=end_event; j++){

		e = &fix->event[j];
		event_host[k]=e->stdv; k++;
		event_host[k]=e->stdv_inv; k++;
		event_host[k]=e->mean; k++;
		event_host[k]=e->log_stdv_1p5; k++;
	}

	/* adjust the maximum probability for this round */
	*h_amax = INT32_MIN; *h_jmax=N_STATES;


	cudaMemset(d_mutex, 0, sizeof(int32_t));
	
	*h_temp_sum_pmax=0;
	cudaMemset(d_temp_sum_pmax, 0, sizeof(int32_t));


	//copy event_host to constant memory
	cudaMemcpyToSymbol(d_ev, event_host, 4*sizeof(int32_t)*number_of_events);

	cudaMemcpy(d_amax, h_amax, sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_jmax, h_jmax, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_alpha,alpha, sizeof(int32_t)*N_STATES*2, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	kernel<<< number_of_blocks, threads_per_block >>>(devPtr_pm, devPtr_log_pr, dev_alpha, dev_beta, p_max, d_amax, d_jmax, d_mutex, i, d_temp_sum_pmax, start_event,end_event);
	cudaDeviceSynchronize();

	cudaMemcpy(alpha,dev_alpha, sizeof(int32_t)*N_STATES*2, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_beta,dev_beta, sizeof(uint16_t)*N_STATES*number_of_events, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_amax,d_amax, sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_jmax,d_jmax, sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_temp_sum_pmax,d_temp_sum_pmax, sizeof(int32_t), cudaMemcpyDeviceToHost);


	
	int counter=0;
	for(k=start_event; k<=end_event; k++){
		for (j = 0; j < N_STATES; ++j) {
			fix->beta[k % TRACEBACK_LEN][j]=h_beta[j+counter*N_STATES];
		}
		counter++;
	}

	if(number_of_events%2!=0){
		for (j = 0; j < N_STATES; ++j) {
			alpha[j]=alpha[j+N_STATES];
			
		}
	}


}


void tcbk_sequence_fill(struct bcall_tcbk * fix)
{
	/* Pr[ MLSS producing e_1 ... e_i, with S_i == j ] */
	int32_t alpha[N_STATES*2];
	struct fix_event * e;
	unsigned int j_max;
	int32_t a_max;
	unsigned int i;
	unsigned int j, k;
	/* Store the maximum probability in each round */
	int32_t p_max = INT32_MIN;

	//DBG(DBG_INFO, "begin");
	cout<<"begin"<<endl;
	fix->sum_log_p_max = 0;

	/* XXX: Read from memory */
	e = &fix->event[0];
	/* adjust the maximum probability for this round */
	a_max = INT32_MIN;
	j_max = N_STATES;
	for (j = 0; j < N_STATES; ++j) {
		struct fix_pm_state * pm = &fix->map[j].pm;
		int32_t ln_pe;

		ln_pe = ln_pemission(pm, e, p_max,  0);

		if (ln_pe > a_max) {
			a_max = ln_pe;
			j_max = j;
		}

		alpha[j] = ln_pe;
		fix->beta[0][j] = N_STATES;
	}


	/* Save p_max for next round */
	p_max = a_max;
	/* Accumulate max probability */
	fix->sum_log_p_max += a_max;

	//--------------------------------------------------------------------------
	//allocate event independent variables
	//--------------------------------------------------------------------------
	//pm
	int32_t pm_rearranged[7*N_STATES]; size_t pm_rearranged_size= sizeof(int32_t) * 7*N_STATES;

	for (j = 0; j < N_STATES; ++j) {
		struct fix_pm_state * pm = &fix->map[j].pm;

		unsigned group_no=j/warp_size;
		unsigned start_index=group_no*warp_size*7 +j%warp_size;

		pm_rearranged[start_index]=pm->sd_mean;
		pm_rearranged[start_index+1*warp_size]=pm->sd_mean_inv;
		pm_rearranged[start_index+2*warp_size]=pm->sd_lambda_p5;
		pm_rearranged[start_index+3*warp_size]=pm->level_mean;
		pm_rearranged[start_index+4*warp_size]=pm->level_stdv_inv_2;
		pm_rearranged[start_index+5*warp_size]=pm->log_sd_lambda_p5;
		pm_rearranged[start_index+6*warp_size]=pm->log_level_stdv_2pi;

	}

	//use page lock memory for pm
	int32_t * devPtr_pm=assign_page_locked_memory_int32_t(pm_rearranged, pm_rearranged_size);


	//log_pr
	int32_t log_pr_rearranged[21*N_STATES]; size_t log_pr_rearranged_size= sizeof(int32_t) *21* N_STATES;
	for (j = 0; j < N_STATES; ++j) {

		unsigned group_no=j/warp_size;
		unsigned start_index=group_no*warp_size*21 +j%warp_size;

		for(k=0;k<21;k++){
			log_pr_rearranged[start_index+ k*warp_size]=fix->map[j].log_pr[k];
		}
	}
	/*cout<<"log_pr from CPU"<<endl;
	for (j = 0; j < 10; ++j) {
		unsigned group_no=j/warp_size;
		unsigned start_index=group_no*warp_size*21 +j%warp_size;

		cout<<"  " << fix->map[j].log_pr[0]<<"   "<<log_pr_rearranged[start_index]<<endl;

	}
	cout<<endl<<endl;*/

	//use page-locked memory
	int32_t * devPtr_log_pr=assign_page_locked_memory_int32_t(log_pr_rearranged, log_pr_rearranged_size);



	//alpha
	int32_t *dev_alpha;
	cudaMalloc((void**)&dev_alpha, sizeof(int32_t)*N_STATES*2) ; // device
	//keep alpha in global memory


	//beta needs a write only memory : so global memory
	uint16_t * dev_beta;
	cudaMalloc((void**)&dev_beta, sizeof(uint16_t)*N_STATES*number_of_events_per_kernel) ; // device
	uint16_t * h_beta;
	h_beta=(uint16_t*)malloc(sizeof(uint16_t)*N_STATES*number_of_events_per_kernel);



	// a_max and j_max;
	int32_t *h_amax;
	h_amax = (int32_t*)malloc(sizeof(int32_t)); //host
	int32_t *d_amax;
	cudaMalloc((void**)&d_amax, sizeof(int32_t));//device

	unsigned int *h_jmax;
	h_jmax = (unsigned int*)malloc(sizeof(unsigned int)); //host
	unsigned int *d_jmax;
	cudaMalloc((void**)&d_jmax, sizeof(unsigned int));//device

	int32_t *d_mutex;
	cudaMalloc((void**)&d_mutex, sizeof(int32_t));

	int32_t *h_temp_sum_pmax;
	h_temp_sum_pmax = (int32_t*)malloc(sizeof(int32_t)); //host
	int32_t *d_temp_sum_pmax;
	cudaMalloc((void**)&d_temp_sum_pmax, sizeof(int32_t));//device
	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------

	for (i = 1; i < fix->n_events; i=i+number_of_events_per_kernel) {

		if ((i % TRACEBACK_CNT) == 1)
			
			fixpt_traceback(fix, i, j_max); // need to update beta



		preprocess_for_parallel(fix, alpha, p_max, i, devPtr_pm, devPtr_log_pr, dev_alpha, dev_beta, h_beta, h_amax, d_amax, h_jmax, d_jmax, d_mutex, h_temp_sum_pmax, d_temp_sum_pmax );

		a_max=*h_amax; j_max=*h_jmax;



		/* Save p_max for next round */
		p_max = a_max;


		if (i<5) cout<<"i= "<<i<<"   a_max= "<<a_max<< " *h_temp_sum_pmax= "<<*h_temp_sum_pmax<<endl;
		fix->sum_log_p_max += *h_temp_sum_pmax; //a_max;
		
	

	}

	fixpt_traceback(fix, i, j_max); // need to update beta

	//cout<<"beta[0][0]= "<<fix->beta[0][0]<<"  beta[0][1]= "<<  fix->beta[0][1]<<"  beta[0][200]= "<<  fix->beta[0][200]<<endl;


	//cout<<"state seq [0]= "<<fix->state_seq[0]<<"  state seq [1]= "<<  fix->state_seq[1]<<"  state seq [200]= "<<  fix->state_seq[200]<<endl;

	//DBG(DBG_INFO, "end");
	cout<<"end"<<endl;

}







#endif /* VITERBI_TCBK_V1_MULTI_EVENTS_CUH_ */
