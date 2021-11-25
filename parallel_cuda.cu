/****************************************************************************
 *
 * parallel_sort_matching.cc
 *
 * Serial and Parallel sort-based matching
 *
 * Copyright (C) 2016, 2018, 2019 Moreno Marzolla, Gabriele D'Angelo
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

#include <vector>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <parallel/algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>


#include "timing.h"
#include "interval.h"
#include "sort_matching_common.hh"
#include "parallel_cuda.hh"


/**
 * Parallel sort-based matching algorithm
 */
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// CUDA copy routine for copying subscriptions to endpoints
__global__ void cuda_copy_subscription(int n, struct endpoint* endpoints, struct interval* sub)
{
    // Push subscription intervals into the endpoints array :  all variables are in GPU memory
        // total number of threads in kernel
        int num_threads=blockDim.x*gridDim.x;
        //printf("num_threads %d\n", num_threads);

    // divide and calculate start and end for this thread
        int n_local=(n+num_threads-1)/num_threads;
        int n_start=n_local*(blockIdx.x*blockDim.x + threadIdx.x);
        int n_end=n_start+n_local;
        //printf ("n %d n_local %d, n_start %d, n_end %d\n",n, n_local, n_start, n_end);

    // for last interval        
        if (n_end>n) {n_end=n;}

    // first the subscription
        for (size_t i=n_start; i<n_end; i++) {
                // check that interval is legal i.e. lower is less than upper
            assert(static_cast<struct interval>(sub[i]).lower < static_cast<struct interval>(sub[i]).upper);
            // initialize endpoint with lower
            endpoints[i] = endpoint(i, static_cast<struct interval>(sub[i]).lower, endpoint::LOWER, endpoint::SUBSCRIPTION);
            // initialize endpoint with upper
            endpoints[i+n] = endpoint(i, static_cast<struct interval>(sub[i]).upper, endpoint::UPPER, endpoint::SUBSCRIPTION);
        }

}
// CUDA copy routine for copying updates to endpoints
__global__ void cuda_copy_updates(int n, int m, struct endpoint* endpoints, struct interval* upd)
{
        // total number of threads in kernel
        int num_threads=blockDim.x*gridDim.x;
        //printf("num_threads %d\n", num_threads);

    // divide and calculate start and end for this thread
        int m_local=(m+num_threads-1)/num_threads;
        int m_start=m_local*(blockIdx.x*blockDim.x + threadIdx.x);
        int m_end=m_start+m_local;
        //printf ("n %d n_local %d, n_start %d, n_end %d\n",n, n_local, n_start, n_end);

    // for last interval        
        if (m_end>m) {m_end=m;}

        for (size_t i=m_start; i<m_end; i++) {
                // check that interval is legal i.e. lower is less than upper
            assert(static_cast<struct interval>(upd[i]).lower < static_cast<struct interval>(upd[i]).upper);
            // initialize endpoint with lower
            endpoints[2*n+i] = endpoint(i, static_cast<struct interval>(upd[i]).lower, endpoint::LOWER, endpoint::UPDATE);
            // initialize endpoint with upper
            endpoints[2*n+m+i] = endpoint(i, static_cast<struct interval>(upd[i]).upper, endpoint::UPPER, endpoint::UPDATE);
        }

}

// cuda routine for pass 1 reduction this is the same as parallel sort
__global__ void cuda_pass_1(int n_endpoints, int n, int m, class epset **sub_insert, class epset **sub_delete, class epset **upd_insert, class epset **upd_delete, struct endpoint *endpoints)
{
        // total number of threads in kernel
        int thread_count=blockDim.x*gridDim.x;
        int my_rank=blockIdx.x*blockDim.x + threadIdx.x;

	//printf ("thread %d , n_endpoints %d, n %d m %d\n",my_rank, n_endpoints, n, m);

        epset *my_sub_insert=sub_insert[my_rank]=new epset(n);
        epset *my_sub_delete=sub_delete[my_rank]=new epset(n);
        epset *my_upd_insert=upd_insert[my_rank]=new epset(m);
        epset *my_upd_delete=upd_delete[my_rank]=new epset(m);

	//printf ("subs_insert size %lu, subs_delete_size %lu upd_insert size %lu , upd_delete size %lu my_rank %d\n",  my_sub_insert->size(), my_sub_delete->size(), my_upd_insert->size(), my_upd_delete->size(),my_rank);

    // divide and calculate start and end for this thread
        const int local_n = (n_endpoints + thread_count - 1) / thread_count;
        const int local_idx_start = my_rank*local_n;
        int local_idx_end = local_idx_start + local_n;
        if (local_idx_end>n_endpoints) {local_idx_end=n_endpoints;}
        //printf("n_endpoints %d num_threads %d my rank %d local_n %d, local_idx_start %d local_edx_end %d\n",n_endpoints, thread_count,my_rank, local_n, local_idx_start, local_idx_end);
        for (size_t idx = local_idx_start; idx < local_idx_end; idx++) {
                struct endpoint& my_ep= endpoints[idx];
		//printf ("endpoint %lu , my_ep.t %d my_ep.e %d my_ep.id %lu\n", idx, my_ep.t, my_ep.e, my_ep.id);
		
                if (my_ep.t == endpoint::SUBSCRIPTION) {
                    if (my_ep.e == endpoint::LOWER) {
                        my_sub_insert->insert(my_ep.id);
			//printf ("sub_insert inserted id %lu\n", my_ep.id);
                    } else {
                        if ( my_sub_insert->find(my_ep.id) )
                        {
                            my_sub_insert->remove(my_ep.id);
			    //printf ("sub_insert removed id %lu\n", my_ep.id);
                        }
                        else {
                            my_sub_delete->insert(my_ep.id);
			//printf ("sub_delete inserted id %lu\n", my_ep.id);
                        }
                    }
                } else {
                    if (my_ep.e == endpoint::LOWER) {
                        my_upd_insert->insert(my_ep.id);
			//printf ("upd_insert inserted id %lu\n", my_ep.id);
                    } else {
                        if ( my_upd_insert->find(my_ep.id) ) {
                            my_upd_insert->remove(my_ep.id);
			    //printf ("upd_insert removed id %lu\n", my_ep.id);
			}
                        else {
                            my_upd_delete->insert(my_ep.id);
			//printf ("upd_delete inserted id %lu\n", my_ep.id);
			}
                    }
                }
            }
	//printf ("subs_insert size %lu, subs_delete_size %lu upd_insert size %lu , upd_delete size %lu my_rank %d\n",  my_sub_insert->size(), my_sub_delete->size(), my_upd_insert->size(), my_upd_delete->size(),my_rank);

}
// COMMENT : merges in parallel kernel function
__global__ 
void cuda_merge_parallel(unsigned int *data, unsigned int *data2, unsigned int buflen, unsigned int *count_array)
{
        // total number of threads in kernel
        int thread_count=blockDim.x*gridDim.x;
        int my_rank=blockIdx.x*blockDim.x + threadIdx.x;

    // divide and calculate start and end for this thread
        const int local_n = (buflen + thread_count - 1) / thread_count;
        const int local_idx_start = my_rank*local_n;
        int local_idx_end = local_idx_start + local_n;
        if (local_idx_end>buflen) {local_idx_end=buflen;}
        //printf("n_endpoints %d num_threads %d my rank %d local_n %d, local_idx_start %d local_edx_end %d\n",n_endpoints, thread_count,my_rank, local_n, local_idx_start, local_idx_end);
	count_array[my_rank]=0;
        for (size_t idx = local_idx_start; idx < local_idx_end; idx++) {
		data[idx]|=data2[idx];
                count_array[my_rank] += __popc(data[idx]);
	}
}

// COMMENT : merges in parallel wrapper
__device__
void cuda_merge(unsigned int *data, unsigned int *data2, unsigned int buflen, size_t *count)
{
    const int n_blocks=10;
    const int n_threads=1;
    unsigned int *count_array=new unsigned int[n_threads*n_blocks];
    *count=0;
    cuda_merge_parallel<<<n_blocks,n_threads>>>(data,data2,buflen,count_array);
    cudaDeviceSynchronize();
    for (int ii=0;ii<n_threads*n_blocks;++ii) {(*count)+=count_array[ii];}
    delete [] count_array;
}

// COMMENT : subtract in parallel kernel function
__global__ 
void cuda_subtract_parallel(unsigned int *data, unsigned int *data2, unsigned int buflen, unsigned int *count_array)
{
        // total number of threads in kernel
        int thread_count=blockDim.x*gridDim.x;
        int my_rank=blockIdx.x*blockDim.x + threadIdx.x;

    // divide and calculate start and end for this thread
        const int local_n = (buflen + thread_count - 1) / thread_count;
        const int local_idx_start = my_rank*local_n;
        int local_idx_end = local_idx_start + local_n;
        if (local_idx_end>buflen) {local_idx_end=buflen;}
        //printf("n_endpoints %d num_threads %d my rank %d local_n %d, local_idx_start %d local_edx_end %d\n",n_endpoints, thread_count,my_rank, local_n, local_idx_start, local_idx_end);
	count_array[my_rank]=0;
        for (size_t idx = local_idx_start; idx < local_idx_end; idx++) {
		data[idx] = data[idx] & ~data2[idx];
                count_array[my_rank] += __popc(data[idx]);
	}
}

// COMMENT : subtract in parallel wrapper
__device__
void cuda_subtract(unsigned int *data, unsigned int *data2, unsigned int buflen, size_t *count)
{
    const int n_blocks=10;
    const int n_threads=1;
    unsigned int *count_array=new unsigned int[n_threads*n_blocks];
    *count=0;
    cuda_subtract_parallel<<<n_blocks,n_threads>>>(data,data2,buflen,count_array);
    cudaDeviceSynchronize();
    for (int ii=0;ii<n_threads*n_blocks;++ii) {(*count)+=count_array[ii];}
    delete [] count_array;
}

// cuda routine for pass 1 reduction this is the same as parallel sort
__global__ void cuda_pass_2(class epset **sub_insert, class epset **sub_delete, class epset **upd_insert, class epset **upd_delete, class epset **subscriptions, class epset **updates, int thread_count, int n, int m)
{
        subscriptions[0]=new epset(n);
        updates[0]=new epset(m);
            for (int p=1; p<thread_count; p++) {
        	updates[p]=new epset(m);
        	subscriptions[p]=new epset(n);
	    }
            for (int p=1; p<thread_count; p++) {
                (*subscriptions[p])
                    .merge( *subscriptions[p-1] )
                    .merge( *sub_insert[p-1] )
                    .subtract( *sub_delete[p-1] );

                (*updates[p])
                    .merge( *updates[p-1] )
                    .merge( *upd_insert[p-1] )
                    .subtract( *upd_delete[p-1] );
                 //printf(" p %d , subs size %lu updates size %lu\n", p, subscriptions[p]->size(), updates[p]->size()); 
            }
}


// cuda routine for pass 3 reduction ; this is the same as parallel sort
__global__ void cuda_pass_3(int n_endpoints, class epset **subscriptions, class epset **updates, struct endpoint* endpoints, size_t *d_nmatches)
{
        // total number of threads in kernel
        int thread_count=blockDim.x*gridDim.x;
        int my_rank=blockIdx.x*blockDim.x + threadIdx.x;

	//printf ("thread %d , n_endpoints %d, n %d m %d\n",my_rank, n_endpoints, n, m);
        epset& my_sub( *subscriptions[my_rank] );
        epset& my_upd( *updates[my_rank] );


    // divide and calculate start and end for this thread
        size_t nmatches=0;
        const int local_n = (n_endpoints + thread_count - 1) / thread_count;
        const int local_idx_start = my_rank*local_n;
        int local_idx_end = local_idx_start + local_n;
        if (local_idx_end>n_endpoints) {local_idx_end=n_endpoints;}
        //printf("n_endpoints %d num_threads %d my rank %d local_n %d, local_idx_start %d local_edx_end %d\n",n_endpoints, thread_count,my_rank, local_n, local_idx_start, local_idx_end);
        for (size_t idx = local_idx_start; idx < local_idx_end; idx++) {
                const endpoint& my_ep = endpoints[idx];

                if (my_ep.t == endpoint::SUBSCRIPTION) {
                    if (my_ep.e == endpoint::LOWER) {
                        // std::cout << "S I " << my_ep.id << std::endl;
                        my_sub.insert(my_ep.id);
                    } else {
                        // std::cout << "S R " << my_ep.id << std::endl;
                        my_sub.remove(my_ep.id);
                        nmatches += my_upd.size();
                    }
                } else {
                    if (my_ep.e == endpoint::LOWER) {
                        // std::cout << "I R " << my_ep.id << std::endl;
                        my_upd.insert(my_ep.id);
                    } else {
                        // std::cout << "I R " << my_ep.id << std::endl;
                        my_upd.remove(my_ep.id);
                        nmatches += my_sub.size();
                    }
                }
            }
	*(d_nmatches+my_rank)=nmatches;
	//printf ("nmatches %lu, subs size %lu , upd size %lu my_rank %d\n", nmatches,  my_sub.size(), my_upd.size(), my_rank);

}

extern "C" 
size_t parallel_cuda( const struct interval* sub_in, size_t n,
                               const struct interval* upd_in, size_t m )
{
    const size_t n_endpoints = 2*(n+m);
    struct timer timing;

    timing_init( &timing );
    timing_start( &timing );

    double total_time;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024);
    // following is created on gpu memory to hold endpoints
    thrust::device_vector<endpoint> endpoints(n_endpoints);

    // following are created on gpu memory to hold the intervals sub and upd , they are initialized in the constructor
    thrust::device_vector<struct interval> sub(sub_in,sub_in+n);
    thrust::device_vector<struct interval> upd(upd_in,upd_in+m);

// COMMENT print only if _PRINT_ is defined to save runtime    
// print the input for checking the output
#ifdef _PRINT_
    std::cout << "Printing input arrays" << std::endl;
    std::cout << "n is " << n << std::endl;
    for (int ii=0;ii<n;++ii) {
		std::cout << "sub [" << ii << "] is " << sub_in[ii].lower << " " << sub_in[ii].upper << std::endl;
    }
    std::cout << "m is " << n << std::endl;
    for (int ii=0;ii<m;++ii) {
		std::cout << "upd [" << ii << "] is " << upd_in[ii].lower << " " << sub_in[ii].upper << std::endl;
    }
#endif
    // Push subscription intervals into the endpoints array :  all variables are in GPU memory
    // 1 blocks of 4 threads 
    const int n_blocks=10;
    const int n_threads=1;
    const int maxproc=n_blocks*n_threads;
    std::cout << "using cuda threads " << maxproc << std::endl;

// COMMENT cast to raw pointers for passing to cuda kernel
    struct endpoint* d_endpoints =  thrust::raw_pointer_cast(&endpoints[0]);
    struct interval* d_subs = thrust::raw_pointer_cast(&sub[0]);
    struct interval* d_upd = thrust::raw_pointer_cast(&upd[0]);
    const int copy_blocks=4;
    const int copy_threads=1;
// now copy the subscription interval in a cuda kernel
    cuda_copy_subscription<<<copy_blocks,copy_threads>>>(n, d_endpoints, d_upd);
    checkCuda(cudaGetLastError());

// wait for above kernel to finish
    checkCuda(cudaDeviceSynchronize());

// now copy the update in a cuda kernel
    cuda_copy_updates<<<copy_blocks,copy_threads>>>(n,m, d_endpoints, d_subs);
    checkCuda(cudaGetLastError());

// wait for above kernel to finish
    checkCuda(cudaDeviceSynchronize());

//COMMENT : print timing for the copy
    timing_stop( &timing );
    total_time = timing_get_average( &timing );
    std::cout << "copy time seconds " << total_time << std::endl;
    fflush (stdout);
// COMMENT init timing for sort
    timing_init( &timing );
    timing_start( &timing );
	// Now sort them using thrust algorithm
    thrust::sort (endpoints.begin(),endpoints.end());

//COMMENT : print timing for the sort
    timing_stop( &timing );
    total_time = timing_get_average( &timing );
    std::cout << "sort time seconds " << total_time << std::endl;
    fflush (stdout);

// COMMENT print only if _PRINT_ is defined to save runtime    
#ifdef _PRINT_
    // Now print the output , can remove later
    std::cout << "Printing endpoints"  << std::endl;
    //size() returns the size of vector
    std::cout << "endpoints has size " << endpoints.size() << std::endl;

    // print contents of device vector
    for(int i = 0; i < endpoints.size(); i++) {
        std::cout << std::setw(10) << static_cast<struct endpoint>(endpoints[i]).v << " " ;
        if (static_cast<struct endpoint>(endpoints[i]).e==endpoint::LOWER)
		std::cout << "Lower ";
	else if (static_cast<struct endpoint>(endpoints[i]).e==endpoint::UPPER)
		std::cout << "Upper ";
        std::cout << static_cast<struct endpoint>(endpoints[i]).id << " ";
        if (static_cast<struct endpoint>(endpoints[i]).t==endpoint::SUBSCRIPTION)
		std::cout << "Sub" << std::endl;
	else if (static_cast<struct endpoint>(endpoints[i]).t==endpoint::UPDATE)
		std::cout << "Upd" << std::endl;
    }
#endif

// COMMENT TODO these will be vectors
// COMMENT init timing for prep
    timing_init( &timing );
    timing_start( &timing );

    thrust::device_vector<epset*>  my_sub_insert(maxproc);
    thrust::device_vector<epset*>  my_sub_delete(maxproc);
    thrust::device_vector<epset*>  my_upd_insert(maxproc);
    thrust::device_vector<epset*>  my_upd_delete(maxproc);
    thrust::device_vector<epset*>  subscriptions(maxproc);
    thrust::device_vector<epset*>  updates(maxproc);
    thrust::device_vector<size_t>  nmatches(maxproc);
// COMMENT convert to raw for passing to cuda kernel
    epset** d_my_sub_insert =  thrust::raw_pointer_cast(&my_sub_insert[0]);
    epset** d_my_sub_delete =  thrust::raw_pointer_cast(&my_sub_delete[0]);
    epset** d_my_upd_insert =  thrust::raw_pointer_cast(&my_upd_insert[0]);
    epset** d_my_upd_delete =  thrust::raw_pointer_cast(&my_upd_delete[0]);
    epset** d_subscriptions =  thrust::raw_pointer_cast(&subscriptions[0]);
    epset** d_updates =  thrust::raw_pointer_cast(&updates[0]);
    size_t *d_nmatches=thrust::raw_pointer_cast(&nmatches[0]);

// COMMENT print timing for init data
    timing_stop( &timing );
    total_time = timing_get_average( &timing );
    std::cout << "init data seconds " << total_time << std::endl;
    fflush (stdout);
// COMMENT init timing for pass 1
    timing_init( &timing );
    timing_start( &timing );
// COMMENT pass 1 parallel
    cuda_pass_1<<<n_blocks,n_threads>>>(n_endpoints,  n,  m, d_my_sub_insert, d_my_sub_delete, d_my_upd_insert, d_my_upd_delete, d_endpoints);
    checkCuda(cudaGetLastError());

// wait for above kernel to finish
    checkCuda(cudaDeviceSynchronize());

// COMMENT print timing for pass 1
    timing_stop( &timing );
    total_time = timing_get_average( &timing );
    std::cout << "pass 1 scan  time seconds " << total_time << std::endl;
    fflush (stdout);
// COMMENT init timing for pass 2
    timing_init( &timing );
    timing_start( &timing );
// COMMENT pass 2 serial
    cuda_pass_2<<<1,1>>>(d_my_sub_insert, d_my_sub_delete, d_my_upd_insert, d_my_upd_delete, d_subscriptions, d_updates, maxproc,n,m);
    checkCuda(cudaGetLastError());

// wait for above kernel to finish
    checkCuda(cudaDeviceSynchronize());

//COMMENT : print timing for the second pass
    timing_stop( &timing );
    total_time = timing_get_average( &timing );
    std::cout << "pass 2  merge / subtract time seconds " << total_time << std::endl;
    fflush (stdout);

// COMMENT init timing for pass 3
    timing_init( &timing );
    timing_start( &timing );

// COMMENT pass 3 parallel this is the same as serial sort

    cuda_pass_3<<<n_blocks, n_threads>>>(n_endpoints, d_subscriptions, d_updates, d_endpoints, d_nmatches);
    checkCuda(cudaGetLastError());

// wait for above kernel to finish
    checkCuda(cudaDeviceSynchronize());

//COMMENT : print timing for the pass 3 scan
    timing_stop( &timing );
    total_time = timing_get_average( &timing );
    std::cout << "pass 3 scan time seconds " << total_time << std::endl;
    fflush (stdout);
    size_t sum = thrust::reduce(nmatches.begin(), nmatches.end(), (size_t) 0, thrust::plus<size_t>());
    // vectors are automatically deleted when the function returns
    return sum;
}
//COMMENT memcpy in cuda parallel kernels
__global__ 
void cuda_memcpy_parallel(unsigned int *data, unsigned int *data2, unsigned int buflen)
{
        // total number of threads in kernel
        int thread_count=blockDim.x*gridDim.x;
        int my_rank=blockIdx.x*blockDim.x + threadIdx.x;

    // divide and calculate start and end for this thread
        const int local_n = (buflen + thread_count - 1) / thread_count;
        const int local_idx_start = my_rank*local_n;
        int local_idx_end = local_idx_start + local_n;
        if (local_idx_end>buflen) {local_idx_end=buflen;}
        //printf("n_endpoints %d num_threads %d my rank %d local_n %d, local_idx_start %d local_edx_end %d\n",n_endpoints, thread_count,my_rank, local_n, local_idx_start, local_idx_end);
	 memcpy( data+local_idx_start, data2+local_idx_start, (local_idx_end-local_idx_start)*sizeof(*data));
}

//COMMENT memcpy in cuda parallel wrapper
__device__
void cuda_memcpy(unsigned int *data, unsigned int *data2, unsigned int buflen)
{
    const int n_blocks=1;
    const int n_threads=4;
    cuda_memcpy_parallel<<<n_blocks,n_threads>>>(data,data2,buflen);
    cudaDeviceSynchronize();
}
//COMMENT memset in parallel kernel function
__global__ 
void cuda_memset_parallel(unsigned int *data, unsigned int val, unsigned int buflen)
{
        // total number of threads in kernel
        int thread_count=blockDim.x*gridDim.x;
        int my_rank=blockIdx.x*blockDim.x + threadIdx.x;

    // divide and calculate start and end for this thread
        const int local_n = (buflen + thread_count - 1) / thread_count;
        const int local_idx_start = my_rank*local_n;
        int local_idx_end = local_idx_start + local_n;
        if (local_idx_end>buflen) {local_idx_end=buflen;}
        //printf("n_endpoints %d num_threads %d my rank %d local_n %d, local_idx_start %d local_edx_end %d\n",n_endpoints, thread_count,my_rank, local_n, local_idx_start, local_idx_end);
	 memset( data+local_idx_start, val, (local_idx_end-local_idx_start)*sizeof(*data));
}

//COMMENT memset in parallel wrapper
__device__
void cuda_memset(unsigned int *data, unsigned int val, unsigned int buflen)
{
    const int n_blocks=1;
    const int n_threads=4;
    cuda_memset_parallel<<<n_blocks,n_threads>>>(data,val,buflen);
    cudaDeviceSynchronize();
}
