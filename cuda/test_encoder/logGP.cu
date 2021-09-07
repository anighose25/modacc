#include <cuda.h>
#include <time.h>
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define DEV_QUERY 0
#define SEQ 1
#define ASYNC 1
#define max(a,b) (a>b)?a:b
#ifndef EVENT_PROFILE
#define EVENT_PROFILE 0
#endif
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
   cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}


void initializeOnes(float *input, int num_elements)
{
    for(unsigned int i = 0; i < num_elements; i++)
    {
        input[i] = 1.0;
    }
}



float calculate_G(cudaEvent_t start[], cudaEvent_t stop[], int num_streams, long long int size)
{
    float event_recorded_time = 0, total_time =0;
    for(int i=0;i<num_streams;i++)
    {
       cudaEventElapsedTime(&event_recorded_time, start[i], stop[i]);
       total_time += event_recorded_time; 
    }
    return total_time/size;
}

float calculate_g(cudaEvent_t start[], cudaEvent_t stop[], int num_streams, long long int size)
{
    float event_recorded_time = 0, total_time =0;
    for(int i=1;i<num_streams;i++)
    {
       cudaEventElapsedTime(&event_recorded_time, start[i],stop[i-1]);
       //printf("Event recorded time for g : %f\n",event_recorded_time);
       total_time += event_recorded_time; 
    }
    return total_time/num_streams;
}



int main(int argc, char *argv[])
{
    //Query Device
#if DEV_QUERY
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> %s Starting...\n", argv[0]);
    printf("> Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // check if device supports hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
    {
        if (deviceProp.concurrentKernels == 0)
        {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                    "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else
        {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n",
            deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

#endif
    //Variable Initialization
    
    
    
    
    long long int size = atoi(argv[1]);
    int num_streams = atoi(argv[2]);
    


    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    size_t nbytes_A = size*sizeof(float);
   
    Stopwatch sw, tsync;


    // host array creation (page-locked)

    float *h_input = NULL; 
    CHECK(cudaMallocHost((void **)&h_input, nbytes_A));
    initializeOnes(h_input, size);

    
    // device memory creation

    float *d_A = NULL;

    CHECK(cudaMalloc((void **)&d_A, nbytes_A));


    // stream configuration


    cudaStream_t memory[num_streams];

    cudaEvent_t start[num_streams], stop[num_streams];
    cudaEvent_t seq_start, seq_end;
    CHECK(cudaEventCreate(&seq_start));
    CHECK(cudaEventCreate(&seq_end));
    for(int i=0;i<num_streams;i++)
    {
        CHECK(cudaEventCreate(&start[i]));
        CHECK(cudaEventCreate(&stop[i]));
    }


    for(int i=0;i<num_streams;i++)
    {
        CHECK(cudaStreamCreate(&memory[i]));
    }



#if SEQ

    sw.start();

    //H2D Copy of 1 byte

    cudaEventRecord(seq_start);
    CHECK(cudaMemcpy(d_A, h_input, 1, cudaMemcpyHostToDevice));
    cudaEventRecord(seq_end);
    float event_recorded_time = 0;
    CHECK(cudaThreadSynchronize());    
    cudaEventElapsedTime(&event_recorded_time, seq_start, seq_end);
    printf("L+o for H2D: %lf\n",event_recorded_time);

    cudaEventRecord(seq_start);
    CHECK(cudaMemcpy(h_input, d_A, 1, cudaMemcpyDeviceToHost));
    cudaEventRecord(seq_end);
    CHECK(cudaThreadSynchronize());    
    cudaEventElapsedTime(&event_recorded_time, seq_start, seq_end);
    printf("L+o for D2H: %lf\n",event_recorded_time);


#endif


#if ASYNC

// Multistream implementation 

    int granularity = num_streams;
    int offset = 0;
    int buffer_offset_A;
    int sub_A =  size/granularity;    
    size_t sub_nbytes_A = nbytes_A/granularity;
    sw.start();
    for(int i=0;i<num_streams;i++)
    {
        buffer_offset_A = i*sub_A;    
        cudaEventRecord(start[i],memory[i]);
        CHECK(cudaMemcpyAsync(&d_A[buffer_offset_A], &h_input[buffer_offset_A], sub_nbytes_A,
                              cudaMemcpyHostToDevice, memory[i]));
        cudaEventRecord(stop[i],memory[i]);
    }
    CHECK(cudaThreadSynchronize());    
    sw.stop();

    printf("H2D Time:%f\n", sw.GetTimeInSeconds());
    for(int i=0;i<num_streams;i++)
    {
        cudaEventSynchronize(stop[i]);
    }    
    printf("G for H2D: %f\n",calculate_G(start,stop,num_streams,size));
    printf("g for H2D: %f\n",calculate_g(start,stop,num_streams,size));
    buffer_offset_A=0;
    sw.restart();
    for(int i=0;i<num_streams;i++)
    {
        buffer_offset_A = i*sub_A;    
        cudaEventRecord(start[i],memory[i]);
        CHECK(cudaMemcpyAsync(&h_input[buffer_offset_A], &d_A[buffer_offset_A], sub_nbytes_A,
                              cudaMemcpyDeviceToHost, memory[i]));
        cudaEventRecord(stop[i],memory[i]);
    }

    CHECK(cudaThreadSynchronize());   
    sw.stop(); 
    printf("D2H Time:%f\n", sw.GetTimeInSeconds());
    for(int i=0;i<num_streams;i++)
    {
        cudaEventSynchronize(stop[i]);
    }    
    printf("G for D2H: %f\n",calculate_G(start,stop,num_streams,size)); 
    printf("g for D2H: %f\n",calculate_g(start,stop,num_streams,size));
#endif
   
}
