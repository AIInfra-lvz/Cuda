#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "include/CudaCommon.h"

template <int NUM_THREADS_PER_BLOCK, int NUM_PER_THREADS, int WARP_SIZE>
__global__ void cudaReduceSumKernel(const float *dInput, const int inputSize, float *dOutput)
{
    __shared__ float inputSharedMem[NUM_THREADS_PER_BLOCK * NUM_PER_THREADS];
    __shared__ float warpSharedMem[WARP_SIZE];

    int i = 0;
    int tIdx = threadIdx.x;
    int tGlobalIdx = blockIdx.x * blockDim.x * NUM_PER_THREADS + threadIdx.x;
    float sum = 0.f;
    int tId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (i = 0; i < NUM_PER_THREADS; ++i)
    {
        inputSharedMem[tIdx + i * NUM_THREADS_PER_BLOCK] = dInput[tGlobalIdx + i * NUM_THREADS_PER_BLOCK];
    }

    __syncthreads();

    #pragma unroll
    for (i = 0; i < NUM_PER_THREADS; ++i)
    {
        sum +=  inputSharedMem[tIdx + i * NUM_THREADS_PER_BLOCK];
    }

    if (blockDim.x >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (blockDim.x >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (blockDim.x >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (blockDim.x >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (blockDim.x >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (laneId == 0) warpSharedMem[tId] = sum;

    __syncthreads();

    if (tId == 0)
    {
        sum = threadIdx.x < WARP_SIZE ? warpSharedMem[threadIdx.x] : 0.f;
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }

    if (tIdx == 0) atomicAdd(dOutput, sum);
}

template <int NUM_THREADS_PER_BLOCK, int NUM_PER_THREADS, int WARP_SIZE>
__global__ void cudaReduceSumKernelV2(const float *dInput, const int inputSize, float *dOutput)
{
    // __shared__ float inputSharedMem[NUM_THREADS_PER_BLOCK * NUM_PER_THREADS];
    __shared__ float warpSharedMem[WARP_SIZE];

    int i = 0;
    int tIdx = threadIdx.x;
    int tGlobalIdx = blockIdx.x * blockDim.x * NUM_PER_THREADS + threadIdx.x * NUM_PER_THREADS;
    volatile float sum = 0.f;
    int tId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    float input_reg[NUM_PER_THREADS] = {0.f};

    *(reinterpret_cast<float4*>(&input_reg[0])) = *(reinterpret_cast<const float4*>(&dInput[tGlobalIdx]));
    __syncthreads();

    #pragma unroll
    for (i = 0; i < NUM_PER_THREADS; ++i)
    {
        sum += input_reg[i];
    }

    if (blockDim.x >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (blockDim.x >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (blockDim.x >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (blockDim.x >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (blockDim.x >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (laneId == 0) warpSharedMem[tId] = sum;

    __syncthreads();

    if (tId == 0)
    {
        sum = threadIdx.x < WARP_SIZE ? warpSharedMem[threadIdx.x] : 0.f;
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }

    if (tIdx == 0) atomicAdd(dOutput, sum);
}

void cudaReduceSum(const float *hInput, const float inputSize, float *hOutput)
{
    constexpr int numDataPerThreads = 4;
    constexpr int numThreadsPerBlock = 256;
    int gridSizeX = (inputSize + numDataPerThreads * numThreadsPerBlock - 1) / (numDataPerThreads * numThreadsPerBlock);
    int numDeviceInputSize = gridSizeX * numDataPerThreads * numThreadsPerBlock;

    float *dInput = nullptr;
    checkCudaErrors(cudaMalloc((void**)&dInput, sizeof(float) * numDeviceInputSize));
    checkCudaErrors(cudaMemset(dInput, 0, sizeof(float) * numDeviceInputSize));
    checkCudaErrors(cudaMemcpy(dInput, hInput, sizeof(float) * inputSize, cudaMemcpyHostToDevice));
    
    float *dOutput = nullptr;
    checkCudaErrors(cudaMalloc((void**)&dOutput, sizeof(float)));
    checkCudaErrors(cudaMemset(dOutput, 0, sizeof(float)));

    dim3 blockSize(numThreadsPerBlock);
    dim3 gridSize(gridSizeX);
    cudaReduceSumKernel<numThreadsPerBlock, numDataPerThreads, 32><<<gridSize, blockSize>>>(dInput, numDeviceInputSize, dOutput);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(hOutput, dOutput, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "cudaReduceSum = " << *hOutput << std::endl;

    checkCudaErrors(cudaMemset(dOutput, 0, sizeof(float)));
    checkCudaErrors(cudaMemset(dInput, 0, sizeof(float) * numDeviceInputSize));
    checkCudaErrors(cudaMemcpy(dInput, hInput, sizeof(float) * inputSize, cudaMemcpyHostToDevice));
    cudaReduceSumKernelV2<numThreadsPerBlock, numDataPerThreads, 32><<<gridSize, blockSize>>>(dInput, numDeviceInputSize, dOutput);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(hOutput, dOutput, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "cudaReduceSumKernelV2 = " << *hOutput << std::endl;

    if (dInput) checkCudaErrors(cudaFree(dInput));
    if (dOutput) checkCudaErrors(cudaFree(dOutput));
}

void cudaReduceSumRun()
{
    constexpr int numData = 10035;
    float *input = new float[numData];
    for (int i = 0; i < numData; ++i)
    {
        input[i] = i;
    }

    float output = 0.f;

    cudaReduceSum(input, numData, &output);

    std::cout << "cudaReduceSum = " << output << std::endl;
    delete[] input;
    input = nullptr;

}

int main()
{
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0)); // 0 是 device ID
    printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    cudaReduceSumRun();

    return 0;
}