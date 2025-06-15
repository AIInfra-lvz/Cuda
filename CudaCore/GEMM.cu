#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>


template <int ROW_M_PER_BLOCK, int COL_N_PER_BLOCK, int NUM_K_PER_BLOCK, int ROW_NUM_M_PER_THREADS, int COL_NUM_N_PER_THREADS, int NUM_K_PER_THREADS>
__global__ void SingleGEMMKernel(const float *dGEMMA, const float *dGEMMB, const int M, const int N, const int K, const int paddingA, const int paddingB, 
    const int paddingC, float *dGEMMC)
{
    __shared__ float sharedGEMMA[2][NUM_K_PER_BLOCK][ROW_M_PER_BLOCK * ROW_NUM_M_PER_THREADS];
    __shared__ float sharedGEMMB[2][NUM_K_PER_BLOCK][COL_N_PER_BLOCK * COL_NUM_N_PER_THREADS];

    int tAY = threadIdx.y;
    int tAX = threadIdx.x;
    int tBY = threadIdx.y;
    int tBX = threadIdx.x;
    int numLoopKB = NUM_K_PER_BLOCK / (ROW_M_PER_BLOCK * ROW_NUM_M_PER_THREADS);

    int gIdxA = (blockIdx.y * blockDim.y + threadIdx.y) * ROW_NUM_M_PER_THREADS * paddingA + threadIdx.x * NUM_K_PER_THREADS;
    int gIdxB = threadIdx.y * numLoopKB * paddingB + (blockIdx.x * blockDim.x + threadIdx.x) * COL_NUM_N_PER_THREADS;

    int tARow = tAX * NUM_K_PER_THREADS;
    int tACol = tAY * ROW_NUM_M_PER_THREADS;
    int tBCol = tBX * COL_NUM_N_PER_THREADS;
    int tBRow = tBY * numLoopKB;
    float regA[NUM_K_PER_THREADS] = {0.f};
    float regC[ROW_NUM_M_PER_THREADS][COL_NUM_N_PER_THREADS] = {{0.f}};

    int i = 0;
    int j = 0;
    int k = 0;
    int m = 0;
    int n = 0;
    
    // double buffer
    #pragma unrool
    for (i = 0; i < ROW_NUM_M_PER_THREADS; ++i)
    {
        *(reinterpret_cast<float4*>(&regA[0])) = *(reinterpret_cast<const float4*>(dGEMMA + gIdxA + i * paddingA));
        #pragma unroll
        for (j = 0; j < NUM_K_PER_THREADS; ++j)
        {
            sharedGEMMA[0][tARow + j][tACol + i] = regA[j];
        }
    }

    #pragma unroll
    for (i = 0; i < numLoopKB; ++i)
    {
        *(reinterpret_cast<float2*>(&sharedGEMMB[0][tBRow + i][tBCol])) = *(reinterpret_cast<const float2*>(dGEMMB + gIdxB + i * paddingB));
    }
    
    __syncthreads();
    int regIdx = 1;

    #pragma unroll
    for (k = NUM_K_PER_BLOCK; k < K; k += NUM_K_PER_BLOCK)
    {
        #pragma unrool
        for (i = 0; i < ROW_NUM_M_PER_THREADS; ++i)
        {
            *(reinterpret_cast<float4*>(&regA[0])) = *(reinterpret_cast<const float4*>(dGEMMA + gIdxA + k + i * paddingA));
            #pragma unroll
            for (j = 0; j < NUM_K_PER_THREADS; ++j)
            {
                sharedGEMMA[regIdx][tARow + j][tACol + i] = regA[j];
            }
        }

        #pragma unroll
        for (i = 0; i < numLoopKB; ++i)
        {
            *(reinterpret_cast<float2*>(&sharedGEMMB[regIdx][tBRow + i])) = *(reinterpret_cast<const float2*>(dGEMMB + gIdxB + (k + i) * paddingB));
        }

        regIdx ^= 1;
        #pragma unroll
        for (i = 0; i < NUM_K_PER_BLOCK; ++i)
        {
            #pragma unroll
            for (m = 0; m < ROW_NUM_M_PER_THREADS; ++m)
            {
                #pragma unroll
                for (n = 0; n < COL_NUM_N_PER_THREADS; ++n)
                {
                    regC[m][n] += sharedGEMMA[regIdx][i][tACol + m] * sharedGEMMB[regIdx][i][tBCol + n];
                }
            }
        }

        __syncthreads();
    }

    regIdx ^= 1;
    #pragma unroll
    for (i = 0; i < NUM_K_PER_BLOCK; ++i)
    {
        #pragma unroll
        for (m = 0; m < ROW_NUM_M_PER_THREADS; ++m)
        {
            #pragma unroll
            for (n = 0; n < COL_NUM_N_PER_THREADS; ++n)
            {
                regC[m][n] += sharedGEMMA[regIdx][i][tACol + m] * sharedGEMMB[regIdx][i][tBCol + n];
            }
        }  
    }

    int gIdxC = (blockIdx.y * blockDim.y + threadIdx.y) * ROW_NUM_M_PER_THREADS * paddingC + (blockIdx.x * blockDim.x + threadIdx.x) * COL_NUM_N_PER_THREADS;

    for (i = 0; i < ROW_NUM_M_PER_THREADS; ++i)
    {
        *(reinterpret_cast<float2*>(dGEMMC + gIdxC + i * paddingC)) = *(reinterpret_cast<float2*>(&regC[i][0]));
    }
}

template <int ROW_M_PER_BLOCK, int COL_N_PER_BLOCK, int NUM_K_PER_BLOCK, int ROW_NUM_M_PER_THREADS, int COL_NUM_N_PER_THREADS, int NUM_K_PER_THREADS>
__global__ void SingleGEMMKernelV2(const float *dGEMMA, const float *dGEMMB, const int M, const int N, const int K, const int paddingA, const int paddingB, 
    const int paddingC, float *dGEMMC)
{
    __shared__ float sharedGEMMA[2][NUM_K_PER_BLOCK][ROW_M_PER_BLOCK * ROW_NUM_M_PER_THREADS + 1];
    __shared__ float sharedGEMMB[2][NUM_K_PER_BLOCK][COL_N_PER_BLOCK * COL_NUM_N_PER_THREADS];

    int tARow = threadIdx.y;
    int tACol = threadIdx.x * NUM_K_PER_THREADS;
    int tBRow = (threadIdx.y * blockDim.x + threadIdx.x) * NUM_K_PER_THREADS / (COL_N_PER_BLOCK * COL_NUM_N_PER_THREADS);
    int tBCol = (threadIdx.y * blockDim.x + threadIdx.x) * NUM_K_PER_THREADS % (COL_N_PER_BLOCK * COL_NUM_N_PER_THREADS);

    int gIdxA = (blockIdx.y * blockDim.y * ROW_NUM_M_PER_THREADS + threadIdx.y) * paddingA + threadIdx.x * NUM_K_PER_THREADS;
    int gIdxB =  tBRow * paddingB + (blockIdx.x * blockDim.x) * COL_NUM_N_PER_THREADS + tBCol;

    int stepBRow = blockDim.y * blockDim.x * NUM_K_PER_THREADS / (COL_N_PER_BLOCK * NUM_K_PER_THREADS  );
    int numLoopKB = (NUM_K_PER_BLOCK * COL_NUM_N_PER_THREADS) / (ROW_M_PER_BLOCK * NUM_K_PER_THREADS);

    float regA[NUM_K_PER_THREADS] = {0.f};
    float regC[ROW_NUM_M_PER_THREADS][COL_NUM_N_PER_THREADS] = {{0.f}};
    float regM[ROW_NUM_M_PER_THREADS] = {0.f};
    float regN[COL_NUM_N_PER_THREADS] = {0.f};

    int i = 0;
    int j = 0;
    int k = 0;
    int m = 0;
    int n = 0;
    int regIdx = 1;
    
    // double buffer
    #pragma unrool
    for (i = 0; i < ROW_NUM_M_PER_THREADS; ++i)
    {
        *(reinterpret_cast<float4*>(regA)) = *(reinterpret_cast<const float4*>(dGEMMA + gIdxA + i * blockDim.y * paddingA));
        #pragma unroll
        for (j = 0; j < NUM_K_PER_THREADS; ++j)
        {
            sharedGEMMA[0][tACol + j][tARow + i * blockDim.y] = regA[j];
        }
    }

    #pragma unroll
    for (i = 0; i < numLoopKB; ++i)
    {
        *(reinterpret_cast<float4*>(&sharedGEMMB[0][tBRow + i * stepBRow][tBCol])) = *(reinterpret_cast<const float4*>(dGEMMB + gIdxB + i * stepBRow * paddingB));
    }
    
    __syncthreads();

    #pragma unroll
    for (k = NUM_K_PER_BLOCK; k < K; k += NUM_K_PER_BLOCK)
    {
        #pragma unrool
        for (i = 0; i < ROW_NUM_M_PER_THREADS; ++i)
        {
            *(reinterpret_cast<float4*>(&regA[0])) = *(reinterpret_cast<const float4*>(dGEMMA + gIdxA + k + i * blockDim.y * paddingA));
            #pragma unroll
            for (j = 0; j < NUM_K_PER_THREADS; ++j)
            {
                sharedGEMMA[regIdx][tACol + j][tARow + i * blockDim.y] = regA[j];
            }
        }

        #pragma unroll
        for (i = 0; i < numLoopKB; ++i)
        {
            *(reinterpret_cast<float4*>(&sharedGEMMB[regIdx][tBRow + i * stepBRow][tBCol])) = *(reinterpret_cast<const float4*>(dGEMMB + gIdxB + (k + i * stepBRow) * paddingB));
        }

        regIdx ^= 1;
        #pragma unroll
        for (i = 0; i < NUM_K_PER_BLOCK; ++i)
        {
            *(reinterpret_cast<float2*>(regM)) = *(reinterpret_cast<const float2*>(&sharedGEMMA[regIdx][i][threadIdx.y]));
            *(reinterpret_cast<float2*>(regN)) = *(reinterpret_cast<const float2*>(&sharedGEMMB[regIdx][i][threadIdx.x]));

            #pragma unroll
            for (m = 0; m < ROW_NUM_M_PER_THREADS; ++m)
            {
                #pragma unroll
                for (n = 0; n < COL_NUM_N_PER_THREADS; ++n)
                {
                    regC[m][n] += regM[m] * regN[n];
                }
            }
        }

        __syncthreads();
    }

    regIdx ^= 1;
    #pragma unroll
    for (i = 0; i < NUM_K_PER_BLOCK; ++i)
    {
        *(reinterpret_cast<float2*>(regM)) = *(reinterpret_cast<const float2*>(&sharedGEMMA[regIdx][i][threadIdx.y * ROW_NUM_M_PER_THREADS]));
        *(reinterpret_cast<float2*>(regN)) = *(reinterpret_cast<const float2*>(&sharedGEMMB[regIdx][i][threadIdx.x * COL_NUM_N_PER_THREADS]));

        #pragma unroll
        for (m = 0; m < ROW_NUM_M_PER_THREADS; ++m)
        {
            #pragma unroll
            for (n = 0; n < COL_NUM_N_PER_THREADS; ++n)
            {
                regC[m][n] += regM[m] * regN[n];
            }
        }
    }

    int gIdxC = (blockIdx.y * blockDim.y + threadIdx.y) * ROW_NUM_M_PER_THREADS * paddingC + (blockIdx.x * blockDim.x + threadIdx.x) * COL_NUM_N_PER_THREADS;

    for (i = 0; i < ROW_NUM_M_PER_THREADS; ++i)
    {
        *(reinterpret_cast<float2*>(dGEMMC + gIdxC + i * paddingC)) = *(reinterpret_cast<float2*>(&regC[i][0]));
    }
}

template <int ROW_M_PER_BLOCK, int COL_N_PER_BLOCK, int NUM_K_PER_BLOCK, int ROW_NUM_M_PER_THREADS, int COL_NUM_N_PER_THREADS, int NUM_K_PER_THREADS>
__global__ void SingleGEMMKernelV3(const float *dGEMMA, const float *dGEMMB, const int M, const int N, const int K, const int paddingA, const int paddingB, 
    const int paddingC, float *dGEMMC)
{
    __shared__ float sharedGEMMA[2][NUM_K_PER_BLOCK][ROW_M_PER_BLOCK * ROW_NUM_M_PER_THREADS + 1];
    __shared__ float sharedGEMMB[2][NUM_K_PER_BLOCK][COL_N_PER_BLOCK * COL_NUM_N_PER_THREADS];

    constexpr int numKAPERTHREADS = 8;

    int tARow = (threadIdx.y * blockDim.x + threadIdx.x) / numKAPERTHREADS;
    int tACol = (threadIdx.y * blockDim.x + threadIdx.x) % numKAPERTHREADS * NUM_K_PER_THREADS;
    int tBRow = (threadIdx.y * blockDim.x + threadIdx.x) * NUM_K_PER_THREADS / (COL_N_PER_BLOCK * COL_NUM_N_PER_THREADS);
    int tBCol = (threadIdx.y * blockDim.x + threadIdx.x) * NUM_K_PER_THREADS % (COL_N_PER_BLOCK * COL_NUM_N_PER_THREADS);

    int gIdxA = (blockIdx.y * blockDim.y * ROW_NUM_M_PER_THREADS + tARow) * paddingA + tACol;
    int gIdxB =  tBRow * paddingB + (blockIdx.x * blockDim.x) * COL_NUM_N_PER_THREADS + tBCol;

    int stepBRow  = blockDim.y * blockDim.x * NUM_K_PER_THREADS / (COL_N_PER_BLOCK * NUM_K_PER_THREADS  );
    int numLoopKB = (NUM_K_PER_BLOCK * COL_NUM_N_PER_THREADS) / (ROW_M_PER_BLOCK * NUM_K_PER_THREADS);
    int stepACol  = blockDim.y * blockDim.x * NUM_K_PER_THREADS / numKAPERTHREADS;
    int numLoopKA = ROW_M_PER_BLOCK * ROW_NUM_M_PER_THREADS / NUM_K_PER_THREADS;

    float regA[NUM_K_PER_THREADS] = {0.f};
    float regC[ROW_NUM_M_PER_THREADS][COL_NUM_N_PER_THREADS] = {{0.f}};
    float regM[ROW_NUM_M_PER_THREADS] = {0.f};
    float regN[COL_NUM_N_PER_THREADS] = {0.f};

    int i = 0;
    int j = 0;
    int k = 0;
    int m = 0;
    int n = 0;
    int regIdx = 1;
    
    // double buffer
    #pragma unrool
    for (i = 0; i < numLoopKA; ++i)
    {
        *(reinterpret_cast<float4*>(regA)) = *(reinterpret_cast<const float4*>(dGEMMA + gIdxA + i * stepACol));
        #pragma unroll
        for (j = 0; j < NUM_K_PER_THREADS; ++j)
        {
            sharedGEMMA[0][tACol + j + i * stepACol][tARow] = regA[j];
        }
    }

    #pragma unroll
    for (i = 0; i < numLoopKB; ++i)
    {
        *(reinterpret_cast<float4*>(&sharedGEMMB[0][tBRow + i * stepBRow][tBCol])) = *(reinterpret_cast<const float4*>(dGEMMB + gIdxB + i * stepBRow * paddingB));
    }
    
    __syncthreads();


    #pragma unroll
    for (k = NUM_K_PER_BLOCK; k < K; k += NUM_K_PER_BLOCK)
    {
        #pragma unrool
        for (i = 0; i < numLoopKA; ++i)
        {
            *(reinterpret_cast<float4*>(regA)) = *(reinterpret_cast<const float4*>(dGEMMA + gIdxA + i * stepACol));
            #pragma unroll
            for (j = 0; j < NUM_K_PER_THREADS; ++j)
            {
                sharedGEMMA[0][tACol + j + i * stepACol][tARow] = regA[j];
            }
        }

        #pragma unroll
        for (i = 0; i < numLoopKB; ++i)
        {
            *(reinterpret_cast<float4*>(&sharedGEMMB[regIdx][tBRow + i * stepBRow][tBCol])) = *(reinterpret_cast<const float4*>(dGEMMB + gIdxB + (k + i * stepBRow) * paddingB));
        }

        regIdx ^= 1;
        #pragma unroll
        for (i = 0; i < NUM_K_PER_BLOCK; ++i)
        {
            *(reinterpret_cast<float2*>(regM)) = *(reinterpret_cast<const float2*>(&sharedGEMMA[regIdx][i][threadIdx.y]));
            *(reinterpret_cast<float2*>(regN)) = *(reinterpret_cast<const float2*>(&sharedGEMMB[regIdx][i][threadIdx.x]));

            #pragma unroll
            for (m = 0; m < ROW_NUM_M_PER_THREADS; ++m)
            {
                #pragma unroll
                for (n = 0; n < COL_NUM_N_PER_THREADS; ++n)
                {
                    regC[m][n] += regM[m] * regN[n];
                }
            }
        }

        __syncthreads();
    }

    regIdx ^= 1;
    #pragma unroll
    for (i = 0; i < NUM_K_PER_BLOCK; ++i)
    {
        *(reinterpret_cast<float2*>(regM)) = *(reinterpret_cast<const float2*>(&sharedGEMMA[regIdx][i][threadIdx.y * ROW_NUM_M_PER_THREADS]));
        *(reinterpret_cast<float2*>(regN)) = *(reinterpret_cast<const float2*>(&sharedGEMMB[regIdx][i][threadIdx.x * COL_NUM_N_PER_THREADS]));

        #pragma unroll
        for (m = 0; m < ROW_NUM_M_PER_THREADS; ++m)
        {
            #pragma unroll
            for (n = 0; n < COL_NUM_N_PER_THREADS; ++n)
            {
                regC[m][n] += regM[m] * regN[n];
            }
        }
    }

    int gIdxC = (blockIdx.y * blockDim.y + threadIdx.y) * ROW_NUM_M_PER_THREADS * paddingC + (blockIdx.x * blockDim.x + threadIdx.x) * COL_NUM_N_PER_THREADS;

    for (i = 0; i < ROW_NUM_M_PER_THREADS; ++i)
    {
        *(reinterpret_cast<float2*>(dGEMMC + gIdxC + i * paddingC)) = *(reinterpret_cast<float2*>(&regC[i][0]));
    }
}

void SingleGEMM(const float *hGEMMA, const float *hGEMMB, const int M, const int N, const int K, float *hGEMMC)
{
    constexpr int rowMPerBlock = 4;
    constexpr int colNPerBlock = 16;
    constexpr int rowNumMPerThreads = 2;
    constexpr int colNumNPerThreads = 2;
    constexpr int stepK = 4;
    constexpr int numKPerBlock = colNPerBlock * stepK;

    int numM = (M + rowMPerBlock * rowNumMPerThreads - 1) / (rowMPerBlock * rowNumMPerThreads);
    int numN = (N + colNPerBlock * colNumNPerThreads - 1) / (colNPerBlock * colNumNPerThreads);
    int numK = (K + numKPerBlock - 1) / numKPerBlock;
    int dM = numM * rowMPerBlock * rowNumMPerThreads;
    int dN = numN * colNPerBlock * colNumNPerThreads;
    int dK = numK * numKPerBlock;

    float *dGEMMA = nullptr;
    size_t pitchA = 0;
    cudaMallocPitch((void**)&dGEMMA, &pitchA, dK * sizeof(float), dM);
    cudaMemset2D(dGEMMA, pitchA, 0, dK * sizeof(float), dM);
    cudaMemcpy2D(dGEMMA, pitchA, hGEMMA, K * sizeof(float), K * sizeof(float), M, cudaMemcpyHostToDevice);

    float *dGEMMB = nullptr;
    size_t pitchB = 0;
    cudaMallocPitch((void**)&dGEMMB, &pitchB, dN * sizeof(float), dK);
    cudaMemset2D(dGEMMB, pitchB, 0, dN * sizeof(float), dK);
    cudaMemcpy2D(dGEMMB, pitchB, hGEMMB, N * sizeof(float), N * sizeof(float), K, cudaMemcpyHostToDevice);

    float *dGEMMC = nullptr;
    size_t pitchC = 0;
    cudaMallocPitch((void**)&dGEMMC, &pitchC, dN * sizeof(float), dM);
    cudaMemset2D(dGEMMC, pitchC, 0, dN * sizeof(float), dM);

    int paddingA = pitchA / sizeof(float);
    int paddingB = pitchB / sizeof(float);
    int paddingC = pitchC / sizeof(float);

    dim3 blockSize(colNPerBlock, rowMPerBlock);
    dim3 gridSize(numN, numM);

    // SingleGEMMKernelV2<rowMPerBlock, colNPerBlock, numKPerBlock, rowNumMPerThreads, colNumNPerThreads, stepK><<<gridSize, blockSize>>>(dGEMMA, dGEMMB, 
    //     dM, dN, dK, paddingA, paddingB, paddingC, dGEMMC);
    // cudaDeviceSynchronize();
    // cudaMemset2D(dGEMMC, pitchC, 0, dN * sizeof(float), dM);
    
    SingleGEMMKernelV3<rowMPerBlock, colNPerBlock, numKPerBlock, rowNumMPerThreads, colNumNPerThreads, stepK><<<gridSize, blockSize>>>(dGEMMA, dGEMMB, 
        dM, dN, dK, paddingA, paddingB, paddingC, dGEMMC);
    cudaDeviceSynchronize();
    cudaMemcpy2D(hGEMMC, sizeof(float) * N, dGEMMC, pitchC, N * sizeof(float), M, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M * N; ++i)
    {
        if ((i+1) % N != 0)
        {
            std::cout << " " << hGEMMC[i] << " ";
        }
        else{
            std::cout << " " << hGEMMC[i] << " " << std::endl;
        }
    }

    if (dGEMMA) cudaFree(dGEMMA);
    if (dGEMMB) cudaFree(dGEMMB);
    if (dGEMMC) cudaFree(dGEMMC);
}

void SingleGEMMRun()
{
    int M = 12;
    int N = 16;
    int K = 24;

    float *hGEMMA = new float[M * K];
    float *hGEMMB = new float[N * K];
    float *hGEMMC = new float[M * N];
    memset(hGEMMC, 0, sizeof(float) * M * N);

    for (int i = 0; i < M * K; ++i)
    {
        hGEMMA[i] = i;
    }

    for (int i = 0; i < N * K; ++i)
    {
        hGEMMB[i] = i;
    }

    SingleGEMM(hGEMMA, hGEMMB, M, N, K, hGEMMC);

    delete[] hGEMMA;
    delete[] hGEMMB;
    hGEMMA = nullptr;
    hGEMMB = nullptr;
}


int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 0 是 device ID
    printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);

    SingleGEMMRun();

    return 0;
}