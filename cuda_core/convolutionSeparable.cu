#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "include/CudaCommon.h"

namespace cg = cooperative_groups;

template<int32_t KERNEL_RADIUS, int32_t ROWS_BLOCKDIM_Y, int32_t ROWS_BLOCKDIM_X, int32_t ROWS_RESULT_STEPS, int32_t ROWS_HALO_STEPS>
__global__ void convolutionRowsKernel(int32_t *d_DstData, int32_t *d_SrcData, float *d_ConvKernel, int32_t imageH, int32_t imageW, int32_t pitch)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + ROWS_HALO_STEPS * 2 ) *  ROWS_BLOCKDIM_X];
    __shared__ float s_KernelData[2 * KERNEL_RADIUS + 1];

    int32_t posX = blockIdx.x * blockDim.x * ROWS_RESULT_STEPS + threadIdx.x;
    int32_t posY = blockIdx.y * blockDim.y + threadIdx.y;

    if (posX >= imageW || posY >= imageH)
    {
        return;
    }

    int32_t baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * blockDim.x + threadIdx.x;
    int32_t baseY = blockIdx.y * blockDim.y + threadIdx.y;
    d_SrcData += baseY * pitch + baseX;
    d_DstData += baseY * pitch + baseX;

//     // Load main data
// #pragma unroll
//     for (uint32_t i = ROWS_HALO_STEPS; i < ROWS_RESULT_STEPS + ROWS_HALO_STEPS; ++i)
//     {
//         s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? d_SrcData[i * ROWS_BLOCKDIM_X] : 0.f;
//     }

    // Load halo left
#pragma unroll
    for (int32_t i = 0; i < ROWS_HALO_STEPS; ++i)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX > -i * ROWS_BLOCKDIM_X) ? d_SrcData[i * ROWS_BLOCKDIM_X] : 0.f;
    }

    // Load halo right
#pragma unroll
    for (int32_t i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; ++i)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_SrcData[i * ROWS_BLOCKDIM_X] : 0.f;
    }

    // Load convolution kernel
#pragma unroll
    for (int32_t i = threadIdx.y * ROWS_BLOCKDIM_X + threadIdx.x; i < KERNEL_RADIUS + KERNEL_RADIUS + 1; i += ROWS_BLOCKDIM_Y * ROWS_BLOCKDIM_X)
    {
        s_KernelData[i] = d_ConvKernel[i];
    }

    cg::sync(cta);

    // Compute and store results
    for (int32_t i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; ++i)
    {
        if (imageW - baseX > i * ROWS_BLOCKDIM_X)
        {
            float sum = 0.f;

#pragma unroll
            for (int32_t j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; ++j)
            {
                sum += s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j] * s_KernelData[j + KERNEL_RADIUS];
            }

            d_DstData[i * ROWS_BLOCKDIM_X] = static_cast<int32_t>(sum);
        }
    }
}

template<int32_t KERNEL_RADIUS, int32_t COLUMNS_BLOCKDIM_Y, int32_t COLUMNS_BLOCKDIM_X, int32_t COLUMNS_RESULT_STEPS, int32_t COLUMNS_HALO_STEPS>
__global__ void convolutionColumnsKernel(int32_t *d_DstData, int32_t *d_SrcData, float *d_ConvKernel, int32_t imageH, int32_t imageW, int32_t pitch)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS * 2 ) *  COLUMNS_BLOCKDIM_Y + 1];
    __shared__ float s_KernelData[2 * KERNEL_RADIUS + 1];

    int32_t posX = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t posY = blockIdx.y * blockDim.y * COLUMNS_RESULT_STEPS + threadIdx.y;

    if (posX >= imageW || posY >= imageH)
    {
        return;
    }

    int32_t baseX = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * blockDim.y + threadIdx.y;
    d_SrcData += baseY * pitch + baseX;
    d_DstData += baseY * pitch + baseX;

//     // Load main data
// #pragma unroll
//     for (uint32_t i = COLUMNS_HALO_STEPS; i < COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; ++i)
//     {
//         s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? d_SrcData[i * COLUMNS_BLOCKDIM_Y * pitch] : 0.f;
//     }

    // Load halo left
#pragma unroll
    for (int32_t i = 0; i < COLUMNS_HALO_STEPS; ++i)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY > -i * COLUMNS_BLOCKDIM_Y) ? d_SrcData[i * COLUMNS_BLOCKDIM_Y * pitch] : 0.f;
    }

    // Load halo right
#pragma unroll
    for (int32_t i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; ++i)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_SrcData[i * COLUMNS_BLOCKDIM_Y * pitch] : 0.f;
    }

    // Load convolution kernel
#pragma unroll
    for (int32_t i = threadIdx.y * COLUMNS_BLOCKDIM_X + threadIdx.x; i < KERNEL_RADIUS + KERNEL_RADIUS + 1; i += COLUMNS_BLOCKDIM_Y * COLUMNS_BLOCKDIM_X)
    {
        s_KernelData[i] = d_ConvKernel[i];
    }

    cg::sync(cta);

    // Compute and store results
    for (int32_t i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; ++i)
    {
        if (imageH - baseY > i * COLUMNS_BLOCKDIM_Y)
        {
            float sum = 0.f;

#pragma unroll
            for (int32_t j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; ++j)
            {
                sum += s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j] * s_KernelData[j + KERNEL_RADIUS];
            }

            d_DstData[i * COLUMNS_BLOCKDIM_Y * pitch] = static_cast<int32_t>(sum);
        }
    }
}

template<int32_t KERNEL_RADIUS>
void convolutionSeparable(int32_t *h_DstData, int32_t *h_SrcData, float *h_KernelData, int32_t imageH, int32_t imageW)
{
    constexpr int32_t rowsResultSteps    = 8;
    constexpr int32_t columnsResultSteps = 8;
    constexpr int32_t rowsBlockDimX      = 16;
    constexpr int32_t rowsBlockDimY      = 4;
    constexpr int32_t columnsBlockDimX   = 4;
    constexpr int32_t columnsBlockDimY   = 16;
    constexpr int32_t rowsHaloSteps      = iDivUp(KERNEL_RADIUS, rowsBlockDimX);
    constexpr int32_t columnsHaloSteps   = iDivUp(KERNEL_RADIUS, columnsBlockDimY);

    int32_t *d_SrcData  = nullptr;
    cudaMalloc(&d_SrcData, sizeof(int32_t) * imageH * imageW);
    cudaMemcpy(d_SrcData, h_SrcData, sizeof(int32_t) * imageH * imageW, cudaMemcpyHostToDevice);

    int32_t *d_DstData  = nullptr;
    cudaMalloc(&d_DstData, sizeof(int32_t) * imageH * imageW);
    cudaMemset(d_DstData, 0, sizeof(int32_t) * imageH * imageW);

    float *d_KernelData = nullptr;
    cudaMalloc(&d_KernelData, sizeof(float) * (KERNEL_RADIUS + KERNEL_RADIUS + 1));
    cudaMemcpy(d_KernelData, h_KernelData, sizeof(float) * (KERNEL_RADIUS + KERNEL_RADIUS + 1), cudaMemcpyHostToDevice);

    dim3 rowsBlockSize(rowsBlockDimX, rowsBlockDimY);
    dim3 rowsGridSize(iDivUp(imageW, rowsBlockDimX * rowsResultSteps), iDivUp(imageH, rowsBlockDimY));

    dim3 columnsBlockSize(columnsBlockDimX, columnsBlockDimY);
    dim3 columnsGridSize(iDivUp(imageW, columnsBlockDimX), iDivUp(imageH, columnsBlockDimY * columnsResultSteps));

    // rows convolution
    convolutionRowsKernel<KERNEL_RADIUS, rowsBlockDimY, rowsBlockDimX, rowsResultSteps, rowsHaloSteps><<<rowsGridSize, rowsBlockSize>>>
                            (d_DstData, d_SrcData, d_KernelData, imageH, imageW, imageW);

    // columns convolution
    convolutionColumnsKernel<KERNEL_RADIUS, columnsBlockDimY, columnsBlockDimX, columnsResultSteps, columnsHaloSteps><<<columnsGridSize, columnsBlockSize>>>
                            (d_DstData, d_DstData, d_KernelData, imageH, imageW, imageW);

    cudaDeviceSynchronize();

    cudaMemcpy(h_DstData, d_DstData, sizeof(int32_t) * imageH * imageW, cudaMemcpyDeviceToHost);

    if (d_KernelData) cudaFree(d_KernelData);
    if (d_SrcData)    cudaFree(d_SrcData);
    if (d_DstData)    cudaFree(d_DstData);
}

#define PI 3.1415926

void generateGaussianKernel(float *kernelData, int32_t kernelRadius, float sigma)
{
    float sum = 0.f;

    for (int32_t i = -kernelRadius; i <= kernelRadius; ++i)
    {
        kernelData[kernelRadius + i] = expf(-(i * i) / (2 * sigma * sigma)); // / (sqrtf(2 * PI * sigma));
        sum += kernelData[kernelRadius + i];
    }

    for (int32_t i = 0; i < kernelRadius + kernelRadius + 1; ++i)
    {
        kernelData[i] /= sum;
    }
}

void convolutionSeparableRun()
{
    constexpr int32_t kernelRadius = 8;
    constexpr float   sigma        = 1.0;
    constexpr int32_t imageH       = 112;
    constexpr int32_t imageW       = 666;

    int32_t *h_SrcData = new int32_t[imageH * imageW];
    
    for (int32_t i = 0; i < imageH * imageW; ++i)
    {
        h_SrcData[i] = i % 256;
    }

    float  *gaussianKernel = new float[kernelRadius + kernelRadius + 1];
 
    generateGaussianKernel(gaussianKernel, kernelRadius, sigma);

    int32_t *h_DstData = new int32_t[imageH * imageW];
    memset(h_DstData, 0, sizeof(int32_t) * imageH * imageW);

    convolutionSeparable<kernelRadius>(h_DstData, h_SrcData, gaussianKernel, imageH, imageW);

    std::cout << "---- ";
    for (int32_t i = 0; i < imageH; ++i)
    {
        for (int32_t j = 0; j < imageW; ++j)
        {
            std::cout << h_DstData[i * imageW + j] << " ";
        }
        
        std::cout << std::endl; 
        std::cout << "----" << std::endl;
    }

    delete[] gaussianKernel;
    delete[] h_SrcData;
    delete[] h_DstData;

    gaussianKernel = nullptr;
    h_SrcData      = nullptr;
    h_DstData      = nullptr;
}

int main()
{
    convolutionSeparableRun();

    return 0;
}