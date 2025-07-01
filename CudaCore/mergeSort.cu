#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "include/CudaCommon.h"

namespace cg = cooperative_groups;

static inline __host__ __device__ uint32_t iDivUp(uint32_t a, uint32_t b) { return ((a % b) == 0) ? (a / b) : (a / b + 1); }

#define W (sizeof(uint32_t) * 8)
static inline __device__ uint32_t nextPowerOfTwo(uint32_t x)
{
    /*
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    */
    return 1U << (W - __clz(x - 1));
}


template <typename T, uint32_t SORT_DIR> 
static inline __device__ uint32_t binarySearchExclusive(T val, T *data, uint32_t L, uint32_t stride)
{
    if (L == 0) 
    {
        return 0;
    }

    uint32_t pos = 0;
    uint32_t newPos = 0;

    for (; stride > 0; stride >>= 1)
    {
        newPos = umin(pos + stride, L);

        if ((SORT_DIR && data[newPos - 1] < val) || (!SORT_DIR && data[newPos - 1] > val))
        {
            pos = newPos;
        }
    }

    return pos;
}

template <typename T, uint32_t SORT_DIR> 
static inline __device__ uint32_t binarySearchInclusive(T val, T *data, uint32_t L, uint32_t stride)
{
    if (L == 0) 
    {
        return 0;
    }

    uint32_t pos = 0;
    uint32_t newPos = 0;

    for (; stride > 0; stride >>= 1)
    {
        newPos = umin(pos + stride, L);

        if ((SORT_DIR && data[newPos - 1] <= val) || (!SORT_DIR && data[newPos - 1] >= val))
        {
            pos = newPos;
        }
    }

    return pos;
}


/*
* param:
* sortDir    direction of sort
*/
template<uint32_t SORT_DIR, uint32_t BLOCK_SIZE>
__global__ void mergeSortShared(int32_t *d_DstArrayKey, int32_t *d_DstArrayVal, int32_t *d_SrcArrayKey, int32_t *d_SrcArrayVal, uint32_t arrayLength)
{
    cg::thread_block cta = cg::this_thread_block();
    constexpr uint32_t numPerThread = 2;
    constexpr uint32_t blockArrayLength = BLOCK_SIZE * numPerThread;

    __shared__ int32_t s_ArrayKey[BLOCK_SIZE * numPerThread];
    __shared__ int32_t s_ArrayVal[BLOCK_SIZE * numPerThread];

    uint32_t g_Pos = blockIdx.x * blockDim.x * numPerThread + threadIdx.x;

    d_SrcArrayKey += g_Pos;
    d_SrcArrayVal += g_Pos;
    d_DstArrayKey += g_Pos;
    d_DstArrayVal += g_Pos;

    s_ArrayKey[threadIdx.x + 0] = d_SrcArrayKey[0];
    s_ArrayVal[threadIdx.x + 0] = d_SrcArrayVal[0];
    s_ArrayKey[threadIdx.x + BLOCK_SIZE] = d_SrcArrayKey[BLOCK_SIZE];
    s_ArrayVal[threadIdx.x + BLOCK_SIZE] = d_SrcArrayVal[BLOCK_SIZE];


    for (uint32_t stride = 1; stride < blockArrayLength; stride <<= 1)
    {
        uint32_t lPos = threadIdx.x & (stride - 1);
        int32_t *baseKey = s_ArrayKey + numPerThread * (threadIdx.x - lPos);
        int32_t *baseVal = s_ArrayVal + numPerThread * (threadIdx.x - lPos);

        cg::sync(cta);
        int32_t keyA = baseKey[lPos + 0];
        int32_t valA = baseVal[lPos + 0];
        int32_t keyB = baseKey[lPos + stride];
        int32_t valB = baseVal[lPos + stride];
        uint32_t posA = binarySearchExclusive<decltype(keyA), SORT_DIR>(keyA, baseKey + stride, stride, stride) + lPos;
        uint32_t posB = binarySearchInclusive<decltype(keyB), SORT_DIR>(keyB, baseKey + 0, stride, stride) + lPos;
        
        cg::sync(cta);
        baseKey[posA] = keyA;
        baseVal[posA] = valA;
        baseKey[posB] = keyB;
        baseVal[posB] = valB;
    }

    cg::sync(cta);
    d_DstArrayKey[0] = s_ArrayKey[threadIdx.x + 0];
    d_DstArrayVal[0] = s_ArrayVal[threadIdx.x + 0];
    d_DstArrayKey[BLOCK_SIZE] = s_ArrayKey[threadIdx.x + BLOCK_SIZE];
    d_DstArrayVal[BLOCK_SIZE] = s_ArrayVal[threadIdx.x + BLOCK_SIZE];
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 1: generate sample ranks
////////////////////////////////////////////////////////////////////////////////
template <uint32_t SORT_DIR, uint32_t SAMPLE_STRIDE>
__global__ void generateSampleRanksKernel(uint32_t *d_RanksA, 
                                          uint32_t *d_RanksB, 
                                          int32_t  *d_SrcKey, 
                                          uint32_t  stride, 
                                          uint32_t  d_ArrayLength, 
                                          uint32_t  threadCount)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= threadCount)
    {
        return;
    }

    const uint32_t i           = pos & (stride / SAMPLE_STRIDE - 1);
    const uint32_t segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_SrcKey += segmentBase;
    d_RanksA += segmentBase / SAMPLE_STRIDE;
    d_RanksB += segmentBase / SAMPLE_STRIDE;

    const uint32_t segmentElementsA = stride;
    const uint32_t segmentElementsB = umin(stride, d_ArrayLength - segmentBase - stride);
    const uint32_t segmentSamplesA  = iDivUp(segmentElementsA, SAMPLE_STRIDE);
    const uint32_t segmentSamplesB  = iDivUp(segmentElementsB, SAMPLE_STRIDE);

    uint32_t sampleCount = stride / SAMPLE_STRIDE;

    if (i < segmentElementsA)
    {
        d_RanksA[i] = i * SAMPLE_STRIDE;
        d_RanksB[i] = binarySearchExclusive<int32_t, SORT_DIR>(
            d_SrcKey[i * SAMPLE_STRIDE], d_SrcKey + stride, segmentElementsB, nextPowerOfTwo(segmentElementsB));
    }

    if (i < segmentSamplesB)
    {
        d_RanksB[i + sampleCount] = i * SAMPLE_STRIDE;
        d_RanksA[i + sampleCount] = binarySearchInclusive<int32_t, SORT_DIR>(
            d_SrcKey[i * SAMPLE_STRIDE + stride], d_SrcKey + 0, segmentElementsA, nextPowerOfTwo(segmentElementsA));
    }
}

template<uint32_t SAMPLE_STRIDE>
void generateSampleRanks(uint32_t *d_RanksA, uint32_t *d_RanksB, int32_t *ikey, uint32_t stride, uint32_t d_ArrayLength, uint32_t sortDir)
{
    uint32_t lastSegmentElements = d_ArrayLength % (2 * stride);
    uint32_t threadCount = (lastSegmentElements > stride) ? (d_ArrayLength + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE)
                                                      : (d_ArrayLength - lastSegmentElements) / (2 * SAMPLE_STRIDE);

    dim3 blockSize(SAMPLE_STRIDE * 2, 1, 1);
    dim3 gridSize(iDivUp(threadCount, blockSize.x), 1, 1);

    if (sortDir)
    {
        generateSampleRanksKernel<1U, SAMPLE_STRIDE><<<gridSize, blockSize>>>(d_RanksA, d_RanksB, ikey, stride, d_ArrayLength, threadCount);
    }
    else
    {
        generateSampleRanksKernel<0U, SAMPLE_STRIDE><<<gridSize, blockSize>>>(d_RanksA, d_RanksB, ikey, stride, d_ArrayLength, threadCount);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 2: generate sample ranks and indices
////////////////////////////////////////////////////////////////////////////////
template <uint32_t SORT_DIR, uint32_t SAMPLE_STRIDE>
__global__ void mergeRanksAndIndicesKernel(uint32_t *d_Limits, uint32_t *d_Ranks, uint32_t stride, uint32_t d_ArrayLength, uint32_t threadCount)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= threadCount) 
    {
        return;
    }

    const uint32_t i           = pos & (stride / SAMPLE_STRIDE - 1);
    const uint32_t segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_Ranks  += (pos - i) * 2;
    d_Limits += (pos - i) * 2;

    const uint32_t segmentElementsA = stride;
    const uint32_t segmentElementsB = umin(stride, d_ArrayLength - segmentBase - stride);
    const uint32_t segmentSamplesA  = iDivUp(segmentElementsA, SAMPLE_STRIDE);
    const uint32_t segmentSamplesB  = iDivUp(segmentElementsB, SAMPLE_STRIDE);

    if (i < segmentSamplesA)
    {
        uint32_t dstPos = binarySearchExclusive<uint32_t, SORT_DIR>(
            d_Ranks[i], d_Ranks+segmentSamplesA, segmentSamplesB, nextPowerOfTwo(segmentSamplesB)) + i;
        d_Limits[dstPos] = d_Ranks[i];
    }

    if (i < segmentSamplesB)
    {
        uint32_t dstPos = binarySearchInclusive<uint32_t, SORT_DIR>(
            d_Ranks[i + segmentSamplesA], d_Ranks, segmentSamplesA, nextPowerOfTwo(segmentSamplesA)) + i;
        d_Limits[dstPos] = d_Ranks[i + segmentSamplesA];
    }
}

template<uint32_t SAMPLE_STRIDE>
void mergeRanksAndIndices(uint32_t *d_LimitsA, uint32_t *d_LimitsB, uint32_t *d_RanksA, uint32_t *d_RanksB, uint32_t stride, uint32_t d_ArrayLength, uint32_t sortDir)
{
    uint32_t lastSegmentElements = d_ArrayLength % (2 * stride);
    uint32_t threadCount = (lastSegmentElements > stride) ? (d_ArrayLength + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE)
                                                      : (d_ArrayLength - lastSegmentElements) / (2 * SAMPLE_STRIDE);   
    
    dim3 blockSize(SAMPLE_STRIDE * 2, 1, 1);
    dim3 gridSize(iDivUp(threadCount, blockSize.x), 1, 1);

    if (sortDir)
    {
        mergeRanksAndIndicesKernel<1U, SAMPLE_STRIDE><<<gridSize, blockSize>>>(d_LimitsA, d_RanksA, stride, d_ArrayLength, threadCount);

        mergeRanksAndIndicesKernel<1U, SAMPLE_STRIDE><<<gridSize, blockSize>>>(d_LimitsB, d_RanksB, stride, d_ArrayLength, threadCount);
    }
    else
    {
        mergeRanksAndIndicesKernel<0U, SAMPLE_STRIDE><<<gridSize, blockSize>>>(d_LimitsA, d_RanksA, stride, d_ArrayLength, threadCount);

        mergeRanksAndIndicesKernel<0U, SAMPLE_STRIDE><<<gridSize, blockSize>>>(d_LimitsB, d_RanksB, stride, d_ArrayLength, threadCount);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 3: merge elementary intervals
////////////////////////////////////////////////////////////////////////////////
template <uint32_t SORT_DIR>
inline __device__ void merge(int32_t            *dstKey,
                             int32_t            *dstVal,
                             int32_t            *srcAKey,
                             int32_t            *srcAVal,
                             int32_t            *srcBKey,
                             int32_t            *srcBVal,
                             uint32_t            lenA,
                             uint32_t            nPowTwoLenA,
                             uint32_t            lenB,
                             uint32_t            nPowTwoLenB,
                             cg::thread_block    cta)
{
    int32_t keyA, valA, keyB, valB, dstPosA, dstPosB;

    if (threadIdx.x < lenA)
    {
        keyA    = srcAKey[threadIdx.x];
        valA    = srcAVal[threadIdx.x];
        dstPosA = binarySearchExclusive<decltype(keyA), SORT_DIR>(keyA, srcBKey, lenB, nPowTwoLenB) + threadIdx.x;
    }

    if (threadIdx.x < lenB) 
    {
        keyB    = srcBKey[threadIdx.x];
        valB    = srcBVal[threadIdx.x];
        dstPosB = binarySearchInclusive<decltype(keyB), SORT_DIR>(keyB, srcAKey, lenA, nPowTwoLenA) + threadIdx.x;
    }

    cg::sync(cta);

    if (threadIdx.x < lenA) 
    {
        dstKey[dstPosA] = keyA;
        dstVal[dstPosA] = valA;
    }

    if (threadIdx.x < lenB) 
    {
        dstKey[dstPosB] = keyB;
        dstVal[dstPosB] = valB;
    }
}

template <uint32_t SORT_DIR, uint32_t SAMPLE_STRIDE>
__global__ void mergeElementaryIntervalsKernel(int32_t  *d_DstKey,
                                               int32_t  *d_DstVal,
                                               int32_t  *d_SrcKey,
                                               int32_t  *d_SrcVal,
                                               uint32_t *d_LimitsA,
                                               uint32_t *d_LimitsB,
                                               uint32_t  stride,
                                               uint32_t  d_ArrayLength)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ int32_t  s_key[2 * SAMPLE_STRIDE];
    __shared__ int32_t  s_val[2 * SAMPLE_STRIDE];

    const uint32_t intervalI   = blockIdx.x & (2 * stride / SAMPLE_STRIDE - 1);
    const uint32_t segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE;
    d_SrcKey += segmentBase;
    d_SrcVal += segmentBase;
    d_DstKey += segmentBase;
    d_DstVal += segmentBase;

    // Set up threadblock-wide parameters
    __shared__ uint32_t startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

    if (0 == threadIdx.x)
    {
        uint32_t segmentElementsA = stride;
        uint32_t segmentElementsB = umin(stride, d_ArrayLength - segmentBase - stride);
        uint32_t segmentSamplesA  = iDivUp(segmentElementsA, SAMPLE_STRIDE);
        uint32_t segmentSamplesB  = iDivUp(segmentElementsB, SAMPLE_STRIDE);
        uint32_t segmentSamples   = segmentSamplesA + segmentSamplesB;

        startSrcA = d_LimitsA[blockIdx.x];
        startSrcB = d_LimitsB[blockIdx.x];
        uint32_t endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1] : segmentElementsA;
        uint32_t endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1] : segmentElementsB;
        lenSrcA      = endSrcA - startSrcA;
        lenSrcB      = endSrcB - startSrcB;
        startDstA    = startSrcA + startSrcB;
        startDstB    = startDstA + lenSrcA;
    }

    cg::sync(cta);

    if (threadIdx.x < lenSrcA) 
    {
        s_key[threadIdx.x + 0] = d_SrcKey[0 + startSrcA + threadIdx.x];
        s_val[threadIdx.x + 0] = d_SrcVal[0 + startSrcA + threadIdx.x];
    }

    if (threadIdx.x < lenSrcB) 
    {
        s_key[threadIdx.x + SAMPLE_STRIDE] = d_SrcKey[stride + startSrcB + threadIdx.x];
        s_val[threadIdx.x + SAMPLE_STRIDE] = d_SrcVal[stride + startSrcB + threadIdx.x];
    }

    // Merge data in shared memory
    cg::sync(cta);
    merge<SORT_DIR>(s_key,
                    s_val,
                    s_key + 0,
                    s_val + 0,
                    s_key + SAMPLE_STRIDE,
                    s_val + SAMPLE_STRIDE,
                    lenSrcA,
                    SAMPLE_STRIDE,
                    lenSrcB,
                    SAMPLE_STRIDE,
                    cta);

    // Store merged data
    cg::sync(cta);

    if (threadIdx.x < lenSrcA) 
    {
        d_DstKey[startDstA + threadIdx.x] = s_key[threadIdx.x];
        d_DstVal[startDstA + threadIdx.x] = s_val[threadIdx.x];
    }

    if (threadIdx.x < lenSrcB) 
    {
        d_DstKey[startDstB + threadIdx.x] = s_key[lenSrcA + threadIdx.x];
        d_DstVal[startDstB + threadIdx.x] = s_val[lenSrcA + threadIdx.x];
    }

}

template<uint32_t SAMPLE_STRIDE>
void mergeElementaryIntervals(int32_t *d_DstKey,
                              int32_t *d_DstVal, 
                              int32_t *d_SrcKey, 
                              int32_t *d_SrcVal, 
                              uint32_t *d_LimitsA, 
                              uint32_t *d_LimitsB, 
                              uint32_t stride, 
                              uint32_t d_ArrayLength, 
                              uint32_t sortDir)
{
    uint32_t lastSegmentElements = d_ArrayLength % (2 * stride);
    uint32_t mergePairs = (lastSegmentElements > stride) ? iDivUp(d_ArrayLength, SAMPLE_STRIDE) : (d_ArrayLength - lastSegmentElements) / SAMPLE_STRIDE;
    dim3 blockSize(SAMPLE_STRIDE, 1, 1);
    dim3 gridSize(mergePairs, 1, 1);

    if (sortDir) 
    {
        mergeElementaryIntervalsKernel<1U, SAMPLE_STRIDE>
            <<<gridSize, blockSize>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, d_LimitsA, d_LimitsB, stride, d_ArrayLength);
        getLastCudaError("mergeElementaryIntervalsKernel<1> failed\n");
    }
    else 
    {
        mergeElementaryIntervalsKernel<0U, SAMPLE_STRIDE>
            <<<mergePairs, SAMPLE_STRIDE>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, d_LimitsA, d_LimitsB, stride, d_ArrayLength);
    }
}

__global__ void fillValueKernel(int *arr, int value, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) 
    {
        arr[idx] = value;
    }
}

void fillValue(int *arr, int value, int N)
{
    constexpr uint32_t blockSize = 1024;

    fillValueKernel<<<iDivUp(N, blockSize), blockSize>>>(arr, value, N);
}

template<int32_t MAX_VALUE, int32_t MIN_VALUE>
void mergeSort(int32_t *h_DstKey, int32_t *h_DstVal, int32_t *h_SrcKey, int32_t *h_SrcVal, uint32_t h_ArrayLength, uint32_t sortDir)
{
    constexpr uint32_t blockSize  = 512;
    constexpr uint32_t sampleStride = 128;
    constexpr uint32_t sharedSize = blockSize * 2;
    uint32_t  blockCount = iDivUp(h_ArrayLength, sharedSize);
    uint32_t  d_ArrayLength = blockCount * sharedSize;
    const uint32_t ranksCount = iDivUp(d_ArrayLength, sampleStride);

    int32_t *d_SrcKey;
    cudaMalloc(&d_SrcKey, sizeof(int32_t) * d_ArrayLength);
    cudaMemcpy(d_SrcKey, h_SrcKey, sizeof(int32_t) * h_ArrayLength, cudaMemcpyHostToDevice);
    
    if (sortDir)
    {
        fillValue(d_SrcKey + h_ArrayLength, MAX_VALUE, d_ArrayLength - h_ArrayLength);
    }
    else
    {
        fillValue(d_SrcKey + h_ArrayLength, MIN_VALUE, d_ArrayLength - h_ArrayLength);
    }

    int32_t *d_SrcVal;
    cudaMalloc(&d_SrcVal, sizeof(int32_t) * d_ArrayLength);
    cudaMemset(d_SrcVal, 0, sizeof(int32_t) * d_ArrayLength);
    cudaMemcpy(d_SrcVal, h_SrcVal, sizeof(int32_t) * h_ArrayLength, cudaMemcpyHostToDevice);

    int32_t *d_DstKey;
    cudaMalloc(&d_DstKey, sizeof(int32_t) * d_ArrayLength);
    cudaMemset(d_DstKey, 0, sizeof(int32_t) * d_ArrayLength);

    int32_t *d_DstVal;
    cudaMalloc(&d_DstVal, sizeof(int32_t) * d_ArrayLength);
    cudaMemset(d_DstVal, 0, sizeof(int32_t) * d_ArrayLength);

    int32_t *d_BufKey;
    cudaMalloc(&d_BufKey, sizeof(int32_t) * d_ArrayLength);
    cudaMemset(d_BufKey, 0, sizeof(int32_t) * d_ArrayLength);

    int32_t *d_BufVal;
    cudaMalloc(&d_BufVal, sizeof(int32_t) * d_ArrayLength);
    cudaMemset(d_BufVal, 0, sizeof(int32_t) * d_ArrayLength);

    uint32_t *d_RanksA;
    cudaMalloc(&d_RanksA, sizeof(uint32_t) * ranksCount);
    cudaMemset(d_RanksA, 0, sizeof(uint32_t) * ranksCount);

    uint32_t *d_RanksB;
    cudaMalloc(&d_RanksB, sizeof(uint32_t) * ranksCount);
    cudaMemset(d_RanksB, 0, sizeof(uint32_t) * ranksCount);

    uint32_t *d_LimitsA;
    cudaMalloc(&d_LimitsA, sizeof(uint32_t) * ranksCount);
    cudaMemset(d_LimitsA, 0, sizeof(uint32_t) * ranksCount);

    uint32_t *d_LimitsB;
    cudaMalloc(&d_LimitsB, sizeof(uint32_t) * ranksCount);
    cudaMemset(d_LimitsB, 0, sizeof(uint32_t) * ranksCount);

    uint stageCount = 0;

    for (uint stride = sharedSize; stride < d_ArrayLength; stride <<= 1, stageCount++)
    ;

    int32_t *ikey, *ival, *okey, *oval;

    if (stageCount & 1)
    {
        ikey = d_BufKey;
        ival = d_BufVal;
        okey = d_DstKey;
        oval = d_DstVal;
    }
    else
    {
        ikey = d_DstKey;
        ival = d_DstVal;
        okey = d_BufKey;
        oval = d_BufVal;
    }

    // First: each block sorts a portion of data using shared memory
    if (sortDir)
    {
        mergeSortShared<1U, blockSize><<<blockCount, blockSize>>>(ikey, ival, d_SrcKey, d_SrcVal, d_ArrayLength);
    }
    else
    {
        mergeSortShared<0U, blockSize><<<blockCount, blockSize>>>(ikey, ival, d_SrcKey, d_SrcVal, d_ArrayLength);
    }
    

    uint32_t lastSegmentElements = 0;

    // Second: perform global sorting across all blocks
    for (uint32_t stride = sharedSize; stride < d_ArrayLength; stride <<= 1)
    {
        lastSegmentElements = d_ArrayLength % (2 * stride);

        // Find sample ranks and prepare for limiters merge
        generateSampleRanks<sampleStride>(d_RanksA, d_RanksB, ikey, stride, d_ArrayLength, sortDir);

        // Merge ranks and indices
        mergeRanksAndIndices<sampleStride>(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, stride, d_ArrayLength, sortDir);

        // Merge elementary intervals
        mergeElementaryIntervals<sampleStride>(okey, oval, ikey, ival, d_LimitsA, d_LimitsB, stride, d_ArrayLength, sortDir);

        if (lastSegmentElements <= stride) 
        {
            // Last merge segment consists of a single array which just needs to be
            // passed through
            checkCudaErrors(cudaMemcpy(okey + (d_ArrayLength - lastSegmentElements),
                                       ikey + (d_ArrayLength - lastSegmentElements),
                                       lastSegmentElements * sizeof(int32_t),
                                       cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(oval + (d_ArrayLength - lastSegmentElements),
                                       ival + (d_ArrayLength - lastSegmentElements),
                                       lastSegmentElements * sizeof(int32_t),
                                       cudaMemcpyDeviceToDevice));
        }

        int32_t *t;
        t    = ikey;
        ikey = okey;
        okey = t;
        t    = ival;
        ival = oval;
        oval = t;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_DstKey, d_DstKey, sizeof(int32_t) * h_ArrayLength, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_DstVal, d_DstVal, sizeof(int32_t) * h_ArrayLength, cudaMemcpyDeviceToHost);

    if (d_LimitsA) cudaFree(d_LimitsA);
    if (d_LimitsB) cudaFree(d_LimitsB);
    if (d_RanksA)  cudaFree(d_RanksA);
    if (d_RanksB)  cudaFree(d_RanksB);
    if (d_BufKey)  cudaFree(d_BufKey);
    if (d_BufVal)  cudaFree(d_BufVal);
    if (d_DstKey)  cudaFree(d_DstKey);
    if (d_DstVal)  cudaFree(d_DstVal);
    if (d_SrcKey)  cudaFree(d_SrcKey);
    if (d_SrcVal)  cudaFree(d_SrcVal);
}

void mergeSortRun()
{
    constexpr uint32_t sortDir = 1U;
    constexpr uint32_t dataNum = 10771;
    constexpr int32_t  maxVal  = 10000000;
    constexpr int32_t  minVal  = -10000000;

    int32_t *h_SrcKey = new int32_t[dataNum];
    int32_t *h_SrcVal = new int32_t[dataNum];

    for (uint32_t i = 0; i < dataNum; ++i)
    {
        h_SrcKey[i] = dataNum - i;
    }

    for (uint32_t i = 0; i < dataNum; ++i)
    {
        h_SrcVal[i] = i;
    }

    int32_t *h_DstKey = new int32_t[dataNum];
    int32_t *h_DstVal = new int32_t[dataNum];
    memset(h_DstKey, 0, sizeof(int32_t) * dataNum);
    memset(h_DstVal, 0, sizeof(int32_t) * dataNum);

    mergeSort<maxVal, minVal>(h_DstKey, h_DstVal, h_SrcKey, h_SrcVal, dataNum, sortDir);
 
    std::cout << "h_DstKey:" << std::endl;
    for (uint32_t i = 0; i < dataNum; ++i)
    {
        std::cout << h_DstKey[i] << " ";
    }

    std::cout << std::endl;
    std::cout << "h_DstVal:" << std::endl;
    for (uint32_t i = 0; i < dataNum; ++i)
    {
        std::cout << h_DstVal[i] << " ";
    }

    delete[] h_SrcKey;
    delete[] h_SrcVal;
    delete[] h_DstKey;
    delete[] h_DstVal;

    h_SrcKey = nullptr;
    h_SrcVal = nullptr;
    h_DstKey = nullptr;
    h_DstVal = nullptr;
}

int main()
{
    mergeSortRun();

    return 0;
}