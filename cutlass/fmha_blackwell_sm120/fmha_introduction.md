
> **File:** fmha_introduction.md  
> **Author:** AIInfra-lvz
> **Description:** A universal FMHA kernels for CUTLASS 4.x-style   
> **Reference:** [cutlass/examples/blackwell_fmha](https://github.com/cutlass/cutlass)

# Preface
The article aims to provide a detailed explanation of the FMHA kernels in CUTLASS 3.x, with the kernels originating from the CUTLASS examples in blackwell_fmha.

The documentation of CUTLASS shows that the GEMM kernel API consists of five layers, from top to bottom: device, kernel, collective, tiled MMA and copy, and atom. Instead of analyzing each layer separately, we will group the last three layers together for analysis, while also considering the first two layers. Meanwhile, the usage of Cute may be involved in the collective layer. 

Additionally, as this article serves as learning notes for CUTLASS, there may be some unreasonable explanations. You are welcome to critique and correct any mistakes.

# Overview
There are two main sections: the first is about FMHA mathematical theory, and the second is about the program execution pipeline. Introducing these two sections first helps to understand the overall logic of FMHA, which in turn aids in grasping the detailed design.
## Mathematical Theory
The attention module in the transformer architecture has several variants: MHA, GQA, VL-MHA, Gen-MHA, and MHLA. The FMHA being called simply implements these principles using a kernel. The documentation currently only introduces the first two; the others will be improved in the future.
### MHA
MHA is the most basic form of attention, and its theory is described below.
$$
\mathrm{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V \\\\
\mathrm{softmax}(x_i) = \frac{2^{\log_2 e \cdot (x_i - \max_j x_j)}}{\sum_k 2^{\log_2 e \cdot (x_k - \max_j x_j)}} 
$$
Obviously, there are three main steps: multiplying matrices Q and K, computing the softmax, and multiplying the result of the first two steps by matrix V. The matrix Q has the same number of heads as matrices K and V.
### GQA
The basic computation of GQA remains unchanged, but a major difference is that matrices K and V have only one head for several heads of matrix Q, so the number of heads in K and V corresponds to the number of groups in Q.
$$
\mathrm{softmax}\left(\frac{Q_i K_{g(i)}^T}{\sqrt{d}}\right) V_{g(i)}
$$
Here, $Q_i$ denotes the $i$-th query, and $K_{g(i)}$ and $V_{g(i)}$ denote the key and value that belong to the same group as $Q_i$. Softmax computation is the same as in MHA.
## Pipeline
As shown in the picture below, the basic computing steps of FMHA are illustrated in a simple flow chart. The complex details will be discussed in the kernel and collective sections.   
![Compute Pipeline](./src_pictures/pipeline.png)    
The flow chart illustrates the flow of each fragmented computation within a block of the CUDA kernel. Each node consists of two parts: stage and detail. The stages include Load, MMA, Softmax, Correction, and Epilogue. These stages represent the organizational structure in the post-kernel layer. The details show the specific events that are carried out in the collective layer. In addition, the side-by-side nodes can be seen as being executed in parallel, and the overlap of multi-stage memory access and computation is not shown.

# Device Layer
Firstly, we will discuss the top layer of CUTLASS: device. The device layer describes the host code responsible for two main tasks: the conversion of outer arguments to inner parameters and the launch methods of CUDA kernels. It serves as the entry point for running the kernel. We will not explain every sentence or line of code, so you may need to read some coding details on your own in fmha_blackwell_sm120/device/fmha.hpp.

The template parameter of device layer class is the kernel layer template class, which provides the methods for the argument conversion, shared memory size, grid shape and other global memory size if needed.
## Task 1
The part directly uses the methods of the kernel layer to convert the arguments to parameters, which is done automically in initialize function. The function will also automically check whether the shared memory size is reasonable and attempt to adjust it if necessary.
```
Status initialize(const Arguments& args, void* workspace = nullptr, cudaStream_t stream = nullptr)
```
Note: This function is non-static, which is different from many other functions.
## Task 2
The part needs to overload several run functions, but only one function acutually executes kernel, while the others call it in internally. These run functions mainly cover scenarios such as unchanged arguments, changed arguments and provide advanced method to run kernel when a non-class object is initialized. Therefore, the run function need to be static.
```
// static methods
static Status run(Params& params, cudaStream_t stream = nullptr)
// changed arguments
Status run(const Arguments& args, void* workspace = nullptr, cudaStream_t stream = nullptr) 
// unchanged arguments
Status run(cudaStream_t stream = nullptr) 
```
Additionly, we also need to overload the opertion ().
```
// changed arguments
Status operator()(const Arguments& args, void* workspace = nullptr, cudaStream_t stream = nullptr) 
{
    return run(args, workspace, stream);
}
// unchanged arguments
Status operator()(cudaStream_t stream = nullptr) 
{
    return run(params_, stream);
}
```
The run function provides two methods to lunch the kernel: the generic way and cluster launch.
```
if constexpr(Kernel::ArchTag::kMinComputeCapability >= 90) 
{
    dim3 cluster(cute::size<0>(typename Kernel::ClusterShape{}),
                cute::size<1>(typename Kernel::ClusterShape{}),
                cute::size<2>(typename Kernel::ClusterShape{}));
    void const* kernel = (void const*) device_kernel<Kernel>;
    void* kernel_params[] = {&params};
    launch_result = ClusterLauncher::launch(grid, cluster, block, smem_size, stream, kernel, kernel_params);
}
else 
{
    launch_result = Status::kSuccess;
    device_kernel<Kernel><<<grid, block, smem_size, stream>>>(params);
}
```
### Cluster
Cluster launch is supported in Hopper and Blackwell architectures, and one of its abilities is to provide a multi-broadcast function. Multi-broadcast can broadcast data to different blocks within the same cluster when those blocks need the same tiled data, which can significantly reduce memory access time.
```
// Shape of the threadblocks in a cluster
using ClusterShape = Shape<_4,_2,_1>;
```
Note:
X indicates the number of blocks along the M dimension to broadcast when executing GEMM. Y indicates the number of blocks along the N dimension, and Z indicates the number of blocks along the K
dimension.

# Kernel Layer
There are also two main sections in the kernel layer: pipeline organization and tile scheduler. The computation pipeline for one tile within a block has already been illustrated in the Overview section, so the following pipeline provides more details about the pipelines.
## Pipeline Origanization
Before introducing pipeline organization, we should learn about three important things: first, the Tensor Memory Accelerator (TMA); second, Tensor Memory (TMEM); and third, warp-specialized kernel schedules. These are helpful for us to grasp the pipeline parameters design and register allocation within pipeline organization.
### Tensor Memory Accelerator (TMA)
A [blog](https://research.colfax-intl.com/tutorial-hopper-tma/) is recommended for a detailed understanding of TMA. This section refers to that blog.
TMA is a new feature introduced in the NVIDIA Hopper™ architecture for doing asynchronous memory copy between a GPU’s global memory (GMEM) and the shared memory (SMEM) of its threadblocks (i.e., CTAs). Therefore, it is only useful on architectures with compute capability 90a or higher. Instead of implementing TMA, this section focuses on understanding what TMA is and its usage limitations. The usage of TMA will be demonstrated in the load stage of the collective layer with code examples.    

We divide the TMA section into three main parts: the first covers TMA load, the second covers TMA store, and the third discusses more advanced operations such as TMA store reduce and TMA load multicast. In essence, TMA load copies data from the GPU’s GMEM into a CTA’s SMEM, while TMA store copies data from a CTA’s SMEM to the GPU’s GMEM. We will introduce most of the necessary concepts about TMA. 
#### TMA Load/Store
TMA load copies data from GMEM into SMEM, while TMA store does the opposite. This copy operation is limited in that only one thread is responsible for issuing the operation, as shown in the code snippet below from the file fmha_blackwell_sm120/collective/sm120_fmha_load_tma_warpspecialized.hpp.
```
uint32_t lane_predicate = cute::elect_one_sync();       // select only one thread, usually the 0th thread.
if (lane_predicate)
{
    auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
    copy(params.tma_load_q.with(*tma_barrier, 0), tQgQ(_, q0_index), tQsQ(_, pipeline_q_producer_state.index()));
}
```
As shown in the corresponding PTX code snippet below from the file cute/arch/copy_sm90_tma.hpp.
```
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    asm volatile (
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "l"(cache_hint)
      : "memory");
#else
    asm volatile (
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "l"(cache_hint)
      : "memory");
#endif
```
However, all the threads within the CTA should wait for the copy operation to complete, as implemented in the PTX code snippet below from the file cutlass/arch/barrier.h.
```
  CUTLASS_HOST_DEVICE
  static void wait(ValueType const* smem_ptr, uint32_t phase) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_cluster_barrier_wait(__LINE__, smem_addr, phase);
    // Arbitrarily large timer value after which try-wait expires and re-tries.
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
        :
        : "r"(smem_addr), "r"(phase), "r"(ticks));

#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
  }
```
Actually, instead of using this API directly, we call higher-level instructions such as the `consumer_wait` API and similar functions provided by the pipeline object from the file cutlass/pipeline/sm_90_pipeline.hpp.      
In addition, compared to TMA load, TMA store has some differences.
```
if (lane_predicate)
{
    copy(params.tma_store_o, tOsO(_,_,_,_1{}), tOgO(_,_,_,o1_index));
}

tma_store_arrive();

tma_store_wait<0>();  
```
`tma_store_arrive()` commits the TMA store operation (technically, as a cp.async.bulk-group), and `tma_store_wait<Count>()` waits until at most Count of the committed TMA store operations are still pending (e.g., if all should be completed, set Count to 0).   
#### TMA Load Multicast
Multicast refers to a situation where we have a tile in a GMEM tensor that we want to copy to multiple SMEM locations in multiple CTAs. This is typically the case in GEMM kernels (i.e., matrix multiplication), where an input matrix column tile is needed for multiple row tiles or vice versa. In such cases, while TMA load is still perfectly functional — we simply provide the same TMA descriptor to the multiple CTAs that need it — the .multicast operand allows us to guarantee L2-cache hits.
```
uint32_t lane_predicate = cute::elect_one_sync();       // select only one thread, usually the 0th thread.
if (lane_predicate)
{
    auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
    copy(params.tma_load_q.with(*tma_barrier, 0), tQgQ(_, q0_index), tQsQ(_, pipeline_q_producer_state.index()));
}
```
If a multicast operation is needed, you can replace the parameter 0 in `params.tma_load_q.with(*tma_barrier, 0)` with a uint16 number as a bitmask. This bitmask specifies which CTAs will participate in the TMA multicast load: each bit set to 1 indicates an active CTA. There can be up to 16 CTAs in a cluster (the maximum non-portable size), and the position of each bit corresponds to the CTA ID in cluster.    
More details about TMA multicast and TMA store reduce are not explained here. They can be found in this [blog](https://research.colfax-intl.com/tutorial-hopper-tma/).
### Tensor Memory (TMEM)
A [blog](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) and [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory-addressing) are recommended for a detailed understanding of TMEM. This section refers to both of them.  

TMEM is introduced by the NVIDIA Blackwell architecture (SM100) and is a dedicated on-chip memory for use by Tensor Cores. Its primary purpose is to replace registers for 5th generation Tensor Core operations by using TMEM instead. The MMA instruction for TMEM is UMMA, which differs from the WGMMA instruction in that it supports low-precision data types, including FP4 and FP6, provides increased throughput across all precisions, and can only be launched by one thread in a CTA. Two adjacent CTAs within an SM cluster, called a CTA pair, can work on UMMA together across two SMs. Even when using two CTAs, only one thread in one CTA launches UMMA.  Obviously, TMEM solves the problem of high register usage when implementing GEMM.  

TMEM is 256KB per SM in size, and is organized 2-dimensionally in 512 columns and 128 rows, or lanes, of 32-bit cells. This inherent 2-D structure is reflected in the 32-bit addresses as well, where bits 31-16 denote the lane ID while 15-0 denote the column. This image from the PTX documentation shows the layout:
![TMEM_layout](./src_pictures/TMEM_Layout.png)
TMEM is allocated dynamically using the tcgen05.alloc instruction. Furthermore, allocation is in units of columns, so in particular every lane of a column is allocated when a column is allocated. The number of columns allocated must be a power of 2 and at least 32. Finally, TMEM must be explicitly deallocated with tcgen05.dealloc. Both tcgen05.alloc and tcgen05.dealloc must be called from a single warp, and the same warp as far as possible both allocate and deallocate. Generally, we use the high-level to allocate or deallocate memory provided by cutlass.
```
// TMEM object
TmemAllocator tmem_allocator;
// allocate TMEM
tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
//deallocate TMEM
tmem_allocator.free(shared_storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
```
Notice that `tmem_allocator.allocate` stores the base 32-bit address of the allocation to a given location in shared memory and we need to select a single fully active warp of the CTA to use `tmem_allocator.allocate` and `tmem_allocator.free`.
```
  CUTE_HOST_DEVICE void
  allocate(int num_columns, uint32_t* dst_ptr) {
  #if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    uint32_t dst_intptr = cute::cast_smem_ptr_to_uint(dst_ptr);
    asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
      :
      : "r"(dst_intptr), "r"(num_columns));
  #else
    CUTE_INVALID_CONTROL_PATH("Attempting to use TMEM allocation PTX without CUTE_ARCH_TCGEN05_TMEM_ENABLED");
  #endif
  }

  __device__
  void
  free(uint32_t tmem_ptr, int num_columns) {
  #if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile(
      "{\n\t"
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32  %0, %1; \n\t"
      "}"
      :
      : "r"(tmem_ptr), "r"(num_columns));
  #else
    CUTE_INVALID_CONTROL_PATH("Attempting to use TMEM allocation PTX without CUTE_ARCH_TCGEN05_TMEM_ENABLED");
  #endif
  }
```
Typically, data gets into TMEM via UMMA operations, and is explicitly moved out to registers using tcgen05.ld for post-processing. It’s also possible for threads to manually load data into TMEM, either from SMEM through tcgen05.cp or from registers through tcgen05.st. However, TMEM access patterns for explicit load and store are very restricted. Each warp within a warpgroup can only access 32 lanes (with warp 0 associated to lanes 0-31, warp 1 to lanes 32-63, and so forth). 

Finally, besides UMMA operations and these data movement instructions, no other operations access data from TMEM. In other words, all pre-processing must happen before the data is loaded onto TMEM, and all post-processing must happen after the data is retrieved out of TMEM.
#### UMMA
UMMA only use TMEM and its low-level is `tcgen05.mma` operation. From the [table of supported matrix shapes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-kind-shapes), we see that MMA instructions are available in shapes 64 x N x 16 with N a multiple of 8 and 128 x N x 16 with N a multiple of 16, where in both cases N is at most 256. (For all data types, K is expected to be 32 bytes wide for dense GEMM.) Note that the largest UMMA atom, 128 x 256 x 16, is twice as large as the largest WGMMA atom. Its accumulator takes up exactly half of TMEM, meaning that several UMMA atoms can be pipelined without sacrificing performance. 
In addition, for executing D = A × B + D, UMMA supports limited matrix multiplication in two situations: (1) both matrices A and B use SMEM, and (2) matrix A uses TMEM while matrix B uses SMEM. In both cases, matrix D must use TMEM. These cases can be described by the instructions SM100_MMA_F16BF16_SS and SM100_MMA_F16BF16_TS, for example:
```
// both of SMEM
TiledMMA tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC,                 
                                                         128, 256,
                                                         UMMA::Major::K, 
                                                         UMMA::Major::K>{});

// A is TMEM, B is SMEM
TiledMMA tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_TS<TypeA, TypeB, TypeC,                 
                                                         128, 256,
                                                         UMMA::Major::K, 
                                                         UMMA::Major::K>{});
```
Notice again that UMMA can be launched by only one thread in a CTA.
#### Copy out of TMEM
Once all the MMAs are done, we need to copy the accumulator results from TMEM to registers. This is done using the PTX tcgen05.ld instruction. CUTLASS abstracts tcgen05.ld as a copy atom, with different variants we saw earlier represented as different copy traits defined in copy atoms found in cute/atom/copy_traits_sm100.hpp. For example, the `SM100_TMEM_LOAD_32dp32b1x` atom describes these details: `32dp` indicates the fully active threads in a warp, `32b` denotes 32 bits for each element, and `1x` indicates the repetition count. More instruction types can be found in the file cute/arch/copy_sm100.hpp. The code snippet below shows the specific definition of the `SM100_TMEM_LOAD_32dp32b1x` atom as an example.
```
// 32 data path lanes, 32-bit pattern, repeated 1 times
struct SM100_TMEM_LOAD_32dp32b1x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];
 
  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    "{%0},"
                    "[%1];\n"
    :  "=r"(dst0)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};
```
Generally, we use a specialized function, `make_tmem_copy`, to deduce a TV-layout from the copy atom and TMEM tensor and create the TiledCopy. 
```
// Create the tiled copy operation for the accumulator (TMEM -> RMEM)
TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
// Get the copy operation for the accumulator assigned to the current thread
ThrCopy   thr_t2r_copy   = tiled_t2r_copy.get_slice(threadIdx.x);
```
One important thing to know about `make_tmem_copy` function is that it is hardcoded to use 4 warps, or 1 warpgroup. As shown in PTX code snippet from the file cute/atom/copy_traits_sm100.hpp.
```
template <class CopyOp, class CopyT,
          class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
auto
make_tmem_copy(Copy_Atom<CopyOp,CopyT> const& atom,
               Tensor<TEngine,TLayout> const& tmem)
{
  static_assert(is_tmem<TEngine>::value, "Expected TMEM tensor.");
  using T      = typename TEngine::value_type;
  using Traits = typename Copy_Atom<CopyOp, CopyT>::Traits;
  static_assert(sizeof_bits_v<CopyT> == sizeof_bits_v<T>,
                "Expected a CopyAtom with the same type-width as the Tensor.");

  // atom thr idx -> tmem addr    4warps where each warp points to the same position within it's own subpartition
  auto atom_t_layout = Layout<Shape<_32,_4>, Stride<_0, decltype(Int<32>{} * TMEM::DP<T>{})>>{};
  // atom val idx -> tmem addr    Cast the CopyOp's value ids to the proper data width
  auto atom_v_layout = coalesce(upcast<sizeof_bits<T>::value>(typename Traits::ValID{}));

  return make_cotiled_copy(atom, make_layout(atom_t_layout, atom_v_layout), tmem.layout());
}
``` 
The `atom_t_layout` in line 312 shows that its shape is `Shape<_32,_4>`, which denotes 32 threads in a warp and 4 warps in a warpgroup.  

As mentioned in the earlier section, certain regions of TMEM are only accessible by a corresponding warp in a warpgroup, based on the warp index mod 4. This [diagram from the PTX manual](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions) shows how the data is assigned to warps, and you can find more details there if needed.    
![TMEM_data_assigned](./src_pictures/TMEM_data_assigned.png)
The pictures below show the TMEM addresses that this maps to.   
![TMEM_address](./src_pictures/TMEM_address.png)
### Warp-Specialized Kernel Schedules
In a block of FMHA, a total of 16 warps are each assigned a specific role, such as Load, MMA, Softmax0, Softmax1, Correction, Epilogue, or Empty. Except for Empty, which is a reserved warp for future use, the other warps carry out specific tasks.
```
struct Sm120FmhaCtxKernelWarpspecializedSchedule
  {

    enum class WarpRole
    {
      Softmax0,
      Softmax1,
      Correction,
      MMA,
      Load,
      Epilogue,
      Empty
    };

    static constexpr WarpRole warp_idx_to_WarpRole(int warp_idx)
    {
      int wg_idx = warp_idx / 4;                        // warp_idx
      if (wg_idx == 0) return WarpRole::Softmax0;       //   0 -  3
      if (wg_idx == 1) return WarpRole::Softmax1;       //   4 -  7
      if (wg_idx == 2) return WarpRole::Correction;     //   8 - 11
      if (warp_idx == 12) return WarpRole::MMA;         //       12
      if (warp_idx == 13) return WarpRole::Load;        //       13
      if (warp_idx == 14) return WarpRole::Epilogue;    //       14
      return WarpRole::Empty;                           //       15
    }

    static const int NumWarpsSoftmax = 4;
    static const int NumWarpsCorrection = 4;
    static const int NumWarpsEpilogue = 1;
    static const int NumWarpsLoad = 1;

    static const bool kDebugUsingPrintf = false;
    static const int NumRegsSoftmax = 192;
    static const int NumRegsCorrection = 96 - (kDebugUsingPrintf ? 16 : 0);
    static const int NumRegsOther = 32 + (kDebugUsingPrintf ? 16 : 0);
    static const int NumRegsEmpty = 24;

    static const int NumWarps = 16;
  };
```
`WarpRole` is not discussed further here; please refer to the Overview Pipeline section for more information. We will focus on explaining the number of warps assigned to each warp role and their corresponding register counts.   
#### Warp Assignment
As the code snippet shows, the MMA, Load, and Epilogue operations each use only one warp. This is because these three operations perform TMA Load, UMMA, and TMA Store, as described in the previous chapter. They are synchronized and must be launched by a single thread in the CTA, so assigning one warp is sufficient. In contrast, the Softmax and Correction operations are post-processing steps for TMEM. It is known that any post-processing operation on TMEM requires 4 warps, or a warp group, which is hardcoded in the `make_tmem_copy` instruction.
#### Register Assignment
Before discussing register assignment, it is helpful to understand the [Miscellaneous Instructions: setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=setmaxnreg%2520dec#miscellaneous-instructions-setmaxnreg). The detailed usage in the high-level CUTLASS API is shown below:
```
// allocate resgisters
cutlass::arch::warpgroup_reg_alloc<RegCount>();
// deallocate registers
cutlass::arch::warpgroup_reg_dealloc<RegCount>();
```
The corresponding PTX instructions are as follows:
```
template<uint32_t RegCount>
CUTLASS_DEVICE
void warpgroup_reg_alloc(){
#if CUDA_CTA_RECONFIG_ACTIVATED
  asm volatile( "setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount) );
#endif
}

template<uint32_t RegCount>
CUTLASS_DEVICE
void warpgroup_reg_dealloc(){
#if CUDA_CTA_RECONFIG_ACTIVATED
  asm volatile( "setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount) );
#endif
}
```
The following requirements are from the [Miscellaneous Instructions: setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=setmaxnreg%2520dec#miscellaneous-instructions-setmaxnreg):   
(1) The `.dec` qualifier releases extra registers, reducing the absolute per-thread maximum register count from its current value to `RegCount`. The `.inc` qualifier requests additional registers, increasing the absolute per-thread maximum register count from its current value to `RegCount`.    
(2) A pool of available registers is maintained per-CTA. Register adjustments requested by `setmaxnreg` instructions are handled by supplying extra registers from this pool to the requesting warp, or by releasing extra registers from the requesting warp back to this pool.    
(3) The `setmaxnreg.inc` instruction blocks execution until enough registers are available in the CTA’s register pool. After `setmaxnreg.inc` obtains new registers from the CTA pool, the initial contents of these registers are undefined and must be initialized before use.    
(4) The operand `RegCount` is an integer constant. Its value must be in the range 24 to 256 (inclusive) and must be a multiple of 8.    
(5) When the `.dec` qualifier is specified, the maximum number of per-thread registers owned by the warp prior to executing the `setmaxnreg` instruction must be greater than or equal to `RegCount`; otherwise, the behavior is undefined. When the `.inc` qualifier is specified, the maximum number of per-thread registers owned by the warp prior to executing the `setmaxnreg` instruction must be less than or equal to `RegCount`; otherwise, the behavior is undefined.    
(6) The mandatory `.sync` qualifier indicates that the `setmaxnreg` instruction causes the executing thread to wait until all threads in the warp execute the same `setmaxnreg` instruction before resuming execution.    
(7) The mandatory `.aligned` qualifier indicates that all threads in the warpgroup must execute the same `setmaxnreg` instruction. In conditionally executed code, the `setmaxnreg` instruction should only be used if it is known that all threads in the warpgroup evaluate the condition identically; otherwise, the behavior is undefined.    

Except for additional debug printf registers, `NumRegsOther` refers to the MMA, Load, and Epilogue operations. These operations mainly use SMEM and TMEM, so they require only a few registers to store several variables, rather than large tile tensor data. Requirements (4), (6), and (7) above explain why `NumRegsEmpty` is set to 24, and why MMA, Load, Epilogue, and Empty all belong to the same warpgroup and maintain a maximum value less than 255 per thread in the CTA.    
Softmax and Correction require more registers because both are post-processing steps for TMEM. They need to copy data from TMEM to registers for additional computation. The specific register values will be demonstrated in the Collective Layer Softmax and Correction section.  

Additionally, as an additional point, we use the high-level API `warpgroup_reg_set` to set the number of registers per thread. The API details are as follows:
```
template<uint32_t RegCount>
CUTLASS_DEVICE void 
warpgroup_reg_set()
{
  if constexpr (RegCount < 128)
  {
    cutlass::arch::warpgroup_reg_dealloc<RegCount>();
  }
  else
  {
    cutlass::arch::warpgroup_reg_alloc<RegCount>();
  }
}
```
As the code snippet above shows, the constant value 128 is used to determine whether to allocate or deallocate registers. Why 128? Consider that the maximum number of registers per-CTA is 65536, and there are 16 warps in a CTA. Therefore, the average number of registers per thread is 128.   
### Pipeline Creation
From the above specifications, it is easy to understand why different warp roles are defined and how to set the corresponding parameters. Now, let's return to the Kernel Layer to see what needs to be done. In fact, the file fmha_blackwell_sm120/kernel/sm120_fmha_fwd_kernel_tma_warpsspecialized.hpp clearly shows that the main task of the kernel layer within FMHA is to create a pipeline variable and establish a producer-consumer mechanism between different warp roles. For example:
```
// between LoadQ and UMMA
typename CollectiveMainloop::PipelineQ::Params pipeline_load_q_params;
if (role == KernelSchedule::WarpRole::Load)
{
  pipeline_load_q_params.role = CollectiveMainloop::PipelineQ::ThreadCategory::Producer;
}
if (role == KernelSchedule::WarpRole::MMA)
{
  pipeline_load_q_params.role = CollectiveMainloop::PipelineQ::ThreadCategory::Consumer;
}
pipeline_load_q_params.is_leader         = lane_predicate && (role == KernelSchedule::WarpRole::Load);
pipeline_load_q_params.transaction_bytes = CollectiveMainloop::TransactionBytesLoadQ;
typename CollectiveMainloop::PipelineQ pipeline_load_q(
  shared_storage.pipelines.load_q,
  pipeline_load_q_params,
  ClusterShape{},  cute::true_type{}, /*mask calc*/cute::false_type{}
);
```
```
// between Softmax and Correction
typename CollectiveMainloop::PipelineC::Params pipeline_s0_corr_params;
if (role == KernelSchedule::WarpRole::Softmax0)
{
  pipeline_s0_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Producer;
}
if (role == KernelSchedule::WarpRole::Correction)
{
  pipeline_s0_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Consumer;
}
pipeline_s0_corr_params.producer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
pipeline_s0_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
typename CollectiveMainloop::PipelineC pipeline_s0_corr(
  shared_storage.pipelines.s0_corr,
  pipeline_s0_corr_params,
  /*barrier init*/ cute::true_type{}
);
```
This example illustrates how to establish a producer-consumer relationship. Note that the pipeline parameters in the two code snippets are slightly different. The essential reasons for these differences, which are related to TMA and TMEM, have already been explained in previous sections, so a more detailed description is omitted here.    

Another interesting aspect is how synchronization between the producer and consumer is achieved. Let's first look at the following code snippet:
```
typename CollectiveMainloop::PipelineQ::PipelineState pipeline_load_q_consumer_state;
typename CollectiveMainloop::PipelineQ::PipelineState pipeline_load_q_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineQ>();
```
The deatil code snippet of `PipelineState` in file cutlass/pipeline/sm90_pipeline.hpp:
```
template<uint32_t Stages_>
struct PipelineState {

  static constexpr uint32_t Stages = Stages_;

  int index_ = 0;
  uint32_t phase_ = 0;
  uint32_t count_ = 0;

  CUTLASS_DEVICE
  PipelineState(): index_{}, phase_{}, count_{} {}

  CUTLASS_DEVICE
  PipelineState(int index, uint32_t phase, uint32_t count)
    : index_(index)
    , phase_(phase)
    , count_(count) {}

  CUTLASS_DEVICE
  void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      ++count_;
      if (index_ == Stages) {
        index_ = 0;
        phase_ ^= 1;
      }
    }
  }
}
```
`Stages` represents the number of memory buffers used for pipelining, determining the degree of overlap between memory access and computation. This is analogous to double buffering or ping-pong methods commonly used in single-precision matrix multiplication with CUDA cores. `index_` is the array subscript indicating the current buffer, and `phase_` indicates whether the buffer is available for use. To better illustrate the roles of `index_` and `phase_`, consider the following code snippet from cutlass/pipeline/sm100_pipeline.hpp and cutlass/pipeline/sm90_pipeline.hpp:
```
// cutlass/pipeline/sm90_pipeline.hpp
template <int Stages_>
class PipelineTmaAsync {
public:
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = cutlass::PipelineState<Stages>;

  struct SharedStorage {
    FullBarrier full_barrier_[Stages];
    EmptyBarrier empty_barrier_[Stages];
  };
  ...
}
```
```
// cutlass/pipeline/sm100_pipeline.hpp
template <
  int Stages_,
  class ClusterShape = Shape<int,int,_1>,
  class AtomThrShape_MNK_ = Shape<_1,_1,_1>
>
class PipelineTmaUmmaAsync {
public:
  static constexpr uint32_t Stages = Stages_;
  using AtomThrShape_MNK = AtomThrShape_MNK_;
private:
  using Impl = PipelineTmaAsync<Stages>;
public:
  using FullBarrier  = typename Impl::FullBarrier;
  using EmptyBarrier = typename Impl::EmptyBarrier;
  ...
}
```
Obviously, both `full_barrier_ptr_` and `empty_barrier_ptr_` are arrays in shared memory (SMEM) and work together to maintain synchronization. `empty_barrier_ptr_` records the status of the producer buffer, while `full_barrier_ptr_` records the status of the consumer buffer. When `producer_acquire` is called, it updates the buffer status in the `empty_barrier_ptr_` array at the position specified by the `index_` and `phase_` fields of `PipelineState`. Similarly, the `full_barrier_ptr_` array is updated for the consumer side. For more details, refer to cutlass/pipeline/sm90_pipeline.hpp.
## Tile Scheduler
This section will illustrate two tile scheduling strategies: persistent and non-persistent tile schedulers.
### No-Persistent Tile scheduler
In a non-persistent tile scheduler, each CTA completes only one tile of computation. For example, `IndividualTileScheduler` is non-persistent in FMHA because its grid shape is set as shown in the following code snippet.
```
template<class ProblemSize, class ClusterShape, class TileShape>
static Params to_underlying_arguments(
  const ProblemSize& problem_size, KernelHardwareInfo hw_info,
  const ClusterShape& cluster_shape, const TileShape& tile_shape)
{
    dim3 grid(round_up(ceil_div(size<0>(problem_size), size<0>(tile_shape)), size<0>(cluster_shape)), 
        size<3,0>(problem_size), 
        size<3,1>(problem_size));
    return Params{ grid };
}
```
### Persistent Tile scheduler
In contrast, the persistent tile scheduler allows each CTA to process multiple work units (tiles), with each work unit's offset being `gridDim.x`. The grid shape is set to the number of SMs if the number of work units exceeds the number of SMs.
```
static dim3 get_grid_shape(Params const& params)
{
    dim3 grid(std::min(params.num_blocks, params.hw_info.sm_count), 1, 1);
    return grid;
}
```
In this way, it can mitigate both load imbalance and the **wave quantization** problem.   
**Wave quantization**   
When the number of work units exceeds the number of available SMs, the work units are processed in multiple waves. One wave is defined as each available SM completing a single work unit. **Wave quantization** occurs when the number of work units is not evenly divisible by the number of available SMs. For example, consider a case with 10 work units and 4 SMs. The work unit execution timeline would look like:    
![Wave quantization](./src_pictures/wave_quantization.png)    
In this case, the first two waves are full waves, with every SM being utilized. However, the final wave is a partial wave, where only half of the SMs are occupied.    

However, the persistent tile scheduler can maintain high SM utilization by reducing the idle time that occurs while SMs wait to be assigned new CTAs, thereby mitigating the wave quantization problem. More information about **wave quantization** and additional performance guidance can be found in the blog post [CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) and the [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-mem).
# Collective Layer
Updating... ... ...