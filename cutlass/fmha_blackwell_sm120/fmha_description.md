
> **File:** fmha_description.md  
> **Description:** A universal device layer for CUTLASS 3.x-style FMHA kernels.  
> **Reference:** [cutlass/examples/blackwell_fmha](https://github.com/cutlass/cutlass)

# Preface
The article aims to provide a detailed explanation of the FMHA kernels in CUTLASS 3.x, with the kernels originating from the CUTLASS examples in blackwell_fmha.

The documentation of CUTLASS shows that the GEMM kernel API consists of five layers, from top to bottom: device, kernel, collective, tiled MMA and copy, and atom. Instead of analyzing each layer separately, we will group the last three layers together for analysis, while also considering the first two layers. Meanwhile, the usage of Cute may be involved in the collective layer. Additionally, as this article serves as learning notes for CUTLASS, there may be some unreasonable explanations. You are welcome to critique and correct any mistakes.

# Device
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

