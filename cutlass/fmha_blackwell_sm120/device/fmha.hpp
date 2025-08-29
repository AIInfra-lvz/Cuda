
/*!
  \file
  \brief An universal device layer for cutlass 3.x-style kernels.
  \reference cutlass/examples/blackwell_fmha (https://github.com/cutlass/cutlass)
*/


#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"

#if !defined(__CUDACC_RTC__)
#include "cutlass/cluster_launch.hpp"
#include "cutlass/trace.h"
#endif // !defined(__CUDACC_RTC__)

////////////////////////////////////////////////////////////////////////

namespace cutlass::fmha::device 
{

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// CUTLASS 3.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Kernel_>
class FMHA 
{
public:
  using Kernel    = Kernel_;
  using Arguments = typename Kernel::Arguments;
  using Params    = typename Kernel::Params;

  static int const kThreadCount = Kernel::MaxThreadsPerBlock;

private:

  Params params_;

  bool is_initialized(bool set = false) 
  {
    static bool initialized = false;

    if (set)
    {
      initialized = true;
    }
    
    return initialized;
  }

  /**
   * @brief Modifies the maximum dynamic shared memory size of active blocks per multiprocessor.
   * Note: By default, the maximum dynamic shared memory size is 48KB.
   */
  static cudaError_t modify_max_dynamic_shared_memory_size(int& smem_size)
  {
    cudaError_t result = cudaSuccess;

    if (smem_size >= (48 << 10))
    {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);

      result = cudaFuncSetAttribute(
        device_kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
      );
    }

    return result;
  }

public:

  /// Access the Params structure
  const Params& params() const 
  {
    return params_;
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(const Arguments& args) 
  {
    if (Kernel::can_implement(args)) 
    {
      return Status::kSuccess;
    }
    else 
    {
      return Status::kInvalid;
    }
  }

  /// Gets the workspace size
  static size_t get_workspace_size(const Arguments& args) 
  {
    size_t workspace_bytes = 0;
    workspace_bytes += Kernel::get_workspace_size(args);
    return workspace_bytes;
  }

  /// Computes the grid shape
  static dim3 get_grid_shape(const Params& params) 
  {
    return Kernel::get_grid_shape(params);
  }

  /**
   * @brief  Gets the maximum number of active blocks per multiprocessor.
   * @return The maximum number of active blocks per multiprocessor, or -1 if an error occurs.
   */
  static int get_max_active_blocks_per_multiprocessor() 
  {
    CUTLASS_TRACE_HOST("FMHA::maximum_active_blocks()");

    int max_active_blocks = -1;
    int smem_size = Kernel::SharedStorageSize;

    // first, account for dynamic smem capacity if needed
    cudaError_t result = modify_max_dynamic_shared_memory_size(smem_size);

    if (cudaSuccess != result)
    {
      result = cudaGetLastError();
      CUTLASS_TRACE_HOST(
        "  cudaFuncSetAttribute() returned error: "
        << cudaGetErrorString(result));
      return -1;
    }

    result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks,
      device_kernel<Kernel>,
      Kernel::MaxThreadsPerBlock,
      smem_size
    );

    if (cudaSuccess != result)
    {
      result = cudaGetLastError();
      CUTLASS_TRACE_HOST(
        "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error: "
        << cudaGetErrorString(result));
      return -1;
    }
    
    CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
    return max_active_blocks;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(const Arguments& args, void* workspace = nullptr, cudaStream_t stream = nullptr)
  {
      CUTLASS_TRACE_HOST("FMHA::initialize() - workspace "
        << workspace << ", stream: " << (stream ? "non-null" : "null"));
      
      // Initialize the workspace
      Status status = Kernel::initialize_workspace(args, workspace, stream);

      if (Status::kSuccess != status)
      {
        return status;
      }

      // Initialize the Params structure
      params_ = Kernel::to_underlying_arguments(args, workspace);

      if (is_initialized())
      {
        return Status::kSuccess;
      }

      // account for dynamic smem capacity if needed
      int smem_size = Kernel::SharedStorageSize;

      cudaError_t result = modify_max_dynamic_shared_memory_size(smem_size);

      if (cudaSuccess != result) 
      {
        result = cudaGetLastError(); // to clear the error bit
        CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result));
        return Status::kErrorInternal;
      }

      is_initialized(true);

      return Status::kSuccess;
  }

  /// Primary run() entry point API that is static allowing users to create and manage their own params.
  /// Supplied params struct must be construct by calling Kernel::to_underling_arguments()
  static Status run(Params& params, cudaStream_t stream = nullptr)
  {
    CUTLASS_TRACE_HOST("FMHA::run()");

    const dim3 block = Kernel::get_block_shape();
    const dim3 grid  = get_grid_shape(params);
    int smem_size    = Kernel::SharedStorageSize;   // configure smem size and carveout
    Status launch_result;

    // Use extended launch API only for mainloops that use it
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

    cudaError_t result = cudaGetLastError();

    if (cudaSuccess == result && Status::kSuccess == launch_result) 
    {
      return Status::kSuccess;
    }
    else 
    {
      CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
      return Status::kErrorInternal;
    }
  }

  //
  // Non-static launch overloads that first create and set the internal params struct of this kernel handle.
  //
  Status run(const Arguments& args, void* workspace = nullptr, cudaStream_t stream = nullptr) 
  {
    Status status = initialize(args, workspace, stream);

    if (Status::kSuccess == status) 
    {
      status = run(params_, stream);
    }

    return status;
  }

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status operator()(const Arguments& args, void* workspace = nullptr, cudaStream_t stream = nullptr) 
  {
    return run(args, workspace, stream);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status run(cudaStream_t stream = nullptr) 
  {
    return run(params_, stream);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status operator()(cudaStream_t stream = nullptr) 
  {
    return run(params_, stream);
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::fmha::device

////////////////////////////////////////////////////////////////////////////////

