
/*!
  \file
  \brief An universal device layer for cutlass 3.x-style kernels.
  \reference cutlass/examples/blackwell_fmha (https://github.com/cutlass/cutlass)
*/

#include <iostream>
#include <memory>
#include <vector>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "device/fmha.hpp"
#include "kernel/sm120_fmha_fwd_kernel_builder.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using namespace cutlass::fmha::kernel;

////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct DeviceAllocation
{
  T* ptr_        = nullptr;
  size_t offset_ = 0;
  size_t size_   = 0;

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  DeviceAllocation() = default;
  DeviceAllocation(size_t size) { reset(size); }
  ~DeviceAllocation() { deAllocate(); }

  void reset(size_t size, size_t offset=0)
  {
    deAllocate();
    auto ret = cudaMalloc(&ptr_, sizeof(T) * (size + offset));
    assert(ret == cudaSuccess);
    size_ = size;
    offset_ = offset;
  }

  T* get() { return ptr_ + offset_; }

  const T* get() const { return ptr_ + offset_; }

  void deAllocate()
  {
    if (ptr_ != nullptr)
    {
      auto ret = cudaFree(ptr_);
      assert(ret == cudaSuccess);
    }
  }

  size_t size() const { return size_; }

  size_t get_storage_size() const { return (size_ + offset_) * sizeof(T); }

  void copy_from_host(const T* ptr, size_t size)
  {
    auto ret = cudaMemcpy(ptr_, ptr, size * sizeof(T), cudaMemcpyDefault);
    assert(ret == cudaSuccess);
  }

  void copy_from_device(const T* ptr, size_t size)
  {
    auto ret = cudaMemcpy(ptr_, ptr, size * sizeof(T), cudaMemcpyDefault);
    assert(ret == cudaSuccess);
  }
};


struct FwdRunner
{
  using Element = cutlass::half_t;
  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;
  using ElementOut = cutlass::half_t;

  using TileShape    = Shape<_256, _128, _128>;                             // (Q, K, Head_dims)
  using ProblemShape = tuple<int, int, int, tuple<tuple<int, int>, int>>;   // Q K D ((H_G H_K), B)

  using StrideQ = tuple<int, _1, tuple<tuple<int, int>, int>>;   // Q D ((H_G H_K), B)
  using StrideK = tuple<int, _1, tuple<tuple<_0, int>, int>>;    // K D ((H_G H_K), B)
  using StrideV = StrideK;                                       // K D ((H_G H_K), B)
  using StrideO = StrideQ;                                       // Q D ((H_G H_K), B)
  using StrideLSE = tuple<_1, tuple<tuple<int, int>, int>>;      // Q   (（H_G H_K), B)

  using Kernel = cutlass::fmha::kernel::Sm120FmhaFwdBuilder<
    Element, 
    ElementAccumulatorQK, ElementAccumulatorPV,
    StrideQ, StrideK, StrideV, 
    StrideO, StrideLSE,
    TileShape,
    ProblemShape>::sm120_fmha_kernel;

  using Operation = cutlass::fmha::device::FMHA<Kernel>;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideLSE stride_LSE;
  uint64_t seed = 0;

  struct DeviceBuffer
  {
    DeviceAllocation<Element> block_Q;
    DeviceAllocation<Element> block_K;
    DeviceAllocation<Element> block_V;
    DeviceAllocation<ElementOut> block_O;
    DeviceAllocation<ElementAccumulatorPV> block_LSE;
    DeviceAllocation<int> device_cumulative_seqlen_q;
    DeviceAllocation<int> device_cumulative_seqlen_kv;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    size_t get_storage_size() const {
      return block_Q.get_storage_size() + block_K.get_storage_size() + block_V.get_storage_size()
          + block_O.get_storage_size() + block_LSE.get_storage_size() + device_cumulative_seqlen_q.get_storage_size()
          + device_cumulative_seqlen_kv.get_storage_size();
    }
  };

  std::vector<std::unique_ptr<DeviceBuffer>> buffers;
  std::vector<int> cumulative_seqlen_q;
  std::vector<int> cumulative_seqlen_kv;

  //
  // Methods
  //

  template<int Q, int K, int Head_Dims, int H_Num, int H_Key, int Batch>
  ProblemShape initialize()
  {
    constexpr int H_G = H_Num / H_Key;

    auto problem_shape_in = cute::make_tuple(Q, K, Head_Dims, cute::make_tuple(cute::make_tuple(H_G, H_Key), Batch));
    ProblemShape problem_shape = problem_shape_in;
    decltype(problem_shape_in) problem_size = problem_shape_in;

    get<2>(problem_size) = cutlass::round_up(get<2>(problem_size), 8);  // alignment

    auto shape_QO = select<0, 2, 3>(problem_size);
    auto shape_KV = select<1, 2, 3>(problem_size);
    auto shape_LSE = select<0,3>(problem_size);

    int SQ = size<0>(problem_size);
    int SK = size<1>(problem_size);
    int D  = size<2>(problem_size);
    int H  = size<3,0>(problem_size);
    int H_K = size<3,0,1>(problem_size);
    int H_Q = size<3,0,0>(problem_size);
    int B   = size<3,1>(problem_size);

    stride_Q = make_stride(H*D, _1{}, make_stride(make_stride(D, H_Q*D), H*D*SQ));
    stride_O = stride_Q;
    stride_K = make_stride(H_K*D, _1{}, make_stride(make_stride(_0{}, D), H_K*D*SK));
    stride_V = stride_K;
    stride_LSE = make_stride(_1{}, make_stride(make_stride(SQ, SQ*H_Q), SQ*H));

    auto buffer_init_fn = [&](auto& buffer) {
      buffer.block_Q.reset(size(shape_QO));
      buffer.block_K.reset(size(shape_KV));
      buffer.block_V.reset(size(shape_KV));
      buffer.block_O.reset(size(shape_QO));
      buffer.block_LSE.reset(size(shape_LSE));

      cudaMemset(buffer.block_Q.get(), 1, buffer.block_Q.size() * sizeof(Element));
      cudaMemset(buffer.block_K.get(), 1, buffer.block_K.size() * sizeof(Element));
      cudaMemset(buffer.block_V.get(), 1, buffer.block_V.size() * sizeof(Element));
      cudaMemset(buffer.block_O.get(), 0, buffer.block_O.size() * sizeof(ElementOut));

      if (!cumulative_seqlen_q.empty())
      {
        buffer.device_cumulative_seqlen_q.reset(cumulative_seqlen_q.size());
        buffer.device_cumulative_seqlen_q.copy_from_host(
          cumulative_seqlen_q.data(), cumulative_seqlen_q.size());
      }

      if (!cumulative_seqlen_kv.empty())
      {
        buffer.device_cumulative_seqlen_kv.reset(cumulative_seqlen_kv.size());
        buffer.device_cumulative_seqlen_kv.copy_from_host(
          cumulative_seqlen_kv.data(), cumulative_seqlen_kv.size());
      }
    };
      
    buffers.push_back(std::make_unique<DeviceBuffer>());
    buffer_init_fn(*buffers.back());

    return problem_shape;
  };

  auto get_arguments(
    const ProblemShape& problem_shape, 
    const cutlass::KernelHardwareInfo& hw_info, 
    int buffer_index) 
  {
    auto problem_shape_ = problem_shape;

    typename Operation::Arguments arguments{
      problem_shape_,
      { buffers[buffer_index]->block_Q.get(), stride_Q,
        buffers[buffer_index]->block_K.get(), stride_K,
        buffers[buffer_index]->block_V.get(), stride_V },
      { buffers[buffer_index]->block_O.get(), stride_O,
        buffers[buffer_index]->block_LSE.get(), stride_LSE },
      hw_info
    };
    return arguments;
  }

  template<int Q, int K, int D, int H, int H_K, int B>
  void run()
  {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    ProblemShape problem_shape = initialize<Q, K, D, H, H_K, B>();

    typename Operation::Arguments arguments = get_arguments(problem_shape, hw_info, 0);

    size_t workspace_size = 0;
    workspace_size = Operation::get_workspace_size(arguments);
    DeviceAllocation<uint8_t> workspace(workspace_size);

    Operation op;

    cutlass::Status status = cutlass::Status::kSuccess;
    status = op.can_implement(arguments);

    if (status != cutlass::Status::kSuccess)
    {
      std::cerr << "This kernel is not supported. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return;
    }

    status = op.initialize(arguments, workspace.get());

    if (status != cutlass::Status::kSuccess)
    {
      std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return;
    }

    status = op.run();

    if (status != cutlass::Status::kSuccess)
    {
      std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return;
    }
  }
};

void runDemo()
{
  constexpr int q = 256;
  constexpr int k = 128;
  constexpr int d = 128;
  constexpr int h = 64;
  constexpr int h_k = 4;
  constexpr int b = 1;

  FwdRunner runner;

  runner.run<q, k, d, h, h_k, b>();

}


int main()
{
  runDemo();

  return 0;
}