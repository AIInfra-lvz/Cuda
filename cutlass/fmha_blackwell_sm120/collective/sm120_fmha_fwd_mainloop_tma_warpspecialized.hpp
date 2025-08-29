/***************************************************************************************************
 * forward collective mainloop for TMA warp-specialized FMHA
 ***************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/arch/simd_sm100.hpp"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

#include "collective/fmha_common.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm120_fmha_load_tma_warpspecialized.hpp"

namespace cutlass::fmha::collective
{
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////

template<
  class Element_,
  class ElementQK_,
  class ElementPV_,
  class TileShape_,
  class StrideQ_,
  class StrideK_,
  class StrideV_,
  class Mask_
>
struct Sm120FwdMainloopTmaWarpspecialized
{
  using ClusterShape = Shape<_1, _1, _1>;
  using ThreadShape  = Shape<_2, _1, _1>;

  using Element   = Element_;
  using ElementQK = ElementQK_;
  using ElementPV = ElementPV_;
  using TileShape = TileShape_;
  using StrideQ   = StrideQ_;
  using StrideK   = StrideK_;
  using StrideV   = StrideV_;
  using Mask      = Mask_;
  
  static constexpr int StageCountQ  = 2;
  static constexpr int StageCountKV = sizeof(Element_) == 1 ? 4 : 3;
  static const int     Alignment    = 128 / sizeof_bits_v<Element>;

  using StagesQ = cutlass::gemm::collective::StageCount<StageCountQ>;
  using StagesKV = cutlass::gemm::collective::StageCount<StageCountKV>;

  using TileShapeQK  = decltype(shape_div(TileShape{}, ThreadShape{}));
  using TileShapePV  = decltype(select<0,2,1>(TileShapeQK{}));

  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, StrideQ, Alignment,
      Element, StrideK, Alignment,
      ElementQK,
      TileShapeQK, ClusterShape, cutlass::gemm::collective::StageCount<3> /* we change it later anyways*/,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

  using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    // the stride for A does not matter since we do not load from smem at all
    Element, StrideK, Alignment,
    Element, decltype(select<1,0,2>(StrideV{})), Alignment,
    ElementPV,
    TileShapePV, ClusterShape, cutlass::gemm::collective::StageCount<3> /* we change it later anyways*/,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

  using SmemLayoutQ = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutA{}, Int<StageCountQ>{}));
  using SmemLayoutK = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutB{}, Int<StageCountKV>{}));
  using SmemLayoutV = decltype(unstageSmemLayout(typename CollectiveMmaPV::SmemLayoutB{}, Int<StageCountKV>{}));

  struct TensorStorage
  {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    };
  };

  enum class TmemAllocation : uint32_t 
  {
    kSizeS = 128,
    kSizeO = 128,
    kSizeP = 32,
    S0 = 0,
    S1 = S0 + kSizeS,
    V0 = S0,  // stats storage from softmax to correction
    V1 = S1,
    P0 = S0 + kSizeP,
    P1 = S1 + kSizeP,
    O0 = S1 + kSizeS,
    O1 = O0 + kSizeO,
    kEnd = O1 + kSizeO
  };

  // indices for V0 / V1
  enum : int 
  {
    kIdxOldRowMax = 0,
    kIdxNewRowMax = 1,
    kIdxFinalRowSum = 0,
    kIdxFinalRowMax = 1
  };

  // from load to mma warp, protects q in smem
  using PipelineQ = cutlass::PipelineTmaUmmaAsync<
    StageCountQ,
    typename CollectiveMmaQK::AtomThrShapeMNK
  >;

  // from load to mma warp, protects k/v in smem
  using PipelineKV = cutlass::PipelineTmaUmmaAsync<
    StageCountKV,
    typename CollectiveMmaQK::AtomThrShapeMNK
  >;

  // from mma to softmax0/1 warp, protects S in tmem
  // (not sure yet about the reverse direction)
  // there is one pipe per softmax warp, and the mma warp alternates between them
  using PipelineS = cutlass::PipelineUmmaAsync<1>;

  // from softmax0/1/ to correction wg
  using PipelineC = cutlass::PipelineAsync<1>;

  // from mma to correction
  using PipelineO = cutlass::PipelineUmmaAsync<2>;

  // from corr to epilogue
  using PipelineE = cutlass::PipelineAsync<2>;

  using OrderBarrierSoftmax = cutlass::OrderedSequenceBarrier<
    /*stages*/ 1, /*groups*/ 2>;

  static const int TransactionBytesLoadQ = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutQ{})) * cute::sizeof_bits_v<Element>);
  static const int TransactionBytesLoadK = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>);
  static const int TransactionBytesLoadV = cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutV{})) * cute::sizeof_bits_v<Element>);

  static_assert(TransactionBytesLoadK == TransactionBytesLoadV, "K and V smem layouts must be of equal size");

  using Load = Sm120FmhaLoadTmaWarpspecialized<
    Element, StrideQ, StrideK, StrideV,
    CollectiveMmaQK, CollectiveMmaPV,
    SmemLayoutQ, SmemLayoutK, SmemLayoutV,
    TensorStorage, PipelineQ, PipelineKV, Mask, TileShape
  >;

  struct Arguments
  {
    typename Load::Arguments load;

    // if zero, defaults to 1/sqrt(D)
    float scale_softmax = 0.0f;

    // scaling factors to dequantize QKV
    float scale_q = 1.0f;
    float scale_k = 1.0f;
    float scale_v = 1.0f;

    // scaling factor to quantize O
    float inv_scale_o = 1.0f;
  };

  struct Params
  {
    typename Load::Params load;

    float scale_softmax;
    float scale_softmax_log2;

    float scale_output;
  };

  template<class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args)
  {
    return true;
  }

  template<class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace)
  {

    float scale_softmax = args.scale_softmax;
    if (scale_softmax == 0.0f)
    {
      scale_softmax = 1.0f / (float) std::sqrt(get<2>(problem_shape));
    }

    float log2_e = static_cast<float>(std::log2(std::exp(1.0)));

    return Params{
        Load::to_underlying_arguments(problem_shape, args.load, workspace),
        args.scale_q * args.scale_k * scale_softmax,
        args.scale_q * args.scale_k * log2_e * scale_softmax,
        args.scale_v * args.inv_scale_o
    };
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params)
  {
    Load::prefetch_tma_descriptors(params.load);
  }

  template<class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void
  load(
      const BlkCoord& blk_coord, const ProblemShape& problem_shape,
      const Params& params, const ParamsProblemShape& params_problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_producer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_producer_state)
  {
    Load load;
    load.load(blk_coord, problem_shape, params.load, params_problem_shape,
        storage,
        pipeline_q, pipeline_q_producer_state,
        pipeline_kv, pipeline_kv_producer_state);
  }

  template<class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE auto
  mma(const BlkCoord& blk_coord,
      const Params& params, const ProblemShape& problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_consumer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_consumer_state,
      PipelineS& pipeline_s0, typename PipelineS::PipelineState& pipeline_s0_producer_state,
      PipelineS& pipeline_s1, typename PipelineS::PipelineState& pipeline_s1_producer_state,
      PipelineO& pipeline_corr, typename PipelineO::PipelineState& pipeline_corr_producer_state)
  {
    auto pipeline_q_release_state  = pipeline_q_consumer_state;
    auto pipeline_kv_release_state = pipeline_kv_consumer_state;

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    typename CollectiveMmaQK::TiledMma mma_qk;
    ThrMMA thr_mma_qk = mma_qk.get_slice(0);      // only one CTA in group because no multicast

    typename CollectiveMmaPV::TiledMma mma_pv;
    TiledMMA mma_pv_ts = to_tiled_mma_sm100_ts(mma_pv);   // p x v : tmem x smem
    ThrMMA thr_mma_pv  = mma_pv_ts.get_slice(0);      // only one CTA in group because no multicast

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

    Tensor tSrQ = thr_mma_qk.make_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.make_fragment_B(sK);
    Tensor tOrV = thr_mma_pv.make_fragment_B(sV);
  
    // tmem layout is
    // S0 S1`O0 O1
    // sequential in memory, where S overlaps with P and V
    Tensor tStS = partition_fragment_C(mma_qk, select<0,1>(TileShapeQK{}));
    Tensor tOtO = partition_fragment_C(mma_pv_ts, select<0,1>(TileShapePV{}));
    
    Tensor tStS0 = tStS;
    tStS0.data() = tStS.data().get() + uint32_t(TmemAllocation::S0);
    Tensor tStS1 = tStS;
    tStS1.data() = tStS.data().get() + uint32_t(TmemAllocation::S1);

    Tensor tOtO0 = tOtO;
    tOtO0.data() = tOtO.data().get() + uint32_t(TmemAllocation::O0);
    Tensor tOtO1 = tOtO;
    tOtO1.data() = tOtO.data().get() + uint32_t(TmemAllocation::O1);

    // Element needs half precision
    Tensor sP    = make_tensor(make_smem_ptr((Element*)nullptr), typename CollectiveMmaPV::SmemLayoutA{});
    Tensor tOrP  = thr_mma_pv.make_fragment_A(sP)(_, _, _, 0);

    Tensor tOrP0 = tOrP;
    tOrP0.data() = tOrP.data().get() + uint32_t(TmemAllocation::P0);
    Tensor tOrP1 = tOrP;
    tOrP1.data() = tOrP.data().get() + uint32_t(TmemAllocation::P1);

    int k_index = 0;
    int v_index = 0;
    int q_index = 0;

    // wait for Q0
    q_index = pipeline_q_consumer_state.index();
    pipeline_q.consumer_wait(pipeline_q_consumer_state);
    ++pipeline_q_consumer_state;

    Tensor tSrQ0 = tSrQ(_, _, _, q_index);

    // wait for K0
    k_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;

    // gemm Q0 * K0 -> S0
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);

    gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);

    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;

    // release K0
    if constexpr (get<1>(ThreadShape{}) > 1)
    {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    // wait for Q1
    if constexpr (get<0>(ThreadShape{}) > 1 || get<2>(ThreadShape{}) > 1)
    {
      q_index = pipeline_q_consumer_state.index();
      pipeline_q.consumer_wait(pipeline_q_consumer_state);
      ++pipeline_q_consumer_state;
    }

    Tensor tSrQ1 = tSrQ(_,_,_,q_index);

    // wait for K1
    if constexpr (get<1>(ThreadShape{}) > 1)
    {
      k_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    // gemm Q1 * K0 -> S1
    pipeline_s1.producer_acquire(pipeline_s1_producer_state);

    gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index), tStS1);

    pipeline_s1.producer_commit(pipeline_s1_producer_state);
    ++pipeline_s1_producer_state;

    // release K0
    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    // wait for V0
    v_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;

    pipeline_corr.producer_acquire(pipeline_corr_producer_state);
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);

    // gemm P0 * V0 -> O0
    gemm_zero_acc(mma_pv_ts, tOrP0, tOrV(_,_,_,v_index), tOtO0);

    pipeline_corr.producer_commit(pipeline_corr_producer_state);
    ++pipeline_corr_producer_state;

    if constexpr (get<1>(ThreadShape{}) > 1)
    {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    mma_pv_ts.accumulate_ = UMMA::ScaleOut::Zero;

    // loop:
    mask_tile_count -= 1;

    for (; mask_tile_count > 0; mask_tile_count -= 1)
    {
      // wait for Ki
      k_index = (pipeline_kv_consumer_state.index());
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      // gemm Q1 * Ki -> S1
      gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);

      pipeline_s0.producer_commit(pipeline_s0_producer_state);
      ++pipeline_s0_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1)
      {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      }

      // gemm P2 * V(i-1) -> O2
      if constexpr (get<1>(ThreadShape{}) > 1)
      {
        v_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
      }

      pipeline_corr.producer_acquire(pipeline_corr_producer_state);
      pipeline_s1.producer_acquire(pipeline_s1_producer_state);

      gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index), tOtO1);

      pipeline_corr.producer_commit(pipeline_corr_producer_state);
      ++pipeline_corr_producer_state;

      // release V(i-1)
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;

      if constexpr (get<1>(ThreadShape{}) > 1)
      {
        k_index = (pipeline_kv_consumer_state.index());
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
      }

      // gemm Q2 * Ki -> S2
      gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index), tStS1);

      pipeline_s1.producer_commit(pipeline_s1_producer_state);
      ++pipeline_s1_producer_state;

      // release Ki
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;

      // wait for Vi
      v_index = (pipeline_kv_consumer_state.index());
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      // gemm P1 * Vi -> O1
      pipeline_corr.producer_acquire(pipeline_corr_producer_state);

      pipeline_s0.producer_acquire(pipeline_s0_producer_state);

      gemm_reset_zero_acc(mma_pv_ts, tOrP0, tOrV(_,_,_,v_index), tOtO0);

      pipeline_corr.producer_commit(pipeline_corr_producer_state);
      ++pipeline_corr_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      }      
    }

    // release Q0
    pipeline_q.consumer_release(pipeline_q_release_state);
    ++pipeline_q_release_state;

    // release Q1
    if constexpr (get<0>(ThreadShape{}) > 1)
    {
      pipeline_q.consumer_release(pipeline_q_release_state);
      ++pipeline_q_release_state;
    }

    // wait for Vi
    if constexpr (get<1>(ThreadShape{}) > 1)
    {
      v_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    // gemm P2 * Vi -> O2
    pipeline_corr.producer_acquire(pipeline_corr_producer_state);
    pipeline_s1.producer_acquire(pipeline_s1_producer_state);

    gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index), tOtO1);

    pipeline_corr.producer_commit(pipeline_corr_producer_state);
    ++pipeline_corr_producer_state;

    // release Vi
    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;

    pipeline_s1.producer_commit(pipeline_s1_producer_state);
    ++pipeline_s1_producer_state;
  }

  template<bool need_apply_mask, class Stage, class BlkCoord, class CoordTensor, class ProblemShape>
  CUTLASS_DEVICE auto
  softmax_step(
      float& row_max, float& row_sum,
      Stage stage, bool final_call,
      const BlkCoord& blk_coord, const CoordTensor& cS,
      const Params& params, const ProblemShape& problem_shape,
      PipelineS& pipeline_s, typename PipelineS::PipelineState& pipeline_s_consumer_state,
      PipelineC& pipeline_c, typename PipelineC::PipelineState& pipeline_c_producer_state,
      OrderBarrierSoftmax& order_s)
    {
      typename CollectiveMmaQK::TiledMma mma_qk;

      Tensor tScS = mma_qk.get_slice(0).partition_C(cS);
      Tensor tStS = partition_fragment_C(mma_qk, select<0,1>(TileShapeQK{}));

      Tensor tStS_v = tStS.compose(make_layout(make_shape(_128{}, _2{})));
      Tensor tScS_v = tScS.compose(make_layout(make_shape(_128{}, _2{})));

      auto tilePLikeFP32 = size<1>(TileShapeQK{}) / Int<sizeof(float)>{} * Int<sizeof(Element)>{};
      Tensor tStS_P = tStS.compose(make_layout(make_shape(_128{}, tilePLikeFP32)));
      Tensor tScS_P = tScS.compose(make_layout(make_shape(_128{}, tilePLikeFP32)));

      /*
       * Notice: current only handle half precision
       */
      // Each thread owns a single row
      using TMEM_LOAD  = SM100_TMEM_LOAD_32dp32b32x;     // 4x32 threads with 128 cols of 32b elements
      using TMEM_STORE = SM100_TMEM_STORE_32dp32b32x;    // 4x32 threads with 128 cols of 32b elements
      using TMEM_STORE_V = SM100_TMEM_STORE_32dp32b2x;   // 4x32 threads with 2 cols of 32b elements

      int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

      auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tStS);
      auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);
      Tensor tTMEM_LOADts  = thr_tmem_load.partition_S(tStS);
      Tensor tTMEM_LOADcS  = thr_tmem_load.partition_D(tScS);
      tTMEM_LOADts.data() = tTMEM_LOADts.data().get() + uint32_t(stage == _0{} ? TmemAllocation::S0 : TmemAllocation::S1);

      auto tiled_tmem_storev = make_tmem_copy(TMEM_STORE_V{}, tStS_v);
      auto thr_tmem_storev  = tiled_tmem_storev.get_slice(thread_idx);
      Tensor tTMEM_STOREVtS = thr_tmem_storev.partition_D(tStS_v);
      Tensor tTMEM_STOREVcS = thr_tmem_storev.partition_S(tScS_v);
      tTMEM_STOREVtS.data() = tTMEM_STOREVtS.data().get() + uint32_t(stage == _0{} ? TmemAllocation::V0 : TmemAllocation::V1);

      auto tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tStS_P);
      auto thr_tmem_store  = tiled_tmem_store.get_slice(thread_idx);
      Tensor tTMEM_STOREtS = thr_tmem_store.partition_D(tStS_P);
      Tensor tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P);
      tTMEM_STOREtS.data() = tTMEM_STOREtS.data().get() + uint32_t(stage == _0{} ? TmemAllocation::P0 : TmemAllocation::P1);

      pipeline_s.consumer_wait(pipeline_s_consumer_state);

      // read all of S from tmem into reg mem
      Tensor tTMEM_LOADrS = make_tensor<ElementQK>(shape(tTMEM_LOADcS));
      copy(tiled_tmem_load, tTMEM_LOADts, tTMEM_LOADrS);
      

      if constexpr (need_apply_mask)
      {
        Mask{}.apply_mask(tTMEM_LOADrS, tTMEM_LOADcS, problem_shape);
      }

      ElementQK old_row_max = row_max;
      {
        // compute rowmax
        float row_max_0 = row_max;
        float row_max_1 = row_max;
        float row_max_2 = row_max;
        float row_max_3 = row_max;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tTMEM_LOADrS); i += 4)
        {
          row_max_0 = ::fmax(row_max_0, tTMEM_LOADrS(i));
          row_max_1 = ::fmax(row_max_1, tTMEM_LOADrS(i+1));
          row_max_2 = ::fmax(row_max_2, tTMEM_LOADrS(i+2));
          row_max_3 = ::fmax(row_max_3, tTMEM_LOADrS(i+3));
        }

        row_max = ::fmax(row_max_0, row_max_1);
        row_max = ::fmax(row_max, row_max_2);
        row_max = ::fmax(row_max, row_max_3);
      }

      ElementQK row_max_safe = row_max == -INFINITY ? 0 : row_max;

      // write back to tmem
      Tensor tTMEM_STOREVrS = make_tensor<ElementQK>(shape(tTMEM_STOREVcS));
      tTMEM_STOREVrS(kIdxOldRowMax) = old_row_max;
      tTMEM_STOREVrS(kIdxNewRowMax) = row_max_safe;
      copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);

      pipeline_c.producer_commit(pipeline_c_producer_state);
      ++pipeline_c_producer_state;

      ElementQK scale = params.scale_softmax_log2;
      ElementQK row_max_scale = row_max_safe * scale;

      float2 scale_fp32x2 = make_float2(scale, scale);
      float2 minus_row_max_scale_fp32x2 = make_float2(-row_max_scale, -row_max_scale);

      constexpr int kConversionsPerStep = 2;
      Tensor tTMEM_STORErS = make_tensor<uint32_t>(shape(tTMEM_STOREcS));
      Tensor tTMEM_STORErS_e = recast<Array<Element, kConversionsPerStep>>(tTMEM_STORErS);
      NumericArrayConverter<Element, ElementQK, kConversionsPerStep> convert;

      const int kReleasePipeCount = 10;  // must be multiple of 2

      order_s.wait();

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTMEM_LOADrS); i += 2)
      {
        float2 in = make_float2(tTMEM_LOADrS(i + 0), tTMEM_LOADrS(i + 1));
        float2 out;
        cute::fma(out, scale_fp32x2, in, minus_row_max_scale_fp32x2);
        tTMEM_LOADrS(i + 0) = out.x;
        tTMEM_LOADrS(i + 1) = out.y;

        tTMEM_LOADrS(i+0) = ::exp2f(tTMEM_LOADrS(i+0));
        tTMEM_LOADrS(i+1) = ::exp2f(tTMEM_LOADrS(i+1));

        Array<ElementQK, kConversionsPerStep> in_conv;

        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kConversionsPerStep; j++)
        {
          in_conv[j] = tTMEM_LOADrS(i + j);
        }
        tTMEM_STORErS_e[i / kConversionsPerStep] = convert(in_conv);

        if (i == size(tTMEM_LOADrS) - kReleasePipeCount)
        {
          order_s.arrive();
        }

        // this prevents register spills in fp16
        if constexpr (size<2>(tTMEM_STORErS) == _2{})
        {
          if (i == size(tTMEM_LOADrS) - 6)
          {
            copy(tiled_tmem_store, tTMEM_STORErS(_, _, 0), tTMEM_STOREtS(_, _, 0));
          }
        }
      }

      // for special cases
      if (size(tTMEM_LOADrS) <= kReleasePipeCount)
      {
        order_s.arrive();
      }

      // tmem_store(reg_S8) -> op_P      tTMEM_STORErS.shape = (FrgV, RestM, RestN, ...)
      CUTE_STATIC_ASSERT_V(size<2>(tTMEM_STORErS) <= _2{});
      CUTE_STATIC_ASSERT_V(size<1>(tTMEM_STORErS) == _1{});
      copy(tiled_tmem_store, tTMEM_STORErS(_, _, size<2>(tTMEM_STORErS) - 1), tTMEM_STOREtS(_, _, size<2>(tTMEM_STORErS) - 1));

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_s.consumer_release(pipeline_s_consumer_state);
      ++pipeline_s_consumer_state;

      pipeline_c.producer_acquire(pipeline_c_producer_state);

      ElementQK acc_scale = 0.5f * ::exp2f(scale * (old_row_max - row_max_safe));
      row_sum *= acc_scale;
      // row_sum = sum(reg_S)
      float2 local_row_sum_f32x2 = make_float2(row_sum, row_sum);
      float2 local_row_sum_1 = make_float2(0, 0);
      float2 local_row_sum_2 = make_float2(0, 0);
      float2 local_row_sum_3 = make_float2(0, 0);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTMEM_LOADrS); i += 8)
      {
        // row_sum += tTMEM_LOADrS(i);
        float2 in = make_float2(tTMEM_LOADrS(i), tTMEM_LOADrS(i+1));
        cute::add(local_row_sum_f32x2, local_row_sum_f32x2, in);

        in = make_float2(tTMEM_LOADrS(i+2), tTMEM_LOADrS(i+2+1));
        cute::add(local_row_sum_1, local_row_sum_1, in);

        in = make_float2(tTMEM_LOADrS(i+4), tTMEM_LOADrS(i+4+1));
        cute::add(local_row_sum_2, local_row_sum_2, in);

        in = make_float2(tTMEM_LOADrS(i+6), tTMEM_LOADrS(i+6+1));
        cute::add(local_row_sum_3, local_row_sum_3, in);
      }

      cute::add(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_1);
      cute::add(local_row_sum_2, local_row_sum_2, local_row_sum_3);
      cute::add(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_2);
      float local_row_sum = local_row_sum_f32x2.x + local_row_sum_f32x2.y;

      row_sum = local_row_sum;

      if (final_call)
      {
        // re-acquire the S part in the final step
        pipeline_s.consumer_wait(pipeline_s_consumer_state);

        Tensor tTMEM_STOREVrS = make_tensor<ElementQK>(shape(tTMEM_STOREVcS));
        tTMEM_STOREVrS(kIdxFinalRowMax) = row_max;
        tTMEM_STOREVrS(kIdxFinalRowSum) = row_sum;
        copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);

        cutlass::arch::fence_view_async_tmem_store();
        pipeline_s.consumer_release(pipeline_s_consumer_state);
      }
    }
  
  template<class Stage, class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE auto
  softmax(
      Stage stage,
      const BlkCoord& blk_coord,
      const Params& params, const ProblemShape& problem_shape,
      PipelineS& pipeline_s, typename PipelineS::PipelineState& pipeline_s_consumer_state,
      PipelineC& pipeline_c, typename PipelineC::PipelineState& pipeline_c_producer_state,
      OrderBarrierSoftmax& order_s)
  {
    int mask_tile_count = Mask{}.get_unmasked_trip_count(blk_coord, TileShape{}, problem_shape);
    
    ElementQK row_max = -INFINITY;
    ElementQK row_sum = 0;

    Tensor cS_base = make_identity_tensor(select<0,1>(TileShapeQK{}));
    auto logical_offset = make_coord(
      get<0>(blk_coord) * get<0>(TileShape{}) + (stage % get<0>(ThreadShape{})) * get<0>(TileShapeQK{}),
      0 + (stage % get<1>(ThreadShape{})) * get<1>(TileShapeQK{})
    );
    Tensor cS = domain_offset(logical_offset, cS_base);

    pipeline_c.producer_acquire(pipeline_c_producer_state);

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1)
    {
      softmax_step<false /* need_apply_mask */>(
          row_max, row_sum, stage,
          (mask_tile_count == 1) &&
              (Mask{}.get_masked_trip_count(blk_coord, TileShape{}, problem_shape) == 0),
          blk_coord, cS, params, problem_shape,
          pipeline_s, pipeline_s_consumer_state,
          pipeline_c, pipeline_c_producer_state,
          order_s
      );

      cS.data() = cS.data() + E<1>{} * get<1>(ThreadShape{}) * get<1>(TileShapeQK{});    //  Traverse the full unmasked tile across the N dimensions. 
    }

    // Masked iterations
    mask_tile_count = Mask{}.get_masked_trip_count(blk_coord, TileShape{}, problem_shape);

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1)
    {
      softmax_step<true /* need_apply_mask */>(
          row_max, row_sum, stage, mask_tile_count == 1,
          blk_coord, cS, params, problem_shape,
          pipeline_s, pipeline_s_consumer_state,
          pipeline_c, pipeline_c_producer_state,
          order_s
      );

      cS.data() = cS.data() + E<1>{} * get<1>(ThreadShape{}) * get<1>(TileShapeQK{});
    }

    pipeline_c.producer_commit(pipeline_c_producer_state);
    ++pipeline_c_producer_state;
  }

  template<class Stage, class TensorO>
  CUTLASS_DEVICE auto
  correction_epilogue(float scale, Stage stage, TensorO const& sO_01)
  {
    using ElementOut = typename TensorO::value_type;

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);
    Tensor sO = sO_01(_,_,stage);

    // As opposed to the softmax, we do not have enough registers here
    // to load all of the values (for tile kv = 128), so we loop
    // good values would be either 32 or 64
    const int kCorrectionTileSize = 32 / sizeof(ElementOut);

    using TMEM_LOAD = std::conditional_t<kCorrectionTileSize == 32, SM100_TMEM_LOAD_32dp32b32x, SM100_TMEM_LOAD_32dp32b16x>;  // 4x32 threads with 64 cols of 32b elem

    typename CollectiveMmaPV::TiledMma mma;
    Tensor cO   = make_identity_tensor(select<0,1>(TileShapePV{}));
    Tensor tOtO = partition_fragment_C(mma, select<0,1>(TileShapePV{}));
    Tensor tOcO = mma.get_slice(0).partition_C(cO);
    Tensor tOsO = mma.get_slice(0).partition_C(sO);

    Tensor tOtO_i = logical_divide(tOtO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOcO_i = logical_divide(tOcO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOsO_i = logical_divide(tOsO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i(make_coord(_, _), _0{}));
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);
    Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i(make_coord(_, _), _));
    Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i(make_coord(_, _), _));
    Tensor tTMEM_LOADsO = thr_tmem_load.partition_D(tOsO_i(make_coord(_, _), _));

    if constexpr (decltype(stage == _0{})::value)
    {
      tTMEM_LOADtO.data() = tTMEM_LOADtO.data().get() + uint32_t(TmemAllocation::O0);
    }
    else
    {
      static_assert(decltype(stage == _1{})::value, "stage is either 0 or 1");
      tTMEM_LOADtO.data() = tTMEM_LOADtO.data().get() + uint32_t(TmemAllocation::O1);
    }

    float2 scale_f32x2 = make_float2(scale, scale);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < get<2>(TileShape{}) / kCorrectionTileSize; i++)
    {
      Tensor tTMEM_LOADtO_i = tTMEM_LOADtO(_, _0{}, _0{}, i);
      Tensor tTMEM_LOADsO_i = tTMEM_LOADsO(_, _0{}, _0{}, i);

      Tensor tTMrO = make_tensor<ElementPV>(shape(tTMEM_LOADcO(_, _0{}, _0{}, i)));

      copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO);

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tTMrO); j += 2)
      {
        float2 in = make_float2(tTMrO(j), tTMrO(j+1));
        float2 out;
        cute::mul(out, scale_f32x2, in);
        tTMrO(j)   = out.x;
        tTMrO(j+1) = out.y;
      }

      constexpr int N = 4 / sizeof(ElementOut);
      NumericArrayConverter<ElementOut, ElementPV, N> convert;

      Tensor tSMrO = make_tensor_like<ElementOut>(tTMrO);

      Tensor tCs = recast<decltype(convert)::source_type>(tTMrO);
      Tensor tCd = recast<decltype(convert)::result_type>(tSMrO);

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tCs); j++)
      {
        tCd(j) = convert.convert(tCs(j));
      }

      Tensor tSMsO_i = recast<uint32_t>(tTMEM_LOADsO_i);
      Tensor tSMrO_i = recast<uint32_t>(tSMrO);

      copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, tSMrO_i, tSMsO_i);
    }

    cutlass::arch::fence_view_async_shared();
  }

  CUTLASS_DEVICE auto
  correction_rescale(float scale, uint32_t tmem_O)
  {
    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    // As opposed to the softmax, we do not have enough registers here
    // to load all of the values (for tile kv = 128), so we loop
    // good values would be either 32 or 64
    const int kCorrectionTileSize = 16;

    // now, just consider to half precision
    using TMEM_LOAD  = SM100_TMEM_LOAD_32dp32b16x;   // use for 4x32 threads with 64 cols of 32b elem
    using TMEM_STORE = SM100_TMEM_STORE_32dp32b16x;  // use for 4x32 threads with 64 cols of 32b elem

    typename CollectiveMmaPV::TiledMma mma;
    Tensor cO   = make_identity_tensor(select<0,1>(TileShapePV{}));
    Tensor tOtO = partition_fragment_C(mma, select<0,1>(TileShapePV{}));
    Tensor tOcO = mma.get_slice(0).partition_C(cO);
    Tensor tOtO_i = tOtO.compose(make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOcO_i = tOcO.compose(make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i);
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);
    Tensor tTMEM_LOADtO  = thr_tmem_load.partition_S(tOtO_i);
    Tensor tTMEM_LOADcO  = thr_tmem_load.partition_D(tOcO_i);
    tTMEM_LOADtO.data()  = tTMEM_LOADtO.data().get() + tmem_O;

    auto tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tOtO_i);
    auto thr_tmem_store   = tiled_tmem_store.get_slice(thread_idx);
    Tensor tTMEM_STOREtO  = thr_tmem_store.partition_D(tOtO_i);
    Tensor tTMEM_STOREcO  = thr_tmem_store.partition_S(tOcO_i);
    tTMEM_STOREtO.data()  = tTMEM_STOREtO.data().get() + tmem_O;

    static_assert(shape(tTMEM_STOREcO) == shape(tTMEM_LOADcO));

    float2 scale_f32x2 = make_float2(scale, scale);
    Tensor tTMrO       = make_tensor<ElementPV>(make_shape(shape(tTMEM_LOADcO), Int<128 / kCorrectionTileSize>{}));

    auto copy_in = [&](int i) {
      Tensor tTMEM_LOADtO_i = tTMEM_LOADtO;
      tTMEM_LOADtO_i.data() = tTMEM_LOADtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i);
    };

    auto copy_out = [&](int i) {
      Tensor tTMEM_STOREtO_i = tTMEM_STOREtO;
      tTMEM_STOREtO_i.data() = tTMEM_STOREtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i);
    };

    copy_in(0);

    int count = get<2>(TileShape{}) / kCorrectionTileSize;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < count; i++)
    {
      if (i != count - 1)
      {
        copy_in(i+1);
      }

      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tTMrO_i); j += 2)
      {
        float2 in = make_float2(tTMrO_i(j), tTMrO_i(j+1));
        float2 out;
        cute::mul(out, scale_f32x2, in);
        tTMrO_i(j)   = out.x;
        tTMrO_i(j+1) = out.y;
      }

      copy_out(i);
    }
  }


  template<
    class BlkCoord, class ProblemShape, class ParamsProblemShape,
    class TensorStorageEpi, class CollectiveEpilogue
  >
  CUTLASS_DEVICE auto
  correction(
      const BlkCoord& blk_coord,
      const Params& params, const ProblemShape & problem_shape,
      const ParamsProblemShape& params_problem_shape,
      TensorStorageEpi& shared_storage_epi,
      PipelineC& pipeline_s0_c, typename PipelineC::PipelineState& pipeline_s0_c_consumer_state,
      PipelineC& pipeline_s1_c, typename PipelineC::PipelineState& pipeline_s1_c_consumer_state,
      PipelineO& pipeline_o, typename PipelineO::PipelineState& pipeline_o_consumer_state,
      PipelineE& pipeline_epi, typename PipelineE::PipelineState& pipeline_epi_producer_state,
      CollectiveEpilogue& epilogue
  )
  {
    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);
    int thread_idx      = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    typename CollectiveMmaQK::TiledMma mma_qk;

    Tensor cS   = make_identity_tensor(select<0,1>(TileShapeQK{}));
    Tensor tScS = mma_qk.get_slice(0).partition_C(cS);
    Tensor tStS = partition_fragment_C(mma_qk, select<0,1>(TileShapeQK{}));

    Tensor tStS_v = tStS.compose(make_layout(make_shape(_128{}, _2{})));
    Tensor tScS_v = tScS.compose(make_layout(make_shape(_128{}, _2{})));

    using TMEM_LOAD_V = SM100_TMEM_LOAD_32dp32b2x;   // 4x32 threads with 2 cols of 32b elem

    auto tiled_tmem_loadv = make_tmem_copy(TMEM_LOAD_V{}, tStS_v);
    auto thr_tmem_loadv   = tiled_tmem_loadv.get_slice(thread_idx);

    Tensor tTMEM_LOADVtS = thr_tmem_loadv.partition_S(tStS_v);
    Tensor tTMEM_LOADVcS = thr_tmem_loadv.partition_D(tScS_v);
    Tensor tTMEM_LOADVtS0 = tTMEM_LOADVtS;
    tTMEM_LOADVtS0.data() = tTMEM_LOADVtS0.data().get() + uint32_t(TmemAllocation::V0);
    Tensor tTMEM_LOADVtS1 = tTMEM_LOADVtS;
    tTMEM_LOADVtS1.data() = tTMEM_LOADVtS1.data().get() + uint32_t(TmemAllocation::V1);

    // ignore first signal from softmax as no correction is required
    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);
    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

    // handle the last iteration differently (i.e. tmem_load/stsm for epi)
    mask_tile_count -= 1;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; --mask_tile_count)
    {
      pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

      Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));

      // read row_wise new global max
      copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);

      // e^(scale * (old_max - new_max)
      float scale = ::exp2f(params.scale_softmax_log2 * (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

      correction_rescale(scale, uint32_t(TmemAllocation::O0));

      pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
      ++pipeline_s1_c_consumer_state;

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;

      pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

      copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);

      scale = ::exp2f(params.scale_softmax_log2 * (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

      correction_rescale(scale, uint32_t(TmemAllocation::O1));

      pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
      ++pipeline_s0_c_consumer_state;

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;
    }

    pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
    ++pipeline_s1_c_consumer_state;
    
    // adjust the value in tmem O to correctly accumulate the current tiled result
    // the final softmax step 
    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

    // read from V0 : row_sum and final row_max
    Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
    copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);

    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;
    
    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    Tensor sO   = make_tensor(make_smem_ptr(shared_storage_epi.smem_o.data()), typename TensorStorageEpi::SmemLayoutO{});
    Tensor gLSE = make_tensor(make_gmem_ptr(epilogue.params.ptr_LSE), select<0,3>(problem_shape), epilogue.params.dLSE);
    
    correction_epilogue(params.scale_output / tTMEM_LOADVrS(kIdxFinalRowSum), _0{}, sO);

    if (epilogue.params.ptr_LSE != nullptr)
    {
      int row_idx = get<0>(tTMEM_LOADVcS(_0{})) + get<0>(TileShape{}) * get<0>(blk_coord);

      [[maybe_unused]]int row_offset = 0;

      ElementPV lse = cutlass::fast_log(tTMEM_LOADVrS(kIdxFinalRowSum)) + params.scale_softmax * tTMEM_LOADVrS(kIdxFinalRowMax);

      if (row_idx < get<0>(problem_shape))
      {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    cutlass::arch::fence_view_async_tmem_load();

    pipeline_o.consumer_release(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;

    pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

        // load from V1
    copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);

    pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
    ++pipeline_s1_c_consumer_state;

    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    correction_epilogue(params.scale_output / tTMEM_LOADVrS(kIdxFinalRowSum), _1{}, sO);

    if (epilogue.params.ptr_LSE != nullptr)
    {
      int row_idx = get<0>(tTMEM_LOADVcS(_0{})) + get<0>(TileShape{}) * get<0>(blk_coord) + get<0>(TileShapeQK{});

      ElementPV lse = cutlass::fast_log(tTMEM_LOADVrS(kIdxFinalRowSum)) + params.scale_softmax * tTMEM_LOADVrS(kIdxFinalRowMax);

      [[maybe_unused]]int row_offset = 0;

      if (row_idx < get<0>(problem_shape))
      {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    cutlass::arch::fence_view_async_tmem_load();

    pipeline_o.consumer_release(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;
  }

  template<
    class BlkCoord, class ProblemShape, class ParamsProblemShape,
    class TensorStorageEpi, class CollectiveEpilogue
  >
  CUTLASS_DEVICE auto
  correction_empty(
      const BlkCoord& blk_coord,
      const Params& params, const ProblemShape& problem_shape,
      const ParamsProblemShape& params_problem_shape,
      TensorStorageEpi& shared_storage_epi,
      PipelineE& pipeline_epi, typename PipelineE::PipelineState& pipeline_epi_producer_state,
      CollectiveEpilogue& epilogue)
  {

    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    Tensor sO   = make_tensor(make_smem_ptr(shared_storage_epi.smem_o.data()), typename TensorStorageEpi::SmemLayoutO{});
    Tensor gLSE = make_tensor(make_gmem_ptr(epilogue.params.ptr_LSE), select<0,3>(problem_shape), epilogue.params.dLSE);
    float lse = -INFINITY;
    int thread_idx = threadIdx.x % (4 * NumThreadsPerWarp);

#define DSHOW(x) print(#x ": "); print(x); print("\n")
    if (threadIdx.x % 128 == 0 && block0()) {
      DSHOW(sO);
    }
#if 1
    using ElementOut = typename CollectiveEpilogue::ElementOut;
    auto tiled_copy = make_cotiled_copy(
        Copy_Atom<UniversalCopy<uint32_t>, ElementOut>{},
        make_ordered_layout(make_shape(_128{}, Int<sizeof(uint32_t) / sizeof(ElementOut)>{}), Step<_1, _0>{}),
        sO.layout());

    auto thr_copy = tiled_copy.get_slice(thread_idx);
    auto tOgO = thr_copy.partition_D(sO);
    auto tOrO = make_tensor<ElementOut>(shape(tOgO(_,_,_,_0{})));
    clear(tOrO);
    
    copy(tiled_copy, tOrO, tOgO(_,_,_,_0{}));
#endif

    if (epilogue.params.ptr_LSE != nullptr)
    {
      int row_idx = thread_idx + get<0>(TileShape{}) * get<0>(blk_coord);

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>)
      {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      if (row_idx < get<0>(problem_shape))
      {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;

    copy(tiled_copy, tOrO, tOgO(_,_,_,_1{}));
    cutlass::arch::fence_view_async_shared();
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    if (epilogue.params.ptr_LSE != nullptr)
    {
      int row_idx = thread_idx + get<0>(TileShape{}) * get<0>(blk_coord) + get<0>(TileShapeQK{});

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>)
      {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      if (row_idx < get<0>(problem_shape))
      {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    cutlass::arch::fence_view_async_shared();
    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;
  }

};

///////////////////////////////////////////////////////////////////////////

} // namespace cutlass::fmha::collective