/********************************************************************************************
 * Load TMA warp-specialized FMHA
 ********************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

#include "collective/fmha_fusion.hpp"

namespace cutlass::fmha::collective
{
using namespace cute;

//////////////////////////////////////////////////////////////////

template<
  class Element,
  class StrideQ,
  class StrideK,
  class StrideV,
  class CollectiveMmaQK,
  class CollectiveMmaPV,
  class SmemLayoutQ,
  class SmemLayoutK,
  class SmemLayoutV,
  class TensorStorage,
  class PipelineQ,
  class PipelineKV,
  class Mask,
  class TileShape
>
class Sm120FmhaLoadTmaWarpspecialized
{
public:

  using TileShapeQK = typename CollectiveMmaQK::TileShape;
  using TileShapePV = typename CollectiveMmaPV::TileShape;
  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
  using TMA_K = typename CollectiveMmaQK::Params::TMA_B;
  using TMA_V = typename CollectiveMmaPV::Params::TMA_B;

  struct Arguments
  {
    const Element* ptr_Q;
    StrideQ dQ;
    const Element* ptr_K;
    StrideK dK;
    const Element* ptr_V;
    StrideV dV;
  };

  struct Params
  {
    TMA_Q tma_load_q;
    TMA_K tma_load_k;
    TMA_V tma_load_v;
  };

  template<class ProblemShape>
  static Params to_underlying_arguments(
      const ProblemShape& problem_shape,
      const Arguments& args,
      void* workspace)
  {

    auto ptr_Q = args.ptr_Q;
    auto ptr_K = args.ptr_K;
    auto ptr_V = args.ptr_V;
    auto dQ    = args.dQ;
    auto dK    = args.dK;
    auto dV    = args.dV;
    auto problem_shape_qk = problem_shape;

    auto params_qk = CollectiveMmaQK::to_underlying_arguments(
        problem_shape_qk,
        typename CollectiveMmaQK::Arguments{
            ptr_Q, dQ,
            ptr_K, dK,
        }, 
        /*workspace=*/ nullptr);

    auto problem_shape_pv = select<0,2,1,3>(problem_shape_qk);
    auto params_pv = CollectiveMmaPV::to_underlying_arguments(
        problem_shape_pv,
        typename CollectiveMmaPV::Arguments {
            ptr_K, dK,  // never used, dummy
            ptr_V, select<1,0,2>(dV),
        }, 
        /*workspace=*/ nullptr);

    return Params{
        params_qk.tma_load_a,
        params_qk.tma_load_b,
        params_pv.tma_load_b
    };
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(const Params& params)
  {
    cute::prefetch_tma_descriptor(params.tma_load_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_k.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_v.get_tma_descriptor());
  }

  template<class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void
  load(
      const BlkCoord& blk_coord_in, const ProblemShape& problem_shape,
      const Params& params, const ParamsProblemShape& params_problem_shape_in,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_producer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_producer_state)
  {
    using X = Underscore;

    BlkCoord blk_coord_q  = blk_coord_in;
    BlkCoord blk_coord_kv = blk_coord_in;
    [[maybe_unused]] ParamsProblemShape params_problem_shape = params_problem_shape_in;      // temporarily don't consider variable length sequence

    int mask_tile_count = Mask{}.get_trip_count(blk_coord_in, TileShape{}, problem_shape);

    ThrMMA mma_qk = typename CollectiveMmaQK::TiledMma{}.get_slice(0);    // only one CTA in group if multicast is not used
    Tensor mQ_qdl_p = params.tma_load_q.get_tma_tensor(select<0,2,3>(problem_shape));

    // Offsets for variable length sequences are not handled yet, so they are temporarily ignored
    int q_offs_0   = 0;
    int q_offs_2_1 = 0;
    Tensor mQ_qdl = domain_offset(make_coord(q_offs_0, _0{}, make_coord(_0{}, q_offs_2_1)), mQ_qdl_p);

    Tensor gQ_qdl   = local_tile(mQ_qdl, TileShapeQK{}, make_coord(_, _, _), Step<_1, X, _1>{});
    Tensor tSgQ_qdl = mma_qk.partition_A(gQ_qdl);
    Tensor sQ       = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});

    auto [tQgQ_qdl, tQsQ] = tma_partition(
      params.tma_load_q, /*not use mulitcast*/ _0{}, /*not use mulitcast*/ make_layout(_1{}),
      group_modes<0,3>(sQ), group_modes<0,3>(tSgQ_qdl)
    );
    Tensor tQgQ = tQgQ_qdl(_, _, _0{}, get<2>(blk_coord_q));

    // compute gK, sK
    Tensor mK_kdl_p = params.tma_load_k.get_tma_tensor(select<1,2,3>(problem_shape));

    int kv_offs_0   = 0;
    int kv_offs_2_1 = 0;

    Tensor mK_kdl = domain_offset(make_coord(kv_offs_0, _0{}, make_coord(_0{}, kv_offs_2_1)), mK_kdl_p);

    Tensor gK_kdl   = local_tile(mK_kdl, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor tSgK_kdl = mma_qk.partition_B(gK_kdl);
    Tensor sK       = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    auto [tKgK_kdl, tKsK] = tma_partition(
      params.tma_load_k, _0{}, make_layout(_1{}),
      group_modes<0,3>(sK), group_modes<0,3>(tSgK_kdl)
    );
    Tensor tKgK = tKgK_kdl(_, _, _0{}, get<2>(blk_coord_kv));

    // compute gV, sV
    ThrMMA mma_pv = typename CollectiveMmaPV::TiledMma{}.get_slice(0);
    Tensor mV_dkl_p = params.tma_load_v.get_tma_tensor(select<2,1,3>(problem_shape));

    Tensor mV_dkl = domain_offset(make_coord(_0{}, kv_offs_0, make_coord(_0{}, kv_offs_2_1)), mV_dkl_p);

    Tensor gV_dkl   = local_tile(mV_dkl, TileShapePV{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor tOgV_dkl = mma_pv.partition_B(gV_dkl);
    Tensor sV       = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});
    auto [tVgV_dkl, tVsV] = tma_partition(
      params.tma_load_v, _0{}, make_layout(_1{}),
      group_modes<0,3>(sV), group_modes<0,3>(tOgV_dkl)
    );
    auto tVgV = tVgV_dkl(_, _0{}, _, get<2>(blk_coord_kv));

    uint32_t lane_predicate = cute::elect_one_sync();

    // Q0
    int q0_index = 2 * get<0>(blk_coord_q);
    int q1_index = 2 * get<0>(blk_coord_q) + 1;

    pipeline_q.producer_acquire(pipeline_q_producer_state);
    if (lane_predicate)
    {
      auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      copy(params.tma_load_q.with(*tma_barrier, 0), tQgQ(_, q0_index), tQsQ(_, pipeline_q_producer_state.index()));
    }
    ++pipeline_q_producer_state;

    // K0
    int k_index = 0;
    pipeline_kv.producer_acquire(pipeline_kv_producer_state);
    if (lane_predicate)
    {
      auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
      copy(params.tma_load_k.with(*tma_barrier, 0), tKgK(_, k_index), tKsK(_, pipeline_kv_producer_state.index()));
    }
    ++pipeline_kv_producer_state;

    // Q1
    pipeline_q.producer_acquire(pipeline_q_producer_state);
    if (lane_predicate)
    {
      auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      copy(params.tma_load_q.with(*tma_barrier, 0), tQgQ(_, q1_index), tQsQ(_, pipeline_q_producer_state.index()));
    }
    ++pipeline_q_producer_state;

    // V0
    pipeline_kv.producer_acquire(pipeline_kv_producer_state);
    if (lane_predicate)
    {
      auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
      copy(params.tma_load_v.with(*tma_barrier, 0), tVgV(_, k_index), tVsV(_, pipeline_kv_producer_state.index()));
    }
    ++pipeline_kv_producer_state;
    ++k_index;

    // loop
    mask_tile_count -= 1;

    for (; mask_tile_count > 0; mask_tile_count -= 1)
    {
      // Ki
      pipeline_kv.producer_acquire(pipeline_kv_producer_state);
      if (lane_predicate)
      {
        auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
        copy(params.tma_load_k.with(*tma_barrier, 0), tKgK(_, k_index), tKsK(_, pipeline_kv_producer_state.index()));
      }
      ++pipeline_kv_producer_state;

      // Vi
      pipeline_kv.producer_acquire(pipeline_kv_producer_state);
      if (lane_predicate)
      {
        auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
        copy(params.tma_load_v.with(*tma_barrier, 0), tVgV(_, k_index), tVsV(_, pipeline_kv_producer_state.index()));
      }
      ++pipeline_kv_producer_state;
      k_index += 1;
    }
  }

};

//////////////////////////////////////////////////////////////////////

} // namespace cutlass::fmha::collective
