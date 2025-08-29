/******************************************************************************************************
 * Epilogue for the forward pass of the FMHA kernel
 ******************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cute/layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

namespace cutlass::fmha::collective
{

////////////////////////////////////////////////////////////////////

template<
  class Element,
  class ElementAcc,
  class TileShape,  // Q, D, _
  class StrideO,    // Q, D, B
  class StrideLSE_  // Q, B
>
struct Sm120FmhaFwdEpilogueTmaWarpspecialized
{
  using Pipeline = cutlass::PipelineAsync<2>;
  using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
        cute::UMMA::Major::K, Element, tuple_element_t<0, TileShape>, tuple_element_t<1, TileShape>>());

  using SmemLayoutO_ = decltype(tile_to_shape(SmemLayoutAtomO{}, replace<2>(TileShape{}, _2{}), Step<_2, _1, _3>{}));
  using SmemLayoutO = SmemLayoutO_;
  using StrideLSE = StrideLSE_;
  using ElementOut = Element;

  static const int NumWarpsEpilogue = 1;
  static const int NumWarpsLoad = 1;

  struct TensorStorage
  {
    using SmemLayoutO = SmemLayoutO_;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>> smem_o;
  };

  struct Arguments
  {
    Element* ptr_O;
    StrideO dO;

    ElementAcc* ptr_LSE;
    StrideLSE dLSE;
  };

  using TMA_O = decltype(make_tma_copy(
    SM90_TMA_STORE{},
    make_tensor((Element*) nullptr, repeat_like(StrideO{}, 0), StrideO{}),
    SmemLayoutO{}(_,_,_0{})
  ));

  struct Params 
  {
    TMA_O tma_store_o;

    ElementAcc* ptr_LSE;
    StrideLSE dLSE;
  };

  template<class ProblemShape>
  static Params to_underlying_arguments(
      const ProblemShape& problem_shape,
      const Arguments& args,
      void* workspace = nullptr)
  {
    auto ptr_O = args.ptr_O;
    StrideO dO = args.dO;
    auto problem_shape_O = select<0,2,3>(problem_shape);

    auto tma_store_o = make_tma_copy(
      SM90_TMA_STORE{},
      make_tensor(ptr_O, problem_shape_O, dO),
      SmemLayoutO{}(_,_,_0{})
    );

    return {
      tma_store_o,
      args.ptr_LSE,
      args.dLSE
    };
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params)
  {
    cute::prefetch_tma_descriptor(params.tma_store_o.get_tma_descriptor());
  }

  const Params& params;

  CUTLASS_DEVICE Sm120FmhaFwdEpilogueTmaWarpspecialized(const Params& params) : params(params) {}

  template<class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE auto
  store(
      const BlkCoord& blk_coord_in, const ProblemShape& problem_shape,
      const Params& params, const ParamsProblemShape& params_problem_shape,
      TensorStorage& shared_storage,
      Pipeline& pipeline, typename Pipeline::PipelineState& pipeline_consumer_state)
  {
    BlkCoord blk_coord = blk_coord_in;
    uint32_t lane_predicate = cute::elect_one_sync();

    using X = Underscore;

    int o0_index = 2 * get<0>(blk_coord);
    int o1_index = 2 * get<0>(blk_coord) + 1;

    Tensor mO_qdl_p = params.tma_store_o.get_tma_tensor(select<0,2,3>(problem_shape));
    // offset mode 0 by (max_length - real_length)
    // offset mode 3,1 by cumulative_length + real_length
    // the ptr is already offset by - max_length
    // so in total this achieves 
    [[maybe_unused]]int offs_0   = 0;
    [[maybe_unused]]int offs_2_1 = 0;

    Tensor mO_qdl = domain_offset(make_coord(offs_0, _0{}, make_coord(_0{}, offs_2_1)), mO_qdl_p);

    Tensor gO_qdl = local_tile(mO_qdl, TileShape{}, make_coord(_, _, _), Step<_1, _1, X>{});
    Tensor gO = gO_qdl(_, _, _, _0{}, get<2>(blk_coord));
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});
    auto block_tma = params.tma_store_o.get_slice(0);
    Tensor tOsO = block_tma.partition_S(sO);
    Tensor tOgO = block_tma.partition_D(gO);

    auto pipeline_release_state = pipeline_consumer_state;

    // O1 O2
    // one pipeline: O
    // wait from corr, issue tma store on smem
    pipeline.consumer_wait(pipeline_consumer_state);
    ++pipeline_consumer_state;

    if (lane_predicate)
    {
      copy(params.tma_store_o, tOsO(_,_,_,_0{}), tOgO(_,_,_,o0_index));
    }
    tma_store_arrive();          // commit store op to the sync queue

    pipeline.consumer_wait(pipeline_consumer_state);
    ++pipeline_consumer_state;

    if (lane_predicate)
    {
      copy(params.tma_store_o, tOsO(_,_,_,_1{}), tOgO(_,_,_,o1_index));
    }
    tma_store_arrive();

    tma_store_wait<1>();         // allow at most 1 outstanding tma op

    pipeline.consumer_release(pipeline_release_state);
    ++pipeline_release_state;

    tma_store_wait<0>();

    pipeline.consumer_release(pipeline_release_state);
    ++pipeline_release_state;
  }   

};

//////////////////////////////////////////////////////////

} // namespace cutlass::fmha::collective

