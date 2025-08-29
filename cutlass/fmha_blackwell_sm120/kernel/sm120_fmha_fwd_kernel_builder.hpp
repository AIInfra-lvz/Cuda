/***************************************************************************************************
 * forward collective mainloop builder for TMA warp-specialized FMHA
 ***************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"

#include "collective/fmha_common.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm120_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm120_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "kernel/sm120_fmha_fwd_kernel_tma_warpspecialized.hpp"

namespace cutlass::fmha::kernel
{
///////////////////////////////////////////////////////

template<
  class Element_,
  class ElementQK_,
  class ElementPV_,
  class StrideQ_,
  class StrideK_,
  class StrideV_,
  class StrideO_,     // Q, D, B
  class StrideLSE_,   // Q, B
  class TileShape_,
  class ProblemShape_,
  typename IsCausalMask  = std::true_type,
  typename kIsPersistent = std::false_type
>
struct Sm120FmhaFwdBuilder
{
  using Element   = Element_;
  using ElementQK = ElementQK_;
  using ElementPV = ElementPV_;
  using TileShape = TileShape_;
  using StrideQ   = StrideQ_;
  using StrideK   = StrideK_;
  using StrideV   = StrideV_;
  using StrideO   = StrideO_;
  using StrideLSE = StrideLSE_;
  using ProblemShape = ProblemShape_;

  using Mask = std::conditional_t<IsCausalMask::value, CausalMask<true>, ResidualMask>;

  using CollectiveMainloop = Sm120FwdMainloopTmaWarpspecialized<
      Element, 
      ElementQK, ElementPV,
      TileShape, 
      StrideQ, StrideK, StrideV,
      Mask>;

  using CollectiveEpilogue = Sm120FmhaFwdEpilogueTmaWarpspecialized<
      Element, 
      ElementPV,
      typename CollectiveMainloop::TileShapePV, 
      StrideO, StrideLSE>;

  using TileScheduler = cutlass::fmha::kernel::IndividualTileScheduler;

  using sm120_fmha_kernel = Sm120FmhaFwdKernelTmaWarpspecialized<
      ProblemShape, 
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler>;

};

//////////////////////////////////////////////////////

} // namespace cutlass::fmha::kernel