/************************************************************************************************************
 * Common definitions for FMHA
 ************************************************************************************************************/
#pragma once

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cute/tensor.hpp"

namespace cutlass::fmha::collective
{
using namespace cute;

//////////////////////////////////////////////////////////////////////

template<typename Atom, typename TA, typename TB, typename TC>
CUTE_DEVICE void 
gemm_reset_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) 
{
  constexpr int rA = decltype(rank(tA))::value;
  constexpr int rB = decltype(rank(tB))::value;
  constexpr int rC = decltype(rank(tC))::value;
  static_assert(rA == 3 && rB == 3 && rC == 3);

  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < size<2>(tA); k_block++)
  {
    cute::gemm(atom, tA(_,_,k_block), tB(_,_,k_block), tC);
    atom.accumulate_ = decltype(atom.accumulate_)::One;
  }
}

template<typename Atom, typename TA, typename TB, typename TC>
CUTE_DEVICE void 
gemm_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC)
{
  atom.accumulate_ = decltype(atom.accumulate_)::Zero;
  gemm_reset_zero_acc(atom, tA, tB, tC);
}

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

template<class Layout, class Stages = _1>
CUTE_DEVICE 
constexpr auto 
unstageSmemLayout(Layout const& layout, Stages stages = {}) 
{
    return composition(layout, prepend<decltype(rank(layout))::value>(make_layout(stages), _));
}


template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, class... TAs, class... TMs>
CUTE_HOST_DEVICE constexpr
auto
to_tiled_mma_sm100_ts(
    TiledMMA<MMA_Atom<
      MMA_Traits<SM100_MMA_F8F6F4_SS, a_type, b_type, c_type,
                    cute::C<M>, cute::C<N>,
                    cute::integral_constant<UMMA::Major, a_major>,
                    cute::integral_constant<UMMA::Major, b_major>,
                    cute::integral_constant<UMMA::ScaleIn, a_neg>,
                    cute::integral_constant<UMMA::ScaleIn, b_neg>>,
      TAs...>, TMs...>)
{
  return TiledMMA<MMA_Atom<
    MMA_Traits<SM100_MMA_F8F6F4_TS<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, UMMA::Saturate::False>>,
    TAs...>, TMs...>{};
}

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, class... TAs, class... TMs>
CUTE_HOST_DEVICE constexpr
auto
to_tiled_mma_sm100_ts(
    TiledMMA<MMA_Atom<
      SM100_MMA_F16BF16_SS<a_type, b_type, c_type,
                    M, N,
                    a_major,
                    b_major,
                    a_neg,
                    b_neg>,
      TAs...>, TMs...>)
{
  return TiledMMA<MMA_Atom<
    SM100_MMA_F16BF16_TS<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, UMMA::Saturate::False>,
    TAs...>, TMs...>{};
}

//////////////////////////////////////////////////////////////////////
    
}