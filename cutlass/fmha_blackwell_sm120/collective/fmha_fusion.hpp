/***************************************************************************************************
 * FMHA Fusion
 ***************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

namespace cutlass::fmha::collective 
{
using namespace cute;

struct VariableLength
{
  int max_length;
  int* cumulative_length = nullptr;
  int total_length = -1;

  CUTE_HOST_DEVICE operator int() const { return max_length; }
};

template<class T> struct is_variable_length_impl : std::false_type {};
template<> struct is_variable_length_impl<VariableLength> : std::true_type {};
template<class T> constexpr bool is_variable_length_v = is_variable_length_impl<remove_cvref_t<T>>::value;

template<class Shape, class Idx>
CUTE_HOST_DEVICE
constexpr auto
apply_variable_length(const Shape& shape, const Idx& idx)
{
  return transform_leaf(shape, [&](auto const& s)
  {
    if constexpr (is_variable_length_v<decltype(s)>) 
    {
      return s.cumulative_length[idx+1] - s.cumulative_length[idx];
    }
    else 
    {
      return s;
    }
  });
}

template<class Shape, class Coord, class Idx>
CUTE_HOST_DEVICE
constexpr auto
apply_variable_length(const Shape& shape, const Coord& coord, const Idx& idx)
{
  auto new_shape = apply_variable_length(shape, idx);
  auto new_coord = transform_leaf(shape, coord, [&](const auto& s, const auto& c) {
    if constexpr (is_variable_length_v<decltype(s)>)
    {
      return cute::make_tuple(c, s.cumulative_length[idx]);
    }
    else
    {
      return c;
    }
  });
  return cute::make_tuple(new_shape, new_coord);
}

struct NoMask
{
  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int
  get_trip_count(
      const BlkCoord& blk_coord,
      const TileShape& tile_shape,
      const ProblemSize& problem_size)
  {
    return ceil_div(get<1>(problem_size), get<1>(tile_shape));
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int 
  get_masked_trip_count(
      const BlkCoord& blk_coord,
      const TileShape& tile_shape,
      const ProblemSize& problem_size)
  {
    return 0;
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int 
  get_unmasked_trip_count(
      const BlkCoord& blk_coord,
      const TileShape& tile_shape,
      const ProblemSize& problem_size)
  {
    return get_trip_count(blk_coord, tile_shape, problem_size);
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE void
  apply_mask(
      AccQK& acc_qk,
      const IndexQK& index_qk,
      const ProblemSize& problem_size)
  {
    return;
  }
};

struct ResidualMask : NoMask
{
  using Base = NoMask;

  template <class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int 
  get_masked_trip_count(
      const BlkCoord& blk_coord,
      const TileShape& tile_shape,
      const ProblemSize& problem_size)
  {
    if (get<1>(problem_size) % get<1>(tile_shape) != 0)
    {
      return 1;
    }

    return 0;
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int 
  get_unmasked_trip_count(
      const BlkCoord& blk_coord,
      const TileShape& tile_shape,
      const ProblemSize& problem_size)
  {
    // if the sequence length does not divide the tile size evenly
    if (get<1>(problem_size) % get<1>(tile_shape) != 0)
    {
      return get_trip_count(blk_coord, tile_shape, problem_size) - 1;
    }

    return get_trip_count(blk_coord, tile_shape, problem_size);
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE void 
  apply_mask(
      AccQK& acc_qk,
      const IndexQK& index_qk,
      const ProblemSize& problem_size)
  {
    // This is useful is seqlen_k % kBlockN != 0 since it masks
    // the remaining elements out from softmax.
    // d % kHeadDim != 0 or seqlen_q % kBlockM do not suffer from similar
    // issues as they are transparently taken care of by TMA and the
    // epilogue, if it is instantiated with predication support.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++)
    {
      auto pos = index_qk(i);

      if (get<1>(pos) >= get<1>(problem_size))
      {
        acc_qk(i) = -INFINITY;
      }
    }
  }
};

// There are two ways to do causal if N_Q != N_K
// (1) The Q is at the beginning of the matrix
// (2) The Q is at the end of the matrix
template<bool kIsQBegin = true>
struct CausalMask : NoMask
{
  using Base = NoMask;

  static constexpr bool IsQBegin = kIsQBegin;

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int 
  get_trip_count(
      const BlkCoord& blk_coord,
      const TileShape& tile_shape,
      const ProblemSize& problem_size)
  {
    // See note below on different ways to think about causal attention
    // Again, we'd add the offset_q into the max_blocks_q calculation
    int max_blocks_k = Base::get_trip_count(blk_coord, tile_shape, problem_size);

    if constexpr (IsQBegin)
    {
      int max_blocks_q = ceil_div((get<0>(blk_coord) + 1) * get<0>(tile_shape), get<1>(tile_shape));
      return std::min(max_blocks_k, max_blocks_q);
    } 
    else
    {
      const int offset_q = get<1>(problem_size) - get<0>(problem_size);
      int max_blocks_q   = ceil_div((get<0>(blk_coord) + 1) * get<0>(tile_shape) + offset_q, get<1>(tile_shape));
      return std::min(max_blocks_k, max_blocks_q);
    }
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int 
  get_masked_trip_count(
      const BlkCoord& blk_coord,
      const TileShape& tile_shape,
      const ProblemSize& problem_size)
  {
    int trip_count = get_trip_count(blk_coord, tile_shape, problem_size);

    if constexpr (IsQBegin)
    {
      return std::min(trip_count, int(ceil_div(size<0>(tile_shape), size<1>(tile_shape))));
    }
    else
    {
      const int offset_tile_q = get<1>(problem_size) % get<1>(tile_shape);
      return std::min(trip_count, int(ceil_div(get<0>(tile_shape) + offset_tile_q, get<1>(tile_shape))));
    }
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE int 
  get_unmasked_trip_count(
      const BlkCoord& blk_coord,
      const TileShape& tile_shape,
      const ProblemSize& problem_size)
  {
    return get_trip_count(blk_coord, tile_shape, problem_size) - get_masked_trip_count(blk_coord, tile_shape, problem_size);
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE void
  apply_mask(
      AccQK& acc_qk,
      const IndexQK& index_qk,
      const ProblemSize& problem_size)
  {
    // There are two ways to do causal if N_Q != N_K
    // (1) is to assume that the Q is at the beginning of the matrix
    //    - this is the default setting.
    // (2) is that it is at the end of the matrix
    //    - this is usually what we want for inference settings
    //      where we only compute the next row and use cache for the rest
    //    - if you'd like this, you only need to set kIsQBegin=false

    if constexpr (IsQBegin)
    {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(acc_qk); i++)
      {
        auto pos = index_qk(i);
        if ((get<0>(pos) < get<1>(pos)) || (get<1>(pos) >= get<1>(problem_size)))
        {
          acc_qk(i) = -INFINITY;
        }
      }
    } 
    else 
    {
      const auto offset_q = get<1>(problem_size) - get<0>(problem_size);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(acc_qk); i++)
      {
        auto pos = index_qk(i);

        if ((get<0>(pos) + offset_q < get<1>(pos)) || (get<1>(pos) >= get<1>(problem_size)))
        {
          acc_qk(i) = -INFINITY;
        }
      }
    }
  }

};

} // namespace cutlass::fmha::collective