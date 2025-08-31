/***************************************************************************************************
 * FMHA Tile Scheduler
 ***************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/kernel_hardware_info.h"

namespace cutlass::fmha::kernel
{
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////

struct IndividualTileScheduler
{
    struct Params 
    {
        dim3 grid;
    };

    CUTLASS_DEVICE IndividualTileScheduler(const Params&) {;}

    bool valid_ = true;

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

    static dim3 get_grid_shape(Params const& params){ return params.grid; }

    CUTLASS_DEVICE bool is_valid() { return valid_; }

    CUTLASS_DEVICE auto get_block_coord() 
    {
        return make_coord(blockIdx.x, _0{}, make_coord(blockIdx.y, blockIdx.z));
    }

    CUTLASS_DEVICE
    IndividualTileScheduler& operator++() 
    {
        valid_ = false;
        return *this;
    }
};



/////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::fmha::kernel