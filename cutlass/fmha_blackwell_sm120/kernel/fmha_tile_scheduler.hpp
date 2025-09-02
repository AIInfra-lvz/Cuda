/***************************************************************************************************
 * FMHA Tile Scheduler
 ***************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/fast_math.h"

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

////////////////////////////////////////////////////////////////////

struct PersistentTileScheduler
{
    struct Params
    {
        int num_blocks;
        FastDivmod divmod_m_block;
        FastDivmod divmod_h;
        FastDivmod divmod_b;

        KernelHardwareInfo hw_info;
    };

    int block_idx = 0;
    Params params;

    CUTLASS_DEVICE
    PersistentTileScheduler(Params const& params) : block_idx(blockIdx.x), params(params) {}

    template<class ProblemSize, class ClusterShape, class TileShape>
    static Params to_underlying_arguments(
      const ProblemSize& problem_size, KernelHardwareInfo hw_info,
      const ClusterShape& cluster_shape, const TileShape& tile_shape)
    {
        // Get SM count if needed, otherwise use user supplied SM count
        int sm_count = hw_info.sm_count;

        if (sm_count <= 0) 
        {
            CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
              "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
       }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
        hw_info.sm_count = sm_count;
        
        int num_m_blocks = cutlass::round_up(ceil_div(size<0>(problem_size), size<0>(tile_shape)), size<0>(cluster_shape));
        int num_blocks   = num_m_blocks * size<3,0>(problem_size) * size<3,1>(problem_size);

        return Params {
            num_blocks,
            { num_m_blocks}, { size<3,0>(problem_size) }, { size<3,1>(problem_size) },
            hw_info
        };
    }

    static dim3 get_grid_shape(Params const& params)
    {
        dim3 grid(std::min(params.num_blocks, params.hw_info.sm_count), 1, 1);
        return grid;
    }

    CUTLASS_DEVICE
    bool is_valid()
    {
        return block_idx < params.num_blocks;
    }

    CUTLASS_DEVICE
    auto get_block_coord()
    {
        using namespace cute;
        int block_decode = block_idx;
        int m_block, bidb, bidh;
        params.divmod_m_block(block_decode, m_block, block_decode);
        params.divmod_b(block_decode, bidb, block_decode);
        params.divmod_h(block_decode, bidh, block_decode);
        return make_coord(m_block, _0{}, make_coord(bidh, bidb));
    }

    CUTLASS_DEVICE
    PersistentTileScheduler& operator++()
    {
        block_idx += gridDim.x;
        return *this;
    }

};

/////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::fmha::kernel