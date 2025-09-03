/***************************************************************************************************
 * Forward kernel for TMA warp-specialized FMHA
 ***************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cute/layout.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"


#include "collective/fmha_fusion.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "collective/fmha_common.hpp"

namespace cutlass::fmha::kernel 
{
using namespace cute;
using namespace cutlass::fmha::collective;

////////////////////////////////////////////////////////////

template<
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
struct Sm120FmhaFwdKernelTmaWarpspecialized
{
  using ProblemShape = ProblemShape_;
  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  struct Sm120FmhaCtxKernelWarpspecializedSchedule
  {

    enum class WarpRole
    {
      Softmax0,
      Softmax1,
      Correction,
      MMA,
      Load,
      Epilogue,
      Empty
    };

    static constexpr WarpRole warp_idx_to_WarpRole(int warp_idx)
    {
      int wg_idx = warp_idx / 4;                        // warp_idx
      if (wg_idx == 0) return WarpRole::Softmax0;       //   0 -  3
      if (wg_idx == 1) return WarpRole::Softmax1;       //   4 -  7
      if (wg_idx == 2) return WarpRole::Correction;     //   8 - 11
      if (warp_idx == 12) return WarpRole::MMA;         //       12
      if (warp_idx == 13) return WarpRole::Load;        //       13
      if (warp_idx == 14) return WarpRole::Epilogue;    //       14
      return WarpRole::Empty;                           //       15
    }

    static const int NumWarpsSoftmax = 4;
    static const int NumWarpsCorrection = 4;
    static const int NumWarpsEpilogue = 1;
    static const int NumWarpsLoad = 1;

    static const bool kDebugUsingPrintf = false;
    static const int NumRegsSoftmax = 184;
    static const int NumRegsCorrection = 96 - (kDebugUsingPrintf ? 16 : 0);
    static const int NumRegsOther = 48 + (kDebugUsingPrintf ? 16 : 0);
    static const int NumRegsEmpty = 24;

    static const int NumWarps = 16;
  };

  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileScheduler      = TileScheduler_;
  using TileShape          = typename CollectiveMainloop::TileShape;
  using KernelSchedule     = Sm120FmhaCtxKernelWarpspecializedSchedule;
  using ClusterShape       = typename CollectiveMainloop::ClusterShape;
  using TmemAllocator      = cute::TMEM::Allocator1Sm;
  using ArchTag            = cutlass::arch::Sm120;

  struct SharedStorage
  {
    typename CollectiveMainloop::TensorStorage mainloop;
    typename CollectiveEpilogue::TensorStorage epilogue;

    struct PipelineStorage
    {
      alignas(16) typename CollectiveMainloop::PipelineQ::SharedStorage  load_q;
      alignas(16) typename CollectiveMainloop::PipelineKV::SharedStorage load_kv;
      alignas(16) typename CollectiveMainloop::PipelineS::SharedStorage  mma_s0;
      alignas(16) typename CollectiveMainloop::PipelineS::SharedStorage  mma_s1;
      alignas(16) typename CollectiveMainloop::PipelineC::SharedStorage  s0_corr;
      alignas(16) typename CollectiveMainloop::PipelineC::SharedStorage  s1_corr;
      alignas(16) typename CollectiveMainloop::PipelineO::SharedStorage  mma_corr;
      alignas(16) typename CollectiveMainloop::PipelineE::SharedStorage  corr_epi;
      alignas(16) typename CollectiveMainloop::OrderBarrierSoftmax::SharedStorage order_s01;
    } pipelines;

    uint32_t tmem_base_ptr;    // for recording TMEM address
  };

  struct Arguments 
  {
    ProblemShape problem_shape;
    typename CollectiveMainloop::Arguments mainloop;
    typename CollectiveEpilogue::Arguments epilogue;
    cutlass::KernelHardwareInfo hw_info;
  };
  
  struct Params
  {
    ProblemShape problem_shape;
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
    typename TileScheduler::Params tile_scheduler;
  };

  static const int NumWarpsSoftmax    = KernelSchedule::NumWarpsSoftmax;
  static const int NumWarpsCorrection = KernelSchedule::NumWarpsCorrection;
  static const int NumWarpsEpilogue   = KernelSchedule::NumWarpsEpilogue;
  static const int NumWarpsLoad       = KernelSchedule::NumWarpsLoad;

  static const int NumRegsSoftmax     = KernelSchedule::NumRegsSoftmax;
  static const int NumRegsCorrection  = KernelSchedule::NumRegsCorrection;
  static const int NumRegsOther       = KernelSchedule::NumRegsOther;
  static const int NumRegsEmpty       = 24;                                  // Now, why 24 that is still confused
  static const int NumWarps           = KernelSchedule::NumWarps;

  static_assert(NumWarpsEpilogue == CollectiveEpilogue::NumWarpsEpilogue);
  static_assert(NumWarpsLoad == CollectiveEpilogue::NumWarpsLoad);

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static const int MaxThreadsPerBlock = NumWarps * cutlass::NumThreadsPerWarp;
  static const int MinBlocksPerMultiprocessor = 1;

  constexpr CUTLASS_HOST_DEVICE
  typename KernelSchedule::WarpRole 
  warp_idx_to_WarpRole(int warp_idx)
  {
    return KernelSchedule::warp_idx_to_WarpRole(warp_idx);
  }

  static size_t get_workspace_size(const Arguments& args) { return 0; }
  static cutlass::Status initialize_workspace(const Arguments& args, /* void* workspace */void*, cudaStream_t)
  {
    return cutlass::Status::kSuccess;
  }

  static bool can_implement(const Arguments& args)
  {
    return CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
  }

  static dim3 get_grid_shape(const Params& params)
  {
    return TileScheduler::get_grid_shape(params.tile_scheduler);
  }

  static dim3 get_block_shape()
  {
    dim3 block(MaxThreadsPerBlock, 1, 1);
    return block;
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace)
  {
    return Params{
        args.problem_shape,
        CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(args.problem_shape, args.hw_info, ClusterShape{}, TileShape{})
      };
  }

  CUTLASS_DEVICE auto apply_batch(const Params &params, ProblemShape const& problem_shape, int batch_idx) 
  {
    return apply_variable_length(params.problem_shape, batch_idx);
  }

  CUTLASS_DEVICE void operator()(const Params &params, char* smem)
  {
    TileScheduler tile_scheduler{params.tile_scheduler};

    int warp_idx = cutlass::canonical_warp_idx_sync();
    auto role    = warp_idx_to_WarpRole(warp_idx);
    uint32_t lane_predicate = cute::elect_one_sync();

    if (role == KernelSchedule::WarpRole::Load && lane_predicate)
    {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }

    if (role == KernelSchedule::WarpRole::Epilogue && lane_predicate)
    {
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    typename CollectiveMainloop::PipelineQ::Params pipeline_load_q_params;
    if (role == KernelSchedule::WarpRole::Load)
    {
      pipeline_load_q_params.role = CollectiveMainloop::PipelineQ::ThreadCategory::Producer;
    }
    if (role == KernelSchedule::WarpRole::MMA)
    {
      pipeline_load_q_params.role = CollectiveMainloop::PipelineQ::ThreadCategory::Consumer;
    }
    pipeline_load_q_params.is_leader         = lane_predicate && (role == KernelSchedule::WarpRole::Load);
    pipeline_load_q_params.transaction_bytes = CollectiveMainloop::TransactionBytesLoadQ;
    typename CollectiveMainloop::PipelineQ pipeline_load_q(
      shared_storage.pipelines.load_q,
      pipeline_load_q_params,
      ClusterShape{},  cute::true_type{}, /*mask calc*/cute::false_type{}
    );

    typename CollectiveMainloop::PipelineKV::Params pipeline_load_kv_params;
    if (role == KernelSchedule::WarpRole::Load)
    {
      pipeline_load_kv_params.role = CollectiveMainloop::PipelineKV::ThreadCategory::Producer;
    }
    if (role == KernelSchedule::WarpRole::MMA)
    {
      pipeline_load_kv_params.role = CollectiveMainloop::PipelineKV::ThreadCategory::Consumer;
    }
    pipeline_load_kv_params.is_leader         = lane_predicate && (role == KernelSchedule::WarpRole::Load);
    pipeline_load_kv_params.transaction_bytes = CollectiveMainloop::TransactionBytesLoadK;
    typename CollectiveMainloop::PipelineKV pipeline_load_kv(
      shared_storage.pipelines.load_kv,
      pipeline_load_kv_params,
      ClusterShape{},  cute::true_type{}, /*mask calc*/cute::false_type{}
    );

    typename CollectiveMainloop::PipelineS::Params pipeline_mma_s0_params;
    if (role == KernelSchedule::WarpRole::MMA)
    {
      pipeline_mma_s0_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Producer;
    }
    if (role == KernelSchedule::WarpRole::Softmax0)
    {
      pipeline_mma_s0_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Consumer;
    }
    pipeline_mma_s0_params.consumer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineS pipeline_mma_s0(
    shared_storage.pipelines.mma_s0,
    pipeline_mma_s0_params,
    ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{}
    );

    typename CollectiveMainloop::PipelineS::Params pipeline_mma_s1_params;
    if (role == KernelSchedule::WarpRole::MMA)
    {
      pipeline_mma_s1_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Producer;
    }
    if (role == KernelSchedule::WarpRole::Softmax1)
    {
      pipeline_mma_s1_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Consumer;
    }
    pipeline_mma_s1_params.consumer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineS pipeline_mma_s1(
    shared_storage.pipelines.mma_s1,
    pipeline_mma_s1_params,
    ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{}
    );

    typename CollectiveMainloop::PipelineC::Params pipeline_s0_corr_params;
    if (role == KernelSchedule::WarpRole::Softmax0)
    {
      pipeline_s0_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Producer;
    }
    if (role == KernelSchedule::WarpRole::Correction)
    {
      pipeline_s0_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Consumer;
    }
    pipeline_s0_corr_params.producer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    pipeline_s0_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineC pipeline_s0_corr(
      shared_storage.pipelines.s0_corr,
      pipeline_s0_corr_params,
      /*barrier init*/ cute::true_type{}
    );

    typename CollectiveMainloop::PipelineC::Params pipeline_s1_corr_params;
    if (role == KernelSchedule::WarpRole::Softmax1)
    {
      pipeline_s1_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Producer;
    }
    if (role == KernelSchedule::WarpRole::Correction)
    {
      pipeline_s1_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Consumer;
    }
    pipeline_s1_corr_params.producer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    pipeline_s1_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineC pipeline_s1_corr(
      shared_storage.pipelines.s1_corr,
      pipeline_s1_corr_params,
      /*barrier init*/ cute::true_type{}
    );

    typename CollectiveMainloop::PipelineO::Params pipeline_mma_corr_params;
    if (role == KernelSchedule::WarpRole::MMA)
    {
      pipeline_mma_corr_params.role = CollectiveMainloop::PipelineO::ThreadCategory::Producer;
    }
    if (role == KernelSchedule::WarpRole::Correction)
    {
      pipeline_mma_corr_params.role = CollectiveMainloop::PipelineO::ThreadCategory::Consumer;
    }
    pipeline_mma_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineO pipeline_mma_corr(
      shared_storage.pipelines.mma_corr,
      pipeline_mma_corr_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{}
    );

    typename CollectiveMainloop::PipelineE::Params pipeline_corr_epi_params;
    if (role == KernelSchedule::WarpRole::Correction)
    {
      pipeline_corr_epi_params.role = CollectiveMainloop::PipelineE::ThreadCategory::Producer;
    }
    if (role == KernelSchedule::WarpRole::Epilogue)
    {
      pipeline_corr_epi_params.role = CollectiveMainloop::PipelineE::ThreadCategory::Consumer;
    }
    pipeline_corr_epi_params.producer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    pipeline_corr_epi_params.consumer_arv_count = NumWarpsEpilogue * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineE pipeline_corr_epi(
      shared_storage.pipelines.corr_epi,
      pipeline_corr_epi_params,
      /*barrier init*/ cute::true_type{}
    );

    typename CollectiveMainloop::OrderBarrierSoftmax::Params params_order_s01;
    params_order_s01.group_id = role == KernelSchedule::WarpRole::Softmax1 ? 1 : 0;
    params_order_s01.group_size = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::OrderBarrierSoftmax order_s01(
      shared_storage.pipelines.order_s01, params_order_s01
    );

    TmemAllocator tmem_allocator;

    __syncthreads();

    pipeline_load_q.init_masks(ClusterShape{});
    pipeline_load_kv.init_masks(ClusterShape{});
    pipeline_mma_s0.init_masks(ClusterShape{});
    pipeline_mma_s1.init_masks(ClusterShape{});
    pipeline_mma_corr.init_masks(ClusterShape{});

    typename CollectiveMainloop::PipelineQ::PipelineState pipeline_load_q_consumer_state;
    typename CollectiveMainloop::PipelineQ::PipelineState pipeline_load_q_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineQ>();

    typename CollectiveMainloop::PipelineKV::PipelineState pipeline_load_kv_consumer_state;
    typename CollectiveMainloop::PipelineKV::PipelineState pipeline_load_kv_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineKV>();

    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s0_consumer_state;
    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s0_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineS>();

    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s1_consumer_state;
    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s1_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineS>();

    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s0_corr_consumer_state;
    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s0_corr_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineC>();

    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s1_corr_consumer_state;
    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s1_corr_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineC>();

    typename CollectiveMainloop::PipelineE::PipelineState pipeline_corr_epi_consumer_state;
    typename CollectiveMainloop::PipelineE::PipelineState pipeline_corr_epi_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineE>();

    typename CollectiveMainloop::PipelineO::PipelineState pipeline_mma_corr_consumer_state;
    typename CollectiveMainloop::PipelineO::PipelineState pipeline_mma_corr_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineO>();

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue{params.epilogue};

    if (role == KernelSchedule::WarpRole::Softmax0 || role == KernelSchedule::WarpRole::Softmax1)
    {
      warpgroup_reg_set<NumRegsSoftmax>();

      CUTLASS_PRAGMA_NO_UNROLL
      while (tile_scheduler.is_valid())
      {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
          params.problem_shape, get<2,1>(blk_coord)
        );

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape))
        {
          continue;
        }

        bool is_softmax_0 = role == KernelSchedule::WarpRole::Softmax0;

        mainloop.softmax(
          is_softmax_0 ? 0 : 1, blk_coord,
          params.mainloop, logical_problem_shape,
          is_softmax_0 ? pipeline_mma_s0 : pipeline_mma_s1,
          is_softmax_0 ? pipeline_mma_s0_consumer_state : pipeline_mma_s1_consumer_state,
          is_softmax_0 ? pipeline_s0_corr : pipeline_s1_corr,
          is_softmax_0 ? pipeline_s0_corr_producer_state : pipeline_s1_corr_producer_state,
          order_s01
        );
        
        ++tile_scheduler;
      }
    }
    else if (role == KernelSchedule::WarpRole::Correction)
    {
      warpgroup_reg_set<NumRegsCorrection>();

      CUTLASS_PRAGMA_NO_UNROLL
      while (tile_scheduler.is_valid())
      {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
          params.problem_shape, get<2,1>(blk_coord)
        );

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape))
        {
          continue;
        }

        if (get<1>(logical_problem_shape) == 0)
        {
          mainloop.correction_empty(
            blk_coord,
            params.mainloop, logical_problem_shape,
            params.problem_shape,
            shared_storage.epilogue,
            pipeline_corr_epi, pipeline_corr_epi_producer_state,
            epilogue
          );

          continue;
        }

        mainloop.correction(
          blk_coord,
          params.mainloop, logical_problem_shape,
          params.problem_shape,
          shared_storage.epilogue,
          pipeline_s0_corr, pipeline_s0_corr_consumer_state,
          pipeline_s1_corr, pipeline_s1_corr_consumer_state,
          pipeline_mma_corr, pipeline_mma_corr_consumer_state,
          pipeline_corr_epi, pipeline_corr_epi_producer_state,
          epilogue
        );

        if constexpr (NumWarpsEpilogue == 0)
        {
          static_assert(NumWarpsCorrection == 1);

          uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
          tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
        }

        ++tile_scheduler;
      }
    }
    else if (role == KernelSchedule::WarpRole::MMA)
    {
      warpgroup_reg_set<NumRegsOther>();

      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();

      CUTLASS_PRAGMA_NO_UNROLL
      while (tile_scheduler.is_valid())
      {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params, 
          params.problem_shape, get<2,1>(blk_coord)
        );

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)
          || 0 == get<1>(logical_problem_shape))
        {
          continue;
        }

        mainloop.mma(
          blk_coord,
          params.mainloop, logical_problem_shape,
          shared_storage.mainloop,
          pipeline_load_q, pipeline_load_q_consumer_state,
          pipeline_load_kv, pipeline_load_kv_consumer_state,
          pipeline_mma_s0, pipeline_mma_s0_producer_state,
          pipeline_mma_s1, pipeline_mma_s1_producer_state,
          pipeline_mma_corr, pipeline_mma_corr_producer_state
        );

        ++tile_scheduler;
      }
    }
    else if (role == KernelSchedule::WarpRole::Load)
    {
      warpgroup_reg_set<NumRegsOther>();

      CUTLASS_PRAGMA_NO_UNROLL
      while (tile_scheduler.is_valid())
      {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params, 
          params.problem_shape, get<2,1>(blk_coord)
        );

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)
          || 0 == get<1>(logical_problem_shape))
        {
          continue;
        }

        mainloop.load(
          blk_coord, logical_problem_shape,
          params.mainloop, params.problem_shape,
          shared_storage.mainloop,
          pipeline_load_q, pipeline_load_q_producer_state,
          pipeline_load_kv, pipeline_load_kv_producer_state
        );

        ++tile_scheduler;
      }
    }
    else if (role == KernelSchedule::WarpRole::Epilogue)
    {
      warpgroup_reg_set<NumRegsOther>();

      CUTLASS_PRAGMA_NO_UNROLL
      while (tile_scheduler.is_valid())
      {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, get<2,1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        epilogue.store(
          blk_coord, logical_problem_shape,
          params.epilogue, params.problem_shape,
          shared_storage.epilogue,
          pipeline_corr_epi, pipeline_corr_epi_consumer_state
        );

        ++tile_scheduler;
      }

      static_assert(NumWarpsEpilogue <= 1);
      if constexpr (NumWarpsEpilogue == 1)
      {
        uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
        tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
      }

    }
    else if (role == KernelSchedule::WarpRole::Empty)
    {
      warpgroup_reg_set<NumRegsEmpty>();

      /* no-op, donate regs and exit */
    }
  }

};

///////////////////////////////////////////////////////////////////////

} // namespace cutlass::fmha::kernel