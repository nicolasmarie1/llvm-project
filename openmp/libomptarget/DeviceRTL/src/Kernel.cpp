//===--- Kernel.cpp - OpenMP device kernel interface -------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the kernel entry points for the device.
//
//===----------------------------------------------------------------------===//

#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"

using namespace _OMP;

#pragma omp declare target

static void inititializeRuntime(int Mode) {
  // Order is important here.
  synchronize::init(Mode);
  mapping::init(Mode);
  state::init(Mode);
}

/// Simple generic state machine for worker threads.
static void genericStateMachine(IdentTy *Ident) {

  uint32_t TId = mapping::getLogicThreadId();

  do {
    ParallelRegionFnTy WorkFn = 0;

    // Wait for the signal that we have a new work function.
    synchronize::threads();

    // Retrieve the work function from the runtime.
    bool IsActive = __kmpc_kernel_parallel(&WorkFn);

    // If there is nothing more to do, break out of the state machine by
    // returning to the caller.
    if (!WorkFn)
      return;

    if (IsActive) {
      ASSERT(!mapping::isSPMDMode());
      ((void (*)(uint32_t, uint32_t))WorkFn)(0, TId);
      __kmpc_kernel_end_parallel();
    }

    synchronize::threads();

  } while (true);
}

namespace {
void runSIMDStateMachine(IdentTy *Ident) {
  uint32_t LaneId = mapping::getThreadIdInWarp();
  do {
    SIMDRegionFnTy WorkFn = nullptr;

    // Wait for the signal that we have a new work function.
    synchronize::warp(mapping::activemask());

    // Retrieve the work function from the runtime.
    bool IsActive = __kmpc_kernel_simd(&WorkFn);

    if (!WorkFn)
      return;

    if (IsActive) {
      ((void (*)(uint32_t, uint32_t))WorkFn)(0, LaneId);
      __kmpc_kernel_end_simd();
    }

    synchronize::warp(mapping::activemask());
  } while (true);
}
} // namespace

extern "C" {

/// Initialization
///
/// \param Ident               Source location identification, can be NULL.
///
int32_t __kmpc_target_init(IdentTy *Ident, int Mode,
                           bool UseGenericStateMachine, bool) {
  Mode = Mode | 0x2;

  inititializeRuntime(Mode);

  // For all SIMD workers, start the simd state machine.
  if (mapping::utils::isSIMDMode(Mode)) {
    uint32_t LaneId = mapping::getThreadIdInWarp();
    if (LaneId) {
      runSIMDStateMachine(Ident);
      return LaneId;
    }
  }

  const bool IsSPMD = mapping::utils::isSPMDMode(Mode);
  if (IsSPMD)
    synchronize::threads();

  if (IsSPMD) {
    state::assumeInitialState(IsSPMD);
    return -1;
  }

  if (mapping::isMainThreadInGenericMode())
    return -1;

  if (UseGenericStateMachine)
    genericStateMachine(Ident);

  return mapping::getThreadIdInBlock();
}

/// De-Initialization
///
/// In non-SPMD, this function releases the workers trapped in a state machine
/// and also any memory dynamically allocated by the runtime.
///
/// \param Ident Source location identification, can be NULL.
///
void __kmpc_target_deinit(IdentTy *Ident, int Mode, bool) {
  const bool IsSPMD = mapping::utils::isSPMDMode(Mode);

  state::assumeInitialState(IsSPMD);
  if (IsSPMD)
    return;

  // Signal the workers to exit the state machine and exit the kernel.
  state::ParallelRegionFn = nullptr;
}

int8_t __kmpc_is_spmd_exec_mode() { return mapping::isSPMDMode(); }
}

#pragma omp end declare target
