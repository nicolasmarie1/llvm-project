//===--------- Mapping.h - OpenMP device runtime mapping helpers -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_MAPPING_H
#define OMPTARGET_MAPPING_H

#include "Types.h"

namespace _OMP {

namespace mapping {

#pragma omp declare target

inline constexpr uint32_t MaxThreadsPerTeam = 1024;

#pragma omp end declare target

/// Initialize the mapping machinery.
void init(int Mode);

/// Return true if the kernel is executed in SPMD mode.
bool isSPMDMode();

/// Return true if the kernel is executed in generic mode.
bool isGenericMode();

/// Return true if the kernel is executed in SIMD mode.
bool isSIMDMode();

/// Return true if the executing thread is the main thread in generic mode.
bool isMainThreadInGenericMode();

/// Return true if the executing thread has the lowest Id of the active threads
/// in the warp.
bool isLeaderInWarp();

/// Return a mask describing all active threads in the warp.
LaneMaskTy activemask();

/// Return a mask describing all threads with a smaller Id in the warp.
LaneMaskTy lanemaskLT();

/// Return a mask describing all threads with a larget Id in the warp.
LaneMaskTy lanemaskGT();

/// Return the thread Id in the warp, in [0, getWarpSize()).
uint32_t getThreadIdInWarp();

/// Return the thread Id in the block, in [0, getBlockSize()).
uint32_t getThreadIdInBlock();

/// Return the logic thread Id, which depends on how we map an OpenMP thread to
/// the target device. In non-SIMD mode, we map an OpenMP thread to a device
/// thread. In SIMD mode, we map an OpenMP thread to a warp, and each thread in
/// the warp is a SIMD lane.
uint32_t getLogicThreadId();

/// Return the warp id in the block.
uint32_t getWarpId();

/// Return the warp size, thus number of threads in the warp.
uint32_t getWarpSize();

/// Return the number of warps in the block.
uint32_t getNumberOfWarpsInBlock();

/// Return the block Id in the kernel, in [0, getKernelSize()).
uint32_t getBlockId();

/// Return the block size, thus number of threads in the block.
uint32_t getBlockSize();

/// Return the number of blocks in the kernel.
uint32_t getNumberOfBlocks();

/// Return the kernel size, thus number of threads in the kernel.
uint32_t getKernelSize();

/// Return the number of processing elements on the device.
uint32_t getNumberOfProcessorElements();

namespace utils {
/// Return true if \p Mode indicates SPMD mode.
inline bool isSPMDMode(int Mode) { return Mode & 0x1; }

/// Return true if \p Mode indicates generic mode.
inline bool isGenericMode(int Mode) { return !isSPMDMode(Mode); }

/// Return true if \p Mode indicates SIMD mode.
inline bool isSIMDMode(int Mode) { return Mode & 0x2; }
} // namespace utils

} // namespace mapping

} // namespace _OMP

#endif
