//===------- WarpAllocator.cpp - Warp memory allocator ------- C++ -*-========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#pragma omp begin declare target device_type(nohost)

#include "Debug.h"
#include "Mapping.h"
#include "Memory.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"
#include "Interface.h"

using namespace ompx;

[[gnu::used, gnu::retain, gnu::weak,
  gnu::visibility(
      "protected")]] DeviceMemoryPoolTy __omp_rtl_device_memory_pool;
[[gnu::used, gnu::retain, gnu::weak,
  gnu::visibility("protected")]] DeviceMemoryPoolTrackingTy
    __omp_rtl_device_memory_pool_tracker;
// TODO: implement Device Debug Allocation Tracker

namespace {
constexpr const size_t Alignment = 16;
constexpr const size_t FirstThreadTeamRatio = 40;
constexpr const size_t FirstThreadWarpRatio = 40;
constexpr const size_t SplitThreadhold = Alignment * 4;

template <typename T> T abs(T V) { return V > 0 ? V : -V; }

//template <uint32_t WARP_SIZE, uint32_t TEAM_SIZE> struct WarpAllocator;
template <uint32_t MAX_TEAM_SIZE> struct WarpAllocator;

class WarpAllocatorEntry {
  template <uint32_t MAX_TEAM_SIZE> friend struct WarpAllocator;

  /// If Size is less than 0, the entry is allocated (in use).
  int64_t Size = 0;
  /// PrevSize is also supposed to be greater than or equal to 0. When it is 0,
  /// it is the first entry of the buffer.
  int64_t PrevSize = 0;

public:
  bool isFirst() const { return !PrevSize; }

  size_t getSize() const { return abs(Size); }
  void setSize(size_t V) { Size = V; }

  void setPrevSize(WarpAllocatorEntry *Prev) {
    PrevSize = Prev ? Prev->getSize() : 0;
  }

  size_t getUserSize() const { return getSize() - sizeof(WarpAllocatorEntry); }

  // Note: isUsed can not be !isUnused or other way around because when Size is
  // 0, it is uninitialized.
  bool isUsed() const { return Size < 0; }
  bool isUnused() const { return Size > 0; }

  void setUsed() {
    assert(isUnused() && "the entry is in use");
    Size *= -1;
  }
  void setUnused() {
    assert(isUsed() && "the entry is not in use");
    Size *= -1;
  }

  char *getUserPtr() { return reinterpret_cast<char *>(this + 1); }
  char *getEndPtr() { return reinterpret_cast<char *>(getNext()); }

  WarpAllocatorEntry *getPrev() { return utils::advance(this, -PrevSize); }
  WarpAllocatorEntry *getNext() { return utils::advance(this, getSize()); }

  static WarpAllocatorEntry *fromUserPtr(void *Ptr) { return fromPtr(Ptr) - 1; }

  static WarpAllocatorEntry *fromPtr(void *Ptr) {
    return reinterpret_cast<WarpAllocatorEntry *>(Ptr);
  }
};

static_assert(sizeof(WarpAllocatorEntry) == 16, "entry size mismatch");

//template <uint32_t WARP_SIZE, uint32_t TEAM_SIZE> struct WarpAllocator {
template <uint32_t MAX_TEAM_SIZE> struct WarpAllocator {
  void init() {
    if (mapping::isSPMDMode() &&
        (mapping::getThreadIdInBlock() || mapping::getBlockIdInKernel()))
      return;

    size_t HeapSize = __omp_rtl_device_memory_pool.Size;

    FirstThreadHeapSize = HeapSize * FirstThreadWarpRatio / 100;
    FirstThreadHeapSize = utils::align_down(FirstThreadHeapSize, Alignment);
    size_t OtherThreadHeapSize =
        (HeapSize - FirstThreadHeapSize) / (mapping::getWarpSize() - 1);
    OtherThreadHeapSize = utils::align_down(OtherThreadHeapSize, Alignment);

    size_t TeamHeapSize = FirstThreadHeapSize / mapping::getMaxTeamWarps();
    TeamHeapSize = utils::align_down(TeamHeapSize, Alignment);
    FirstTeamSize = TeamHeapSize;

    printf("Team Size: %d, WarpSize: %d, ThreadinBlock: %d\n", mapping::getMaxTeamWarps(), mapping::getWarpSize(), mapping::getMaxTeamWarps() * mapping::getWarpSize());
    printf("TeamAllocator Init: Total Team Memory Size (%ldMB), 1st Thread in Warp (%ldMB), Any thread in warp(%ldMB)\n",
        HeapSize / (1024 * 1024), TeamHeapSize / (1024 * 1024), OtherThreadHeapSize / (1024 * 1024));

    char *LastLimit = reinterpret_cast<char *>(__omp_rtl_device_memory_pool.Ptr);
    for (int I = 0; I < mapping::getWarpSize(); ++I) {
      for (int J = 0; J < mapping::getMaxTeamWarps(); ++J) {
        Entries[I * mapping::getMaxTeamWarps() + J] = nullptr;
        Limits[I * mapping::getMaxTeamWarps() + J] = LastLimit + TeamHeapSize * (J + 1);
      }
      LastLimit += I ? OtherThreadHeapSize : FirstThreadHeapSize;
      Limits[I * mapping::getMaxTeamWarps() + mapping::getMaxTeamWarps() - 1] =
          LastLimit;
      TeamHeapSize = OtherThreadHeapSize / mapping::getMaxTeamWarps();
      TeamHeapSize = utils::align_down(TeamHeapSize, Alignment);
    }
  }

  void *allocate(size_t Size) {
    int32_t TeamSlot = getTeamSlot();
    int32_t TIdInWarp = mapping::getThreadIdInWarp();

    Size = utils::align_up(Size + sizeof(WarpAllocatorEntry), Alignment);

    // Error our early if the requested size is larger than the entire block.
    if (Size > getBlockSize(TIdInWarp, TeamSlot))
      return nullptr;

    WarpAllocatorEntry *E = nullptr;
    {
      mutex::LockGuard LG(Locks[TIdInWarp * mapping::getMaxTeamWarps() + TeamSlot]);

      auto *LastEntry = Entries[TIdInWarp * mapping::getMaxTeamWarps() + TeamSlot];
      auto *NewWatermark = (LastEntry ? LastEntry->getEndPtr()
                                      : getBlockBegin(TIdInWarp, TeamSlot)) +
                           Size;
      if (NewWatermark >= Limits[TIdInWarp * mapping::getMaxTeamWarps() + TeamSlot]) {
        E = findMemorySlow(Size, TIdInWarp, TeamSlot);
      } else {
        E = LastEntry ? LastEntry->getNext()
                      : WarpAllocatorEntry::fromPtr(
                            getBlockBegin(TIdInWarp, TeamSlot));
        E->setSize(Size);
        E->setPrevSize(LastEntry);
        Entries[TIdInWarp * mapping::getMaxTeamWarps() + TeamSlot] = E;
      }

      if (!E)
        return nullptr;

      E->setUsed();
    }

    return E->getUserPtr();
  }

  void deallocate(void *Ptr) {
    WarpAllocatorEntry *E = WarpAllocatorEntry::fromUserPtr(Ptr);

    auto TeamSlot = getTeamSlot();
    auto TIdInWarp = mapping::getThreadIdInWarp();

    mutex::LockGuard LG(Locks[TIdInWarp * mapping::getMaxTeamWarps() + TeamSlot]);
    if (E->isUnused())
      return;
    E->setUnused();
    // Is last entry?
    if (E == Entries[TIdInWarp * mapping::getMaxTeamWarps() + TeamSlot]) {
      do {
        E = E->getPrev();
      } while (!E->isFirst() && !E->isUsed());
      Entries[TIdInWarp * mapping::getMaxTeamWarps() + TeamSlot] = E;
    }
  }

  memory::MemoryAllocationInfo getMemoryAllocationInfo(void *P) {
    if (!utils::isInRange(P, reinterpret_cast<char *>(__omp_rtl_device_memory_pool.Ptr),
                          __omp_rtl_device_memory_pool.Size))
      return {};

    auto TeamSlot = getTeamSlot();
    auto TIdInWarp = mapping::getThreadIdInWarp();
    for (int I = TIdInWarp; I < TIdInWarp + mapping::getWarpSize(); ++I) {
      int TId = I % mapping::getWarpSize();
      for (int J = TeamSlot; J < TeamSlot + mapping::getMaxTeamWarps(); ++J) {
        int SId = J % mapping::getMaxTeamWarps();
        if (P < getBlockBegin(TId, SId) || P >= getBlockEnd(TId, SId))
          continue;

        mutex::LockGuard LG(Locks[I * mapping::getMaxTeamWarps() + SId]);
        WarpAllocatorEntry *E = Entries[I * mapping::getMaxTeamWarps() + SId];
        if (!E)
          return {};
        if (E->getEndPtr() <= P)
          return {};
        bool isFirst = false;
        while (!isFirst) {
          if (E->getUserPtr() <= P && P < E->getEndPtr()) {
            if (!E->isUsed())
              return {};
            return {E->getUserPtr(), E->getUserSize()};
          }
          isFirst = E->isFirst();
          E = E->getPrev();
        }
      }
    }
    return {};
  }

private:

  char *getBlockBegin(int32_t TIdInWarp, int32_t TeamSlot) const {
    if (TeamSlot)
      return Limits[TIdInWarp * mapping::getMaxTeamWarps() + TeamSlot - 1];
    if (TIdInWarp)
      return Limits[(TIdInWarp - 1) * mapping::getMaxTeamWarps() + mapping::getMaxTeamWarps() - 1];
    return reinterpret_cast<char *>(__omp_rtl_device_memory_pool.Ptr);
  }
  char *getBlockEnd(int32_t TIdInWarp, int32_t TeamSlot) const {
    return Limits[TIdInWarp * mapping::getMaxTeamWarps() + TeamSlot];
  }

  size_t getBlockSize(int32_t TIdInWarp, int32_t TeamSlot) const {
    return getBlockEnd(TIdInWarp, TeamSlot) -
           getBlockBegin(TIdInWarp, TeamSlot);
  }

  int32_t getTeamSlot() { return mapping::getBlockIdInKernel() % 
    mapping::getMaxTeamWarps(); }

  WarpAllocatorEntry *findMemorySlow(size_t Size, int32_t TIdInWarp,
                                     int32_t TeamSlot) {
    char *Ptr = getBlockBegin(TIdInWarp, TeamSlot);
    char *Limit = getBlockEnd(TIdInWarp, TeamSlot);

    WarpAllocatorEntry *E = WarpAllocatorEntry::fromPtr(Ptr);
    do {
      if (!E->isUsed() && E->getSize() >= Size)
        break;
      E = E->getNext();
      if (reinterpret_cast<char *>(E) + Size > Limit)
        return nullptr;
    } while (1);

    size_t OldSize = E->getSize();
    if (OldSize - Size >= SplitThreadhold) {
      auto *OldNext = E->getNext();
      E->setSize(Size);
      auto *LeftOverE = E->getNext();
      LeftOverE->setPrevSize(E);
      LeftOverE->setSize(OldSize - Size);
      OldNext->setPrevSize(LeftOverE);
    }

    return E;
  }

  WarpAllocatorEntry *Entries[MAX_TEAM_SIZE]; //[WARP_SIZE][TEAM_SIZE];
  char *Limits[MAX_TEAM_SIZE]; //[WARP_SIZE][TEAM_SIZE];
  mutex::TicketLock Locks[MAX_TEAM_SIZE]; //[WARP_SIZE][TEAM_SIZE];
  size_t FirstThreadHeapSize;
  size_t FirstTeamSize;
};

// Max team size (hread blocl size)
WarpAllocator<1024> Allocator;

} // namespace

namespace ompx {
namespace memory {
MemoryAllocationInfo getMemoryAllocationInfo(void *P) {
  return Allocator.getMemoryAllocationInfo(P);
}
} // namespace memory
} // namespace ompx

extern "C" {

void *malloc(size_t Size) {
  if (!Size)
    return nullptr;
  void *P = Allocator.allocate(Size);
  assert(P && "allocator out of memory");
  assert(reinterpret_cast<intptr_t>(P) % Alignment == 0 &&
         "misaligned address");
  return P;
}

void free(void *P) {
  if (!P)
    return;
  Allocator.deallocate(P);
}

void *realloc(void *ptr, size_t new_size) {
  void *NewPtr = malloc(new_size);
  if (!NewPtr)
    return nullptr;
  WarpAllocatorEntry *E = WarpAllocatorEntry::fromUserPtr(ptr);
  __builtin_memcpy(NewPtr, ptr, utils::min(E->getUserSize(), new_size));
  free(ptr);
  return NewPtr;
}

void __kmpc_target_init_allocator() { Allocator.init(); }
}

#pragma omp end declare target
