//===------- HostRPC.cpp - Implementation of host RPC ------------- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma omp begin declare target device_type(nohost)

#include "HostRPC.h"

#include "Debug.h"
#include "LibC.h"
#include "Memory.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

#include "llvm/Frontend/OpenMP/OMPDeviceConstants.h"
#include "llvm-libc-types/rpc_opcodes_t.h"
#include "llvm-libc-types/rpc_port_t.h"

#ifdef HOSTRPC_DEBUG
#define DEBUG_PREFIX "host-rpc-device"
#define DP(FMT, ...)                                                           \
  { printf("%s --> " FMT, DEBUG_PREFIX, __VA_ARGS__); }
#else
#define DP(FMT, ...)
#endif

using namespace ompx;
using namespace hostrpc;

using ArgType = llvm::omp::OMPTgtHostRPCArgType;

#ifdef HOSTRPC_PROFILING
int32_t HostRPCId;
double GetDescStart;
double GetDescEnd;
double AddArgStart;
double AddArgEnd;
double IssueAndWaitStart;
double IssueAndWaitEnd;
double CopyBackStart;
double CopyBackEnd;
#endif


// libc rpc functions forward declare:
// TODO: replace when a proper header exposing device functions is created
extern "C" {
  rpc_port_t rpc_open_port(rpc_opcode_t);
  void rpc_send_n(rpc_port_t *handle, const void *src, size_t size);
  void rpc_recv_n(rpc_port_t *handle, void *dst, size_t *size);
  void rpc_close_port(rpc_port_t *handle);
}


namespace {
size_t HostRPCMemoryBufferCurrentPosition = 0;
constexpr const size_t Alignment = 16;

static_assert(sizeof(intptr_t) == sizeof(int64_t), "pointer size not match");

struct HostRPCArgInfo {
  void *BasePtr;
  int64_t Type;
  int64_t Size;
  HostRPCArgInfo *Next;
};

struct HostRPCPointerMapEntry {
  void *BasePtr;
  void *MappedBasePtr;
  int64_t Size;
  int64_t Kind;
};

void *getMappedPointer(Descriptor *D, void *BasePtr, int64_t Size,
                       int64_t Offset, int64_t Kind) {
  assert(D->ArgMap && "ArgMap should not be nullptr");

  HostRPCPointerMapEntry *MapTable =
      reinterpret_cast<HostRPCPointerMapEntry *>(D->ArgMap);
  int I = 0;
  for (; I < D->NumArgs && MapTable[I].BasePtr; ++I)
    if (MapTable[I].BasePtr == BasePtr)
      return utils::advance(MapTable[I].MappedBasePtr, Offset);

  MapTable[I].BasePtr = BasePtr;
  MapTable[I].MappedBasePtr = malloc(Size);
  MapTable[I].Size = Size;
  MapTable[I].Kind = Kind;

  if (Kind & ArgType::OMP_HOST_RPC_ARG_COPY_TO) {
    __builtin_memcpy(MapTable[I].MappedBasePtr, BasePtr, Size);
    DP("getMappedPointer: copy %ld bytes memory from %p to %p.\n", Size,
       BasePtr, MapTable[I].MappedBasePtr);
  }

  return utils::advance(MapTable[I].MappedBasePtr, Offset);
}

void copybackIfNeeded(Descriptor *D) {
  if (!D->ArgMap)
    return;

  auto *MapTable = reinterpret_cast<HostRPCPointerMapEntry *>(D->ArgMap);
  for (int I = 0; I < D->NumArgs && MapTable[I].BasePtr; ++I)
    if (MapTable[I].Kind & ArgType::OMP_HOST_RPC_ARG_COPY_FROM) {
      __builtin_memcpy(MapTable[I].BasePtr, MapTable[I].MappedBasePtr,
                       MapTable[I].Size);
      DP("copybackIfNeeded: copy %ld bytes memory from %p to %p.\n",
         MapTable[I].Size, MapTable[I].MappedBasePtr, MapTable[I].BasePtr);
    }
}

} // namespace

extern "C" {
__attribute__((noinline, used)) void *
__kmpc_host_rpc_get_desc(int32_t CallId, int32_t NumArgs, void *ArgInfo) {

  DP("get desc for request (id=%d), NumArgs=%d, ArgInfo=%p.\n", CallId, NumArgs,
     ArgInfo);
#ifdef HOSTRPC_DEBUG
  {
    void **AIs = reinterpret_cast<void **>(ArgInfo);
    for (int I = 0; I < NumArgs; ++I)
      DP("ArgInfo[%d]=%p.\n", I, AIs[I]);
  }
#endif

#ifdef HOSTRPC_PROFILING
  HostRPCId = CallId;
  GetDescStart = omp_get_wtime();
#endif

  Descriptor *D = (Descriptor *) malloc(sizeof(Descriptor));

  D->Id = CallId;
  D->ArgInfo = reinterpret_cast<void **>(ArgInfo);
  D->NumArgs = NumArgs;
  D->Status = EXEC_STAT_CREATED;
  D->ReturnValue = 0;
  D->Args =
      reinterpret_cast<Argument *>(malloc(sizeof(Argument) * NumArgs));
  D->ArgMap = malloc(sizeof(HostRPCPointerMapEntry) * NumArgs);

  assert(!NumArgs || (D->Args && D->ArgMap) && "out of host rpc memory!");

  // Reset the map table.
  auto *ArgMap = reinterpret_cast<HostRPCPointerMapEntry *>(D->ArgMap);
  for (int I = 0; I < NumArgs; ++I)
    ArgMap[I].BasePtr = nullptr;

#ifdef HOSTRPC_PROFILING
  GetDescEnd = omp_get_wtime();
  AddArgStart = omp_get_wtime();
#endif

  return D;
}

__attribute__((noinline, used)) void
__kmpc_host_rpc_add_arg(void *Desc, int64_t ArgVal, int32_t ArgNum) {
  auto *D = reinterpret_cast<Descriptor *>(Desc);
  assert(ArgNum < D->NumArgs && "out-of-range arguments");
  Argument &ArgInDesc = D->Args[ArgNum];

  DP("add arg (no=%d), arg=%lx to request (id=%d).\n", ArgNum, ArgVal, D->Id);

  // This early branch can rule out nullptr and zero scalar value because it
  // doesn't matter whether it is a pointer or scalar value.
  if (ArgVal == 0) {
    ArgInDesc.Value = 0;
    ArgInDesc.ArgType = Type::ARG_LITERAL;

    DP("arg (no=%d) is null, done.\n", ArgNum);

    return;
  }

  void *ArgPtr = reinterpret_cast<void *>(ArgVal);

  const auto *AI = reinterpret_cast<HostRPCArgInfo *>(D->ArgInfo[ArgNum]);

  DP("try to find arg (no=%d) from args AI=%p\n", ArgNum, AI);

  if (AI) {
    // Let's first check if Arg is a scalar.
    if (AI->Type == ArgType::OMP_HOST_RPC_ARG_SCALAR) {
      assert(AI->BasePtr == ArgPtr && "invalid scalar argument info");
      assert(AI->Next == nullptr && "invalid scalar argument info");

      ArgInDesc.Value = ArgVal;
      ArgInDesc.ArgType = Type::ARG_LITERAL;

      DP("arg (no=%d) is scalar, done.\n", ArgNum);

      return;
    }

    // Then let's see if it is a literal pointer that we don't need copy.
    auto *P = AI;
    while (P) {
      if (P->Type == ArgType::OMP_HOST_RPC_ARG_PTR && P->BasePtr == ArgPtr) {
        ArgInDesc.Value = ArgVal;
        ArgInDesc.ArgType = Type::ARG_POINTER;

        DP("arg (no=%d) is literal pointer, done.\n", ArgNum);

        return;
      }
      P = P->Next;
    }

    // Next we check if it is within the range of any buffer described in
    // argument info.
    P = AI;
    while (P) {
      if ((P->Type & ArgType::OMP_HOST_RPC_ARG_PTR) && P->Size) {
        if (utils::isInRange(ArgPtr, P->BasePtr, P->Size)) {
          auto Size = P->Size;
          auto Offset = utils::getPtrDiff(ArgPtr, P->BasePtr);

          ArgInDesc.Value = utils::ptrtoint(
              getMappedPointer(D, P->BasePtr, Size, Offset, P->Type));
          ArgInDesc.ArgType = Type::ARG_POINTER;

          DP("found a match for arg (no=%d). done.\n", ArgNum);

          return;
        }
      }
      P = P->Next;
    }
  }

  // Now we can't find a match from argument info, then we assume it is from
  // dynamic allocation.
  memory::MemoryAllocationInfo MAI = memory::getMemoryAllocationInfo(ArgPtr);
  if (MAI.isValid()) {
    auto Size = MAI.Size;
    auto Offset = utils::getPtrDiff(ArgPtr, MAI.BasePtr);

    ArgInDesc.Value = utils::ptrtoint(
        getMappedPointer(D, MAI.BasePtr, Size, Offset,
                         /* Kind */ ArgType::OMP_HOST_RPC_ARG_COPY_TOFROM));
    ArgInDesc.ArgType = Type::ARG_POINTER;

    DP("arg (no=%d) is from malloc. done.\n", ArgNum);

    return;
  }

  printf("request (id=%d) arg (no=%d, val=%p) is unknown. send it to host "
         "directly.\n",
         D->Id, ArgNum, ArgPtr);

  ArgInDesc.Value = ArgVal;
  ArgInDesc.ArgType = Type::ARG_POINTER;
}

__attribute__((noinline, used)) int64_t
__kmpc_host_rpc_send_and_wait(void *Desc) {
  Descriptor *D = reinterpret_cast<Descriptor *>(Desc);
  int32_t Id = D->Id;

#ifdef HOSTRPC_PROFILING
  AddArgEnd = omp_get_wtime();
  IssueAndWaitStart = omp_get_wtime();
#endif



//  // WORKING back & forth of an uint64_t
//
//  printf("[HostRPC] [Device]: Start \n");
//
//  rpc_port_t port = rpc_open_port(RPC_GPUFIRST);
//
//  uint64_t size_send = sizeof(uint64_t);
//  void *buf_send = malloc(size_send);
//  *((uint64_t *) buf_send) = 123456789;
//
//  printf("[Hostrpc] [Device] [SEND]: %lu\n", *((uint64_t *) buf_send));
//  printf("[HostRPC] [Device] [SEND] Size: %lu\n", size_send);
//
//  rpc_send_n(&port, buf_send, size_send);
//
//
//  uint64_t size_recv = sizeof(uint64_t);
//  void *buf_recv = malloc(size_recv);
//
//  rpc_recv_n(&port, buf_recv, &size_recv);
//
//  printf("[HostRPC] [Device] [RECV]: %lu\n", *((uint64_t *) buf_recv));
//  printf("[HostRPC] [Device] [RECV] Size: %lu\n", size_recv);
//
//  rpc_close_port(&port);
//
//  assert(size_send == size_recv);
//
//  printf("[HostRPC] [Device]: End \n");
//
//  // END of working part


  rpc_port_t port = rpc_open_port(RPC_GPUFIRST);

  Argument *Args = D->Args;

  rpc_send_n(&port, D, sizeof(Descriptor));
  rpc_send_n(&port, Args, sizeof(Argument) * D->NumArgs);

  // CPU is calling the function here

  // unuse
  uint64_t size_recv = 0;

  rpc_recv_n(&port, D, &size_recv);
  rpc_recv_n(&port, Args, &size_recv);

  D->Args = Args;

  (void) size_recv;
  rpc_close_port(&port);


#ifdef HOSTRPC_PROFILING
  IssueAndWaitEnd = omp_get_wtime();
#endif

  DP("finish waiting for request (id=%d).\n", Id);

  int64_t Ret = D->ReturnValue;

  assert(!D->NumArgs || D->ArgMap && "arg map should not be nullptr");

#ifdef HOSTRPC_PROFILING
  CopyBackStart = omp_get_wtime();
#endif

  if (D->ArgMap) {
    DP("copy memory back for request (id=%d).\n", Id);
    copybackIfNeeded(D);
    DP("finish copy memory back for request (id=%d).\n", Id);
  }

#ifdef HOSTRPC_PROFILING
  CopyBackEnd = omp_get_wtime();
#endif

  // free memory allocated for the call
  HostRPCPointerMapEntry *MapTable = reinterpret_cast<HostRPCPointerMapEntry *>(D->ArgMap);
  for(int i = 0; i < D->NumArgs && MapTable[i].BasePtr; ++i){
    free(MapTable[i].MappedBasePtr);
  }
  free(D->Args);
  free(D->ArgMap);
  free(D);

  DP("request (id=%d) is done with return code=%lx.\n", Id, Ret);

#ifdef HOSTRPC_PROFILING
  printf("[host-rpc-profiling-device] id=%d, init=%lf, add_arg=%lf, wait=%lf, "
         "copy=%lf.\n",
         HostRPCId, GetDescEnd - GetDescStart, AddArgEnd - AddArgStart,
         IssueAndWaitEnd - IssueAndWaitStart, CopyBackEnd - CopyBackStart);
#endif

  return Ret;
}

__attribute__((noinline, used)) void
__kmpc_launch_parallel_51_kernel(const char *name, int32_t gtid,
                                 int32_t if_expr, int32_t num_threads,
                                 void **args, int64_t nargs) {
  constexpr const int64_t NumArgs = 6;
  HostRPCArgInfo ArgInfoArray[NumArgs];
  void *ArgInfo[NumArgs];
  for (unsigned I = 0; I < NumArgs; ++I) {
    ArgInfoArray[I].BasePtr = 0;
    ArgInfoArray[I].Type = ArgType::OMP_HOST_RPC_ARG_SCALAR;
    ArgInfoArray[I].Size = 0;
    ArgInfoArray[I].Next = nullptr;
    ArgInfo[I] = &ArgInfoArray[I];
  }
  auto *D = (Descriptor *)__kmpc_host_rpc_get_desc(0, NumArgs, (void *)ArgInfo);

  // Set up arg info struct.
  ArgInfoArray[0].BasePtr = const_cast<char *>(name);
  ArgInfoArray[0].Type = ArgType::OMP_HOST_RPC_ARG_COPY_TO;
  ArgInfoArray[0].Size = strlen(name) + 1;
  ArgInfoArray[1].BasePtr =
      reinterpret_cast<void *>(static_cast<int64_t>(gtid));
  ArgInfoArray[2].BasePtr =
      reinterpret_cast<void *>(static_cast<int64_t>(if_expr));
  ArgInfoArray[3].BasePtr =
      reinterpret_cast<void *>(static_cast<int64_t>(num_threads));
  ArgInfoArray[5].BasePtr = reinterpret_cast<void *>(nargs);

  // We need to treat args in a little bit different way because nargs might be
  // on the stack.
  ArgInfoArray[4].Size = sizeof(void *) * nargs;
  void *Args = nullptr;
  if (nargs) {
    Args = malloc(ArgInfoArray[4].Size);
    __builtin_memcpy(Args, args, ArgInfoArray[4].Size);
  }
  ArgInfoArray[4].BasePtr = Args;

  D->Id = CALLID___kmpc_launch_parallel_51_kernel;

  __kmpc_host_rpc_add_arg(D, reinterpret_cast<int64_t>(name), 0);
  __kmpc_host_rpc_add_arg(D, gtid, 1);
  __kmpc_host_rpc_add_arg(D, if_expr, 2);
  __kmpc_host_rpc_add_arg(D, num_threads, 3);
  __kmpc_host_rpc_add_arg(D, reinterpret_cast<int64_t>(Args), 4);
  __kmpc_host_rpc_add_arg(D, nargs, 5);

  (void)__kmpc_host_rpc_send_and_wait(D);
}
}

#pragma omp end declare target
