//===------- Memory.cpp - OpenMP device runtime memory allocator -- C++ -*-===//
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

#include "Memory.h"

char *CONSTANT(omptarget_device_heap_buffer)
    __attribute__((used, retain, weak, visibility("protected")));

size_t CONSTANT(omptarget_device_heap_size)
    __attribute__((used, retain, weak, visibility("protected")));

__attribute__((used, retain, weak, visibility("protected")))
size_t omptarget_device_heap_cur_pos = 0;

extern "C" {

void *malloc(size_t Size) {
  constexpr const size_t Alignment = 16;
  Size = (Size + Alignment - 1) & ~(Alignment - 1);

  if (Size + omptarget_device_heap_cur_pos < omptarget_device_heap_size) {
    void *R = omptarget_device_heap_buffer + omptarget_device_heap_cur_pos;
    omptarget_device_heap_cur_pos += Size;
    return R;
  }

  return nullptr;
}

void free(void *) {}
}

#pragma omp end declare target
