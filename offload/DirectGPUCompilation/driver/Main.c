//===------- Main.c - Direct compilation program start point ------ C -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string.h>
extern int __user_main(int, char *[]);
extern void __kmpc_target_init_allocator(void);

#define TEAM_LIMIT 20

#ifdef SINGLE_THREAD_EXECUTION
#define THREAD_LIMIT 1
#else
#define THREAD_LIMIT 1024
#endif

int main(int argc, char *argv[]) {
#pragma omp target enter data map(to: argv[:argc])

  for (int I = 0; I < argc; ++I) {
#pragma omp target enter data map(to: argv[I][:strlen(argv[I])])
  }

  int Ret = 0;
#pragma omp target enter data map(to: Ret)

#pragma omp target teams num_teams(1) thread_limit(1)
  { __kmpc_target_init_allocator(); }

#pragma omp target teams num_teams(TEAM_LIMIT) thread_limit(THREAD_LIMIT) // 214 max
  {
    Ret = __user_main(argc, argv);
  }

#pragma omp target exit data map(from: Ret)

  return Ret;
}
