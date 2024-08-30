//===------- Main.c - Direct compilation program start point ------ C -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

extern int __user_main(int, char *[]);
extern void __kmpc_target_init_allocator(void);

//#ifdef SINGLE_THREAD_EXECUTION
//#define THREAD_LIMIT 1
//#define TEAM_LIMIT 1
//#else
// 110 * 960 = 105 600 threads max for now
// could probably do a 110 * 1024 = 112640 if we fixe offload
//#define TEAM_LIMIT 110
//#define THREAD_LIMIT 961 // limit is not included, max is 960

// 220 * 512 = 112 640 threads max
//#define TEAM_LIMIT 221
//#define THREAD_LIMIT 513 // limit is not included, max is 512

// 440 * 256 = 112 640 threads max
//#define TEAM_LIMIT 440
//#define THREAD_LIMIT 257 // limit is not included, max is 256

// 880 * 128 = 112 640 threads max
//#define TEAM_LIMIT 880
//#define THREAD_LIMIT 129 // limit is not included, max is 128

// 1320 * 64 = 84 480 threads max
//#define TEAM_LIMIT 1320 // will not do more that 1320 thread block per gpu, i don't know why
//#define THREAD_LIMIT 65 // limit is not included, max is 65

// /!\ Do not do less that 64 thread per teams, that will put more that 1 warp
// per teams and the synchronisations functions are not build to handle that.

// Light for testing
//#define TEAM_LIMIT 880
//#define THREAD_LIMIT 65 // limit is not included, max is 64

//#define TEAM_LIMIT 1760
//#define TEAM_LIMIT 1320

//#define TEAM_LIMIT 660
//#define THREAD_LIMIT 128

//#endif


//#pragma omp begin declare target device_type(nohost)
//void scheduler_init(void);
//#pragma omp begin declare variant match(device = {arch(amdgcn)})
//void scheduler_init(void){
//  __builtin_amdgcn_ds_gws_init(TEAM_LIMIT - 1, 0);
//}
//#pragma omp end declare variant
//#pragma omp end declare target

#define DEBUG

#pragma omp begin declare target device_type(nohost)
extern int __omp_rtl_mpi_mode;
#pragma omp end declare target

int main(int argc, char *argv[]) {

  char* MPI_Ranks   = getenv("MPI_RANKS");
  char* MPI_Threads = getenv("MPI_THREADS");

  int nb_teams   = 8;
  int nb_threads = 64;

  if (MPI_Ranks != NULL)
    nb_teams = atoi(MPI_Ranks);

  if (MPI_Threads != NULL)
    nb_threads = atoi(MPI_Threads);

  int min_size = 64;

  // nb. threads should be a modulo Wave Front/Warp Size
  nb_threads -= nb_threads % min_size;

#ifdef DEBUG
  printf("Using %d MPI_Ranks and %d MPI_Threads\n", nb_teams, nb_threads);
#endif

#pragma omp target enter data map(to: argv[:argc])

  for (int I = 0; I < argc; ++I) {
#pragma omp target enter data map(to: argv[I][:strlen(argv[I]) + 1])
  }

  int Ret = 0;
#pragma omp target enter data map(to: Ret)

#pragma omp target teams num_teams(1) thread_limit(nb_threads)
  {
    __kmpc_target_init_allocator();
  }

#pragma omp target teams num_teams(nb_teams) thread_limit(nb_threads)
  {
#ifdef DEBUG
    if (omp_get_team_num() == 0) {
      printf("NB PROCS: %d / NB MAX THREADS BLOCK: %d\n", omp_get_num_procs(), omp_get_num_procs() / (1024 / 64));
      #pragma omp parallel
      #pragma omp single
      printf("OMP TEAMS: %d / OMP THREADS: %d\n", omp_get_num_teams(), omp_get_num_threads());
    }
#endif

    __omp_rtl_mpi_mode = 1; // tell omp to lie about hardware config
    Ret = __user_main(argc, argv);
  }

#pragma omp target exit data map(from: Ret)

  return Ret;
}
