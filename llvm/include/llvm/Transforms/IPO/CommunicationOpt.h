//===- CommunicationOpt.h - Communication optimizations ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_COMMUNICATION_H
#define LLVM_TRANSFORMS_IPO_COMMUNICATION_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// Communication optimizations pass.
class CommunicationOptPass : public PassInfoMixin<CommunicationOptPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class CommunicationOptCGSCCPass
    : public PassInfoMixin<CommunicationOptCGSCCPass> {
public:
  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_COMMUNICATION_H
