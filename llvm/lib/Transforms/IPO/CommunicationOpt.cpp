//===-- CommunicationOpt.cpp - Collection of communication optimizations --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Communication optimization.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/CommunicationOpt.h"

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"

#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "communication-opt"

#if !defined(NDEBUG)
static constexpr auto TAG = "[" DEBUG_TYPE "]";
#endif

namespace {
class CommunicationOpt {

};
}

namespace {

struct CommunicationOptCGSCCLegacyPass : public CallGraphSCCPass {
  CallGraphUpdater CGUpdater;
  static char ID;

  CommunicationOptCGSCCLegacyPass() : CallGraphSCCPass(ID) {
    initializeCommunicationOptCGSCCLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    CallGraphSCCPass::getAnalysisUsage(AU);
  }

  bool runOnSCC(CallGraphSCC &CGSCC) override {
    SmallVector<Function *, 16> SCC;
    for (CallGraphNode *CGN : CGSCC) {
      Function *Fn = CGN->getFunction();
      if (!Fn || Fn->isDeclaration())
        continue;
      SCC.push_back(Fn);
    }

    if (SCC.empty())
      return false;

    Module &M = CGSCC.getCallGraph().getModule();
    CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
    CGUpdater.initialize(CG, CGSCC);

    return false;
  }

  bool doFinalization(CallGraph &CG) override { return CGUpdater.finalize(); }
};

} // end anonymous namespace

char CommunicationOptCGSCCLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(CommunicationOptCGSCCLegacyPass,
                      "communication-opt-cgscc",
                      "Communication code motion optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_END(CommunicationOptCGSCCLegacyPass, "communication-opt-cgscc",
                    "Communication code motion optimizations", false, false)

Pass *llvm::createCommunicationOptCGSCCLegacyPass() {
  return new CommunicationOptCGSCCLegacyPass();
}
