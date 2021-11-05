//===- hlstool.cpp - a CIRCT-based HLS compiler ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements 'hlstool', which composes together a variety of
// libraries in a way that is convenient to work with as a user.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/HandshakeToFIRRTL.h"
#include "circt/Conversion/Passes.h"
#include "circt/Conversion/StandardToHandshake.h"
#include "circt/Dialect/Calyx/CalyxDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true));

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false), cl::Hidden);

/// Process a single buffer of the input.
static LogicalResult processBuffer(MLIRContext &context, TimingScope &ts,
                                   llvm::SourceMgr &sourceMgr) {
  // Parse the input.
  OwningModuleRef module;
  auto parserTimer = ts.nest("MLIR Parser");
  module = parseSourceFile(sourceMgr, &context);

  if (!module)
    return failure();

  // Setup pass manager.
  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  applyPassManagerCLOptions(pm);

  // Calyx lowering

  // Handshake lowering
  pm.addPass(createHandshakeDataflowPass());
  pm.nest<handshake::FuncOp>().addPass(createCanonicalizerPass());
  pm.nest<handshake::FuncOp>().addPass(createHandshakeInsertBufferPass());
  pm.addPass(createHandshakeToFIRRTLPass());

  // FIRRTL lowering
  pm.addNestedPass<firrtl::CircuitOp>(
      firrtl::createLowerFIRRTLTypesPass(false));
  pm.addPass(createLowerFIRRTLToHWPass(false));

  // Hardware emission
  pm.nest<hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());
  pm.nest<hw::HWModuleOp>().addPass(sv::createPrettifyVerilogPass());
  pm.addPass(createExportVerilogPass(llvm::outs()));

  if (failed(pm.run(module.get())))
    return failure();

  // We intentionally "leak" the Module into the MLIRContext instead of
  // deallocating it.  There is no need to deallocate it right before process
  // exit.
  (void)module.release();
  return success();
}

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether the
/// user set the verifyDiagnostics option.
static LogicalResult
processInputSplit(MLIRContext &context, TimingScope &ts,
                  std::unique_ptr<llvm::MemoryBuffer> buffer) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return processBuffer(context, ts, sourceMgr);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);
  (void)processBuffer(context, ts, sourceMgr);
  return sourceMgrHandler.verify();
}

/// Process the entire input provided by the user, splitting it up if the
/// corresponding option was specified.
static LogicalResult processInput(MLIRContext &context, TimingScope &ts,
                                  std::unique_ptr<llvm::MemoryBuffer> input) {
  return splitAndProcessBuffer(
      std::move(input),
      [&](std::unique_ptr<MemoryBuffer> buffer, raw_ostream &) {
        return processInputSplit(context, ts, std::move(buffer));
      },
      llvm::outs());
}

static LogicalResult executeHlstool(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Register our dialects.
  context.loadDialect<handshake::HandshakeDialect, esi::ESIDialect,
                      calyx::CalyxDialect, arith::ArithmeticDialect,
                      StandardOpsDialect, memref::MemRefDialect>();

  // Process the input.
  if (failed(processInput(context, ts, std::move(input))))
    return failure();

  return success();
}

/// Main driver for hlstool command.  This sets up LLVM and MLIR, and parses
/// command line options before passing off to 'executeHlstool'.  This is set up
/// so we can `exit(0)` at the end of the program to avoid teardown of the
/// MLIRContext and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  registerLoweringCLOptions();

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR-based HLS compiler\n");

  MLIRContext context;

  // Do the guts of the hlstool process.
  auto result = executeHlstool(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
