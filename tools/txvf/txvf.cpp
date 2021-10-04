//===- txvf.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Executable MLIR thingy
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinDialect.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"

#include "BuiltinInterpreter.h"
#include "DialectSimulator.h"
#include "HandshakeInterpreter.h"
#include "MemrefInterpreter.h"
#include "StandardInterpreter.h"

using namespace llvm;
using namespace mlir;

#define DEBUG_TYPE "TXVF"

static cl::list<std::string>
    inputFiles("files", cl::OneOrMore, cl::value_desc("list"),
               cl::desc("<list of one or more simulation files>"),
               cl::CommaSeparated);

static cl::opt<std::string>
    toplevelFunction("top-level-function", cl::Required,
                     cl::desc("The top-level function to execute"));

static cl::opt<std::string>
    referenceModel("ref", cl::Optional,
                   cl::desc("The dialect namespace of the reference model"));

namespace circt {
namespace txvf {

/// Adds the set of modules contained within the input source files to the input
/// modules vector.
static LogicalResult getModules(SmallVectorImpl<OwningModuleRef> &modules,
                                MLIRContext *ctx) {
  if (inputFiles.size() == 0) {
    errs() << "Error: No input files specified\n";
    return failure();
  }

  for (auto fn : inputFiles) {
    auto file_or_err = MemoryBuffer::getFileOrSTDIN(fn.c_str());
    if (std::error_code error = file_or_err.getError()) {
      errs() << "Error: Could not open input file '" << fn
             << "': " << error.message() << "\n";
      return failure();
    }

    // Load the MLIR module.
    SourceMgr source_mgr;
    source_mgr.AddNewSourceBuffer(std::move(*file_or_err), SMLoc());
    modules.emplace_back(mlir::parseSourceFile(source_mgr, ctx));
    if (!modules.back()) {
      errs() << "Error: Found no modules in input file '" << fn << "'\n";
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "Loaded module from " << fn << "\n");
  }
  return success();
}

/// Locates the top-level function within the set of loaded modules.
static mlir::Operation *
getTopLevelOperation(SmallVectorImpl<OwningModuleRef> &modules) {
  Operation *toplevelOp = nullptr;
  for (auto &mod : modules) {
    mlir::Operation *mainOp = mod->lookupSymbol(toplevelFunction);
    if (mainOp) {
      if (toplevelOp) {
        errs() << "Error: Found multiple definitions of top-level symbol '"
               << toplevelFunction << "'\n";
        return nullptr;
      }
      toplevelOp = mainOp;
    }
  }
  return toplevelOp;
}

static void registerDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<handshake::HandshakeDialect>();
}

int txvfMain(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "CIRCT Transactional Cosimulation\n");
  mlir::DialectRegistry registry;
  registerDialects(registry);
  mlir::MLIRContext context(registry);

  /// Load modules.
  SmallVector<OwningModuleRef> modules;
  if (getModules(modules, &context).failed())
    return 1;

  /// Find top-level symbol.
  mlir::Operation *toplevelOp = getTopLevelOperation(modules);

  if (toplevelOp == nullptr)
    return 1;

  /// Build backplane.
  DialectSimBackplane backplane(&context, modules, referenceModel);

  /// Build simulators.
  auto standardInterpreter =
      std::make_shared<DialectSimInterpreter>(&context, backplane);
  standardInterpreter->addImpl<StandardArithmeticInterpreter>();
  standardInterpreter->addImpl<MemrefInterpreter>();
  standardInterpreter->addImpl<BuiltinInterpreter>();
  backplane.registerSimulator<FuncOpTransactor>(standardInterpreter);
  backplane.addCallTransactor<CallOpTransactor>();

  /// Handshake simulator
  auto handshakeInterpreter =
      std::make_shared<HandshakeInterpreter>(&context, backplane);
  handshakeInterpreter->addImpl<StandardArithmeticInterpreter>();
  handshakeInterpreter->addImpl<HandshakeInterpreterImpl>();
  backplane.registerSimulator<HandshakeFuncOpTransactor>(handshakeInterpreter);

  /// Go simulate!
  Transaction exitTransaction;
  if (backplane.instantiate(toplevelOp, exitTransaction).failed()) {
    toplevelOp->emitOpError()
        << "Error during simulation of top-level entry op\n";
    return 1;
  }

  /// Print results.
  if (backplane.print(toplevelOp, exitTransaction).failed())
    return 1;

  return 0;
}

} // namespace txvf
} // namespace circt

int main(int argc, char **argv) { circt::txvf::txvfMain(argc, argv); }
