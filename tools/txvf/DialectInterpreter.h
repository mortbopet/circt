

#ifndef CIRCT_TXVF_DIALECTINTERPRETER_H
#define CIRCT_TXVF_DIALECTINTERPRETER_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"

#include "DialectSimulator.h"

using namespace llvm;
using namespace mlir;

namespace circt {
namespace txvf {

class DialectSimInterpreter;

/// DialectSimInterpreterImpl represents a partial dialect interpreter. These
/// provide the capabilities to simulate all or a subset of a given dialects'
/// operations.
class DialectSimInterpreterImpl {
public:
  DialectSimInterpreterImpl(DialectSimInterpreter *parent) : sim(parent) {}
  virtual LogicalResult execute(Operation *op, SimContextPtr &simctx) = 0;

protected:
  DialectSimInterpreter *sim = nullptr;
};

class DialectSimInterpreter : public DialectSim {
public:
  using DialectSim::DialectSim;

  template <typename Impl>
  void addImpl() {
    getContext()->getOrLoadDialect(Impl::getDialectNamespace());
    interpreters[Impl::getDialectNamespace()] =
        std::shared_ptr<DialectSimInterpreterImpl>(
            std::make_shared<Impl>(this));
  }

  virtual LogicalResult execute(SimContextPtr &simctx) override {
#define DEBUG_TYPE "DialectSimInterpreter"

    /// Execute the instruction iterator until control is yielded from this
    /// simulator.
    Operation *priorOp = nullptr;
    while (!simctx->yielded()) {
      Operation *op = &*simctx->instrIter; /// todo(mortbopet): too dangerous?

      if (priorOp && op == priorOp)
        return op->emitOpError()
               << "Instruction iterator was not modified during "
                  "execution of the prior operation";

      LLVM_DEBUG(llvm::dbgs() << "Interpreting: " << op->getName() << " at "
                              << op->getLoc() << "\n");
      auto it = interpreters.find(op->getDialect()->getNamespace());
      assert(it != interpreters.end() &&
             "No interpreter registerred for the operation");
      /// todo(mortbopet): here we're reusing the interpreters... If we want to
      /// be able to support multithreading, this needs to be rethinked.
      if (it->second->execute(op, simctx).failed())
        return failure();
      priorOp = op;
    }
    return success();
  }

protected:
  llvm::DenseMap<StringRef, std::shared_ptr<DialectSimInterpreterImpl>>
      interpreters;
  llvm::SmallVector<StringRef> supportedDialects;
};

} // namespace txvf
} // namespace circt

#endif // CIRCT_TXVF_DIALECTINTERPRETER_H
