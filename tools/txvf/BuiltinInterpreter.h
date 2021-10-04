#ifndef CIRCT_TXVF_BUILTINSIMULATOR_H
#define CIRCT_TXVF_BUILTINSIMULATOR_H

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/TypeSwitch.h"

#include "DialectInterpreter.h"

using namespace llvm;
using namespace mlir;

namespace circt {
namespace txvf {

class BuiltinInterpreter : public DialectSimInterpreterImpl {
public:
  using DialectSimInterpreterImpl::DialectSimInterpreterImpl;
  static StringRef getDialectNamespace() {
    return BuiltinDialect::getDialectNamespace();
  }

  LogicalResult execute(Operation *op, SimContextPtr &simctx) override {
    std::vector<Any> inValues(op->getNumOperands());

    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<FuncOp>([&](auto op) { return execute(op, simctx, inValues); })
        .Default(
            [](auto op) { return op->emitOpError() << "Unknown operation!"; });
  }

  LogicalResult execute(FuncOp op, SimContextPtr &simctx, std::vector<Any> &) {
    /// Set instruction iterator to the first op in the body, and go execute.
    simctx->instrIter = op.getBody().front().begin();
    return success();
  }
};

} // namespace txvf
} // namespace circt

#endif // CIRCT_TXVF_BUILTINSIMULATOR_H
