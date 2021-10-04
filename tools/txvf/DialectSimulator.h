#ifndef CIRCT_TXVF_DIALECTSIMULATOR_H
#define CIRCT_TXVF_DIALECTSIMULATOR_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"

#include "SimUtils.h"
#include "TestLib.h"

using namespace llvm;
using namespace mlir;

namespace circt {
namespace txvf {

class DialectSimBackplane;
class DialectSim {
public:
  explicit DialectSim(MLIRContext *ctx, DialectSimBackplane &backplane)
      : context(ctx), backplane(backplane) {}

  /// todo(mortbopet): make sure docs are up-to-date.
  /// A call op should either map to a well-defined symbol in the source module
  /// itself or an external symbol in another IR. If the latter, this will trap
  /// to the backplane to instantiate a new dialect simulator.
  LogicalResult call(Operation *sourceOp, StringRef calledSymbol,
                     SimContextPtr &simctx, ArrayRef<Any> ins,
                     std::vector<Any> &outs);
  virtual void finish(LogicalResult) {}

  MLIRContext *getContext() { return context; }
  virtual LogicalResult execute(SimContextPtr &simctx) = 0;

private:
  MLIRContext *context = nullptr;
  DialectSimBackplane &backplane;
};

class DialectSimBackplane {
public:
  DialectSimBackplane(MLIRContext *ctx,
                      SmallVectorImpl<OwningModuleRef> &modules, StringRef ref)
      : context(ctx), modules(modules), ref(ref) {}

  template <typename EntryTransactorImpl>
  void registerSimulator(std::shared_ptr<DialectSim> sim) {
    auto transactor = std::make_shared<EntryTransactorImpl>(this);
    entryTransactors[EntryTransactorImpl::getOperationName()] =
        std::shared_ptr<EntryTransactor>(transactor);
    simulators[transactor.get()] = sim;
  }

  LogicalResult print(Operation *op, Transaction &exitTransaction);

  /// Instantiates a simulator for a top-level operation with an empty
  /// transaction.
  LogicalResult instantiate(Operation *op, Transaction &exitTransaction) {
    Transaction enterTransaction(std::make_shared<SimContext>());
    return enter(op, enterTransaction, exitTransaction);
  }

  /// A transactor will call this function with a symbol to execute + an encoded
  /// transaction.
  LogicalResult call(Operation *sourceOp, StringRef symbol,
                     SimContextPtr &sourceContext, ArrayRef<Any> ins,
                     std::vector<Any> &outs);

  /// Enters into a simulator for an operation with the given transaction. An
  /// entry transactor for the operation will be used to decode the transaction.
  LogicalResult enter(Operation *targetOp, Transaction &enterTransaction,
                      Transaction &exitTransaction);

  template <typename TransactorImpl>
  void addCallTransactor() {
    auto transactor = std::make_shared<TransactorImpl>(this);
    callTransactors[TransactorImpl::getOperationName()] =
        std::shared_ptr<CallTransactor>(transactor);
  }

  /// Validates a set of exit transactions against the golden model.
  LogicalResult validate(Location location,
                         llvm::DenseMap<StringRef, Transaction> &transactions);

private:
  /// Maintain a mapping between a dialect name and the simulator which supporst
  /// execution of ops within that dialect.
  ///
  /// todo(mortbopet): We'd want to split std to i.e. arithmetic and
  /// control-flow parts, so this logic doesn't necessarily hold true.
  llvm::DenseMap<EntryTransactor *, std::shared_ptr<DialectSim>> simulators;

  /// todo(mortbopet): document
  llvm::DenseMap<StringRef, std::shared_ptr<EntryTransactor>> entryTransactors;
  llvm::DenseMap<StringRef, std::shared_ptr<CallTransactor>> callTransactors;

  MLIRContext *context;
  SmallVectorImpl<OwningModuleRef> &modules;
  /// Dialect namespace of the reference model.
  StringRef ref;
  TestLib lib;
};

} // namespace txvf
} // namespace circt

#endif // CIRCT_TXVF_DIALECTSIMULATOR_H
