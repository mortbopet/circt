#ifndef CIRCT_TXVF_STANDARDINTERPRETER_H
#define CIRCT_TXVF_STANDARDINTERPRETER_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include "llvm/ADT/TypeSwitch.h"

#include "DialectInterpreter.h"

using namespace llvm;
using namespace mlir;

namespace circt {
namespace txvf {

class FuncOpTransactor : public EntryTransactor {
public:
  using EntryTransactor::EntryTransactor;
  Transaction encode(SimContextPtr &simctx, ArrayRef<Any> outs) final;
  SimContextPtr decode(Operation *target, Transaction &transaction) final;
  static StringRef getOperationName() {
    return mlir::FuncOp::getOperationName();
  }
  LogicalResult print(llvm::raw_fd_ostream &out, Operation *target,
                      Transaction &transaction) final;
};

class CallOpTransactor : public CallTransactor {
public:
  using CallTransactor::CallTransactor;
  Transaction encode(SimContextPtr &simctx, ArrayRef<Any> ins) final;
  void decode(std::vector<Any> &outs, Operation *target,
              Transaction &transaction) final;
  static StringRef getOperationName() {
    return mlir::CallOp::getOperationName();
  }
};

class StandardArithmeticInterpreter : public DialectSimInterpreterImpl {
public:
  using DialectSimInterpreterImpl::DialectSimInterpreterImpl;
  static StringRef getDialectNamespace() {
    return StandardOpsDialect::getDialectNamespace();
  }

  /// Entry
  LogicalResult execute(Operation *op, SimContextPtr &simctx) override;

  LogicalResult execute(mlir::ConstantIndexOp, SimContextPtr &,
                        std::vector<Any> & /*inputs*/,
                        std::vector<Any> & /*outputs*/);
  LogicalResult execute(mlir::ConstantIntOp, SimContextPtr &,
                        std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(mlir::AddIOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::ShiftLeftOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::SignedShiftRightOp, SimContextPtr &,
                        std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(mlir::UnsignedShiftRightOp, SimContextPtr &,
                        std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(mlir::AddFOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::CmpIOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::CmpFOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::SubIOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::SubFOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::MulIOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::MulFOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::SignedDivIOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::AndOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::OrOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::XOrOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::UnsignedDivIOp, SimContextPtr &,
                        std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(mlir::DivFOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::IndexCastOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::SignExtendIOp, SimContextPtr &,
                        std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(mlir::ZeroExtendIOp, SimContextPtr &,
                        std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(mlir::BranchOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::CondBranchOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::ReturnOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::CallOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
};

} // namespace txvf
} // namespace circt

#endif // CIRCT_TXVF_STANDARDINTERPRETER_H
