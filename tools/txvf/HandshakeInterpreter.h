#ifndef CIRCT_TXVF_STANDARDSIMULATOR_H
#define CIRCT_TXVF_STANDARDSIMULATOR_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/Dialect.h"

#include "llvm/ADT/TypeSwitch.h"

#include "DialectInterpreter.h"

using namespace llvm;
using namespace mlir;

namespace circt {
namespace txvf {

class HandshakeFuncOpTransactor : public EntryTransactor {
public:
  using EntryTransactor::EntryTransactor;
  Transaction encode(SimContextPtr &simctx, ArrayRef<Any> outs) final;
  SimContextPtr decode(Operation *target, Transaction &transaction) final;
  static StringRef getOperationName() {
    return handshake::FuncOp::getOperationName();
  }
  LogicalResult print(llvm::raw_fd_ostream &out, Operation *target,
                      Transaction &transaction) final;
};

class HandshakeInstanceOpTransactor : public CallTransactor {
public:
  using CallTransactor::CallTransactor;
  Transaction encode(SimContextPtr &simctx, ArrayRef<Any> ins) final;
  void decode(std::vector<Any> &outs, Operation *target,
              Transaction &transaction) final;
  static StringRef getOperationName() {
    return handshake::InstanceOp::getOperationName();
  }
};

class HandshakeInterpreter : public DialectSimInterpreter {
public:
  using DialectSimInterpreter::DialectSimInterpreter;
  static StringRef getDialectNamespace() {
    return handshake::HandshakeDialect::getDialectNamespace();
  }
  virtual LogicalResult execute(SimContextPtr &simctx) final;
};

class HandshakeInterpreterImpl : public DialectSimInterpreterImpl {
public:
  using DialectSimInterpreterImpl::DialectSimInterpreterImpl;
  static StringRef getDialectNamespace() {
    return handshake::HandshakeDialect::getDialectNamespace();
  }

  /// Entry
  virtual LogicalResult execute(Operation *op, SimContextPtr &simctx) final;

  /// Generic cases
  LogicalResult execute(handshake::ReturnOp, SimContextPtr &,
                        std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(handshake::JoinOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(handshake::StoreOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(handshake::SinkOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(handshake::MergeOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(handshake::BranchOp, SimContextPtr &,
                        std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(handshake::ForkOp, SimContextPtr &, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(handshake::ConstantOp, SimContextPtr &,
                        std::vector<Any> &, std::vector<Any> &);

  /// Special cases
  LogicalResult execute(handshake::LoadOp, SimContextPtr &);
  LogicalResult execute(handshake::MemoryOp, SimContextPtr &);
  LogicalResult execute(handshake::ConditionalBranchOp, SimContextPtr &);
  LogicalResult execute(handshake::MergeOp, SimContextPtr &);
  LogicalResult execute(handshake::ControlMergeOp, SimContextPtr &);
  LogicalResult execute(handshake::MuxOp, SimContextPtr &);
  LogicalResult execute(handshake::SinkOp, SimContextPtr &);
};

} // namespace txvf
} // namespace circt

#endif // CIRCT_TXVF_HANDSHAKEINTERPRETER_H
