#ifndef CIRCT_TXVF_MEMREFSIMULATOR_H
#define CIRCT_TXVF_MEMREFSIMULATOR_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include "llvm/ADT/TypeSwitch.h"

#include "DialectInterpreter.h"

using namespace llvm;
using namespace mlir;

namespace circt {
namespace txvf {

class MemrefInterpreter : public DialectSimInterpreterImpl {
public:
  using DialectSimInterpreterImpl::DialectSimInterpreterImpl;
  static StringRef getDialectNamespace() {
    return memref::MemRefDialect::getDialectNamespace();
  }

  /// Entry
  LogicalResult execute(Operation *op, SimContextPtr &simctx) override;

  /// Specializations
  LogicalResult execute(memref::LoadOp, SimContextPtr &simctx,
                        std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(memref::StoreOp, SimContextPtr &simctx,
                        std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(memref::AllocOp, SimContextPtr &simctx,
                        std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(memref::AllocaOp, SimContextPtr &simctx,
                        std::vector<Any> &, std::vector<Any> &);
};

} // namespace txvf
} // namespace circt

#endif // CIRCT_TXVF_MEMREFSIMULATOR_H
