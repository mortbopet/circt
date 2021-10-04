#include "MemrefInterpreter.h"

namespace circt {
namespace txvf {

LogicalResult MemrefInterpreter::execute(Operation *op, SimContextPtr &simctx) {
  std::vector<Any> inValues;
  std::vector<Any> outValues(op->getNumResults());

  /// Gather input values from the context
  for (auto operand : op->getOperands())
    inValues.push_back(simctx->getValue(operand));

  auto res = llvm::TypeSwitch<Operation *, LogicalResult>(op)
                 .Case<memref::AllocOp, memref::AllocaOp, memref::LoadOp,
                       memref::StoreOp>([&](auto op) {
                   LogicalResult res = execute(op, simctx, inValues, outValues);
                   if (res.failed())
                     return res;

                   /// The matched operations are all non-control flow
                   /// operation. The standard arithmetic interpreter assumes
                   /// that instructions are executed operation, so schedule
                   /// the following operation.
                   ++simctx->instrIter;
                   return res;
                 })
                 .Default([](auto op) {
                   return op->emitOpError() << "Unknown operation!";
                 });

  if (res.failed())
    return failure();

  /// Write output values into the context
  for (auto [simRes, opRes] : llvm::zip(outValues, op->getResults()))
    simctx->setValue(opRes, simRes);

  return success();
}

LogicalResult MemrefInterpreter::execute(mlir::memref::LoadOp op,
                                         SimContextPtr &, std::vector<Any> &in,
                                         std::vector<Any> &out) {
  ArrayRef<int64_t> shape = op.getMemRefType().getShape();
  uint64_t address = 0;
  for (unsigned i = 0; i < shape.size(); ++i)
    address = address * shape[i] + any_cast<APInt>(in[i + 1]).getZExtValue();

  auto mem = any_cast<SimMemoryPtr>(in[0]);
  if (!mem)
    return op.emitError() << "Unknown memory";

  auto result = mem->read(address);
  if (!result.hasValue())
    return op.emitError() << "Out-of-bounds access to memory. Memory has "
                          << mem->size() << " elements but requested element "
                          << address;

  out[0] = result.getValue();
  return success();
}

LogicalResult MemrefInterpreter::execute(mlir::memref::StoreOp op,
                                         SimContextPtr &, std::vector<Any> &in,
                                         std::vector<Any> &) {
  ArrayRef<int64_t> shape = op.getMemRefType().getShape();
  uint64_t address = 0;
  for (unsigned i = 0; i < shape.size(); ++i)
    address = address * shape[i] + any_cast<APInt>(in[i + 2]).getZExtValue();

  auto mem = any_cast<SimMemoryPtr>(in[1]);
  if (mem->write(address, in[0]).failed())
    return op.emitError() << "Out-of-bounds access to memory. Memory has "
                          << mem->size() << " elements but requested element "
                          << address;

  return success();
}

LogicalResult MemrefInterpreter::execute(mlir::memref::AllocOp op,
                                         SimContextPtr &simctx,
                                         std::vector<Any> &,
                                         std::vector<Any> &out) {
  auto mem = std::make_shared<SimMemory>(op.getType());
  simctx->memories.push_back(mem);
  out[0] = mem;
  return success();
}

LogicalResult MemrefInterpreter::execute(mlir::memref::AllocaOp op,
                                         SimContextPtr &simctx,
                                         std::vector<Any> &,
                                         std::vector<Any> &out) {
  auto mem = std::make_shared<SimMemory>(op.getType());
  simctx->memories.push_back(mem);
  out[0] = mem;
  return success();
}

} // namespace txvf
} // namespace circt
