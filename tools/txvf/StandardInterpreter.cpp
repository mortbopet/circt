#include "StandardInterpreter.h"

namespace circt {
namespace txvf {

//===----------------------------------------------------------------------===//
// Transactor
//===----------------------------------------------------------------------===//

LogicalResult FuncOpTransactor::print(llvm::raw_fd_ostream &out,
                                      Operation *target,
                                      Transaction &transaction) {
  auto funcOp = dyn_cast<mlir::FuncOp>(target);
  assert(funcOp && "expected to decode to a FuncOp target");

  if (transaction.results.size() != funcOp.getType().getNumResults())
    return target->emitOpError() << "Expected " << transaction.results.size()
                                 << " results in the transaction but had "
                                 << transaction.results.size();

  for (auto [resType, res] :
       llvm::zip(funcOp.getType().getResults(), transaction.results)) {
    out << printAnyValueWithType(resType, res.value) << " ";
  }
  out << "\n";
  return success();
}

/// todo: this seems general... Will it also look like this for more quirky IRs?
Transaction FuncOpTransactor::encode(SimContextPtr &simctx,
                                     ArrayRef<Any> outs) {
  Transaction tx;
  llvm::transform(outs, std::back_inserter(tx.results),
                  [&](Any out) { return Transaction::Value(out, Type()); });
  return tx;
}

SimContextPtr FuncOpTransactor::decode(Operation *target,
                                       Transaction &transaction) {
  auto newCtx = std::make_shared<SimContext>();

  auto funcOp = dyn_cast<mlir::FuncOp>(target);
  assert(funcOp && "expected to decode to a FuncOp target");

  /// Create associations with the entry block arguments in the func op.
  auto &entryBlock = funcOp.getBody().front();

  assert(entryBlock.getNumArguments() == transaction.args.size() &&
         "Mismatch between expected and actual number of arguments");

  for (auto [blockArg, txArg] :
       llvm::zip(entryBlock.getArguments(), transaction.args))
    newCtx->setValue(blockArg, txArg.value);

  return newCtx;
}

Transaction CallOpTransactor::encode(SimContextPtr &simctx, ArrayRef<Any> ins) {
  Transaction tx(std::make_shared<SimContext>());
  llvm::transform(ins, std::back_inserter(tx.args),
                  [&](Any in) { return Transaction::Value(in, Type()); });
  return tx;
}

/// todo: this seems general... Will it also look like this for more quirky IRs?
void CallOpTransactor::decode(std::vector<Any> &outs, Operation *target,
                              Transaction &transaction) {
  outs.clear();
  llvm::transform(transaction.results, std::back_inserter(outs),
                  [&](Transaction::Value res) { return res.value; });
}

//===----------------------------------------------------------------------===//
// Interpreter
//===----------------------------------------------------------------------===//

LogicalResult StandardArithmeticInterpreter::execute(Operation *op,
                                                     SimContextPtr &simctx) {
  std::vector<Any> inValues;
  std::vector<Any> outValues(op->getNumResults());

  /// Gather input values from the context
  for (auto operand : op->getOperands())
    inValues.push_back(simctx->getValue(operand));

  auto res =
      llvm::TypeSwitch<Operation *, LogicalResult>(op)
          .Case<mlir::ConstantIndexOp, mlir::ConstantIntOp, mlir::AddIOp,
                mlir::AddFOp, mlir::CmpIOp, mlir::CmpFOp, mlir::SubIOp,
                mlir::SubFOp, mlir::MulIOp, mlir::MulFOp, mlir::SignedDivIOp,
                mlir::UnsignedDivIOp, mlir::DivFOp, mlir::IndexCastOp,
                mlir::SignExtendIOp, mlir::ZeroExtendIOp, mlir::CallOp,
                mlir::AndOp, mlir::OrOp, mlir::XOrOp, mlir::ShiftLeftOp,
                mlir::SignedShiftRightOp, mlir::UnsignedShiftRightOp>(
              [&](auto op) {
                LogicalResult res = execute(op, simctx, inValues, outValues);
                if (res.failed())
                  return res;

                /// The matched operations are all non-control flow operation.
                /// The standard arithmetic interpreter assumes that
                /// instructions are executed operation, so schedule the
                /// following operation.
                ++simctx->instrIter;
                return res;
              })
          .Case<mlir::BranchOp, mlir::CondBranchOp>(
              [&](auto op) { return execute(op, simctx, inValues, outValues); })
          .Case<mlir::ReturnOp>(
              [&](auto op) { return execute(op, simctx, inValues, outValues); })
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

LogicalResult StandardArithmeticInterpreter::execute(mlir::ConstantIndexOp op,
                                                     SimContextPtr &,
                                                     std::vector<Any> &,
                                                     std::vector<Any> &out) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>("value");
  out[0] = attr.getValue().sextOrTrunc(32);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::ConstantIntOp op,
                                                     SimContextPtr &,
                                                     std::vector<Any> &,
                                                     std::vector<Any> &out) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>("value");
  out[0] = attr.getValue();
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::AddIOp,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) + any_cast<APInt>(in[1]);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::AddFOp,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) + any_cast<APFloat>(in[1]);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::CmpIOp op,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  APInt in0 = any_cast<APInt>(in[0]);
  APInt in1 = any_cast<APInt>(in[1]);
  APInt out0(1, mlir::applyCmpPredicate(op.getPredicate(), in0, in1));
  out[0] = out0;
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::AndOp,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) & any_cast<APInt>(in[1]);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::ShiftLeftOp,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) << any_cast<APInt>(in[1]);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::SignedShiftRightOp,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]).ashr(any_cast<APInt>(in[1]));
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::UnsignedShiftRightOp,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]).lshr(any_cast<APInt>(in[1]));
  return success();
}
LogicalResult StandardArithmeticInterpreter::execute(mlir::OrOp,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) | any_cast<APInt>(in[1]);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::XOrOp,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) ^ any_cast<APInt>(in[1]);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::CmpFOp op,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  APFloat in0 = any_cast<APFloat>(in[0]);
  APFloat in1 = any_cast<APFloat>(in[1]);
  APInt out0(1, mlir::applyCmpPredicate(op.getPredicate(), in0, in1));
  out[0] = out0;
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::SubIOp,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) - any_cast<APInt>(in[1]);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::SubFOp,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) + any_cast<APFloat>(in[1]);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::MulIOp,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) * any_cast<APInt>(in[1]);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::MulFOp,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) * any_cast<APFloat>(in[1]);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::SignedDivIOp op,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  if (!any_cast<APInt>(in[1]).getZExtValue())
    return op.emitError() << "Division By Zero!";

  out[0] = any_cast<APInt>(in[0]).sdiv(any_cast<APInt>(in[1]));
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::UnsignedDivIOp op,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  if (!any_cast<APInt>(in[1]).getZExtValue())
    return op.emitError() << "Division By Zero!";
  out[0] = any_cast<APInt>(in[0]).udiv(any_cast<APInt>(in[1]));
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::DivFOp,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) / any_cast<APFloat>(in[1]);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::IndexCastOp,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  out[0] = in[0];
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::SignExtendIOp op,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  int64_t width = op.getType().getIntOrFloatBitWidth();
  out[0] = any_cast<APInt>(in[0]).sext(width);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::ZeroExtendIOp op,
                                                     SimContextPtr &,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &out) {
  int64_t width = op.getType().getIntOrFloatBitWidth();
  out[0] = any_cast<APInt>(in[0]).zext(width);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::BranchOp branchOp,
                                                     SimContextPtr &simctx,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &) {
  mlir::Block *dest = branchOp.getDest();

  // Write block arguments
  for (auto out : enumerate(dest->getArguments()))
    simctx->setValue(out.value(), in[out.index()]);

  simctx->instrIter = dest->begin();
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(
    mlir::CondBranchOp condBranchOp, SimContextPtr &simctx,
    std::vector<Any> &in, std::vector<Any> &) {
  APInt condition = any_cast<APInt>(in[0]);
  mlir::Block *dest;
  std::vector<Any> inArgs;
  if (condition != 0) {
    dest = condBranchOp.getTrueDest();
    inArgs.resize(condBranchOp.getNumTrueOperands());
    for (auto in : enumerate(condBranchOp.getTrueOperands()))
      inArgs[in.index()] = simctx->getValue(in.value());

  } else {
    dest = condBranchOp.getFalseDest();
    inArgs.resize(condBranchOp.getNumFalseOperands());
    for (auto in : enumerate(condBranchOp.getFalseOperands()))
      inArgs[in.index()] = simctx->getValue(in.value());
  }
  for (auto out : enumerate(dest->getArguments()))
    simctx->setValue(out.value(), inArgs[out.index()]);

  simctx->instrIter = dest->begin();
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::ReturnOp,
                                                     SimContextPtr &simctx,
                                                     std::vector<Any> &in,
                                                     std::vector<Any> &) {
  simctx->yield(in);
  return success();
}

LogicalResult StandardArithmeticInterpreter::execute(mlir::CallOp callOp,
                                                     SimContextPtr &simctx,
                                                     std::vector<Any> &ins,
                                                     std::vector<Any> &outs) {
  if (sim->call(callOp, callOp.callee(), simctx, ins, outs).failed())
    return failure();
  return success();
}

} // namespace txvf
} // namespace circt
