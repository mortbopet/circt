#include "HandshakeInterpreter.h"

#include "llvm/ADT/TypeSwitch.h"

#include <list>

namespace circt {
namespace txvf {

namespace {
struct ReadyList {
  std::list<mlir::Operation *> ops;

  void scheduleIfNeeded(mlir::Operation *op) {
    if (llvm::find(ops, op) == ops.end()) {
      LLVM_DEBUG(llvm::dbgs() << "\tscheduling " << *op << "\n");
      ops.push_back(op);
    }
  }

  void scheduleUses(mlir::Value value) {
    for (auto &use : value.getUses())
      scheduleIfNeeded(use.getOwner());
  }
};

struct HandshakeContext {
  bool reschedule = false;
  ReadyList readylist;
  llvm::DenseMap<unsigned, SimMemoryPtr> localMemories;
};
using HandshakeContextPtr = std::shared_ptr<HandshakeContext>;

static HandshakeContextPtr getHandshakeContext(SimContextPtr &simContext) {
  auto handshakeCtx =
      simContext->userPtrs[handshake::HandshakeDialect::getDialectNamespace()];
  assert(handshakeCtx.hasValue() &&
         "Expected handshake user context to be set");
  return any_cast<HandshakeContextPtr>(handshakeCtx);
}

} // namespace

//===----------------------------------------------------------------------===//
// Transactor
//===----------------------------------------------------------------------===//

LogicalResult HandshakeFuncOpTransactor::print(llvm::raw_fd_ostream &out,
                                               Operation *target,
                                               Transaction &transaction) {
  auto funcOp = dyn_cast<handshake::FuncOp>(target);
  assert(funcOp && "expected to decode to a handshake::FuncOp target");

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
Transaction HandshakeFuncOpTransactor::encode(SimContextPtr &,
                                              ArrayRef<Any> outs) {
  /// Return everything but the control argument
  Transaction tx;
  for (unsigned i = 0; i < outs.size() - 1; ++i)
    tx.results.push_back(Transaction::Value(outs[i], Type()));
  return tx;
}

SimContextPtr HandshakeFuncOpTransactor::decode(Operation *target,
                                                Transaction &transaction) {
  auto newCtx = std::make_shared<SimContext>();

  auto funcOp = dyn_cast<handshake::FuncOp>(target);
  assert(funcOp && "expected to decode to a handshake::FuncOp target");

  /// Create associations with the entry block arguments in the func op. We
  /// expect the transaction to carry a value for each of the inputs apart from
  /// the control input.
  auto &entryBlock = funcOp.getBody().front();
  assert((entryBlock.getNumArguments() - 1) == transaction.args.size() &&
         "Mismatch between expected and actual number of arguments");

  /// Write the arguments into the context.
  for (auto [blockArg, txArg] :
       llvm::zip(entryBlock.getArguments(), transaction.args))
    newCtx->setValue(blockArg, txArg.value);

  /// Add the none argument as implicitly valid.
  APInt apnonearg(1, 0);
  newCtx->setValue(entryBlock.getArguments().back(), apnonearg);

  return newCtx;
}

Transaction HandshakeInstanceOpTransactor::encode(SimContextPtr &,
                                                  ArrayRef<Any> ins) {
  Transaction tx(std::make_shared<SimContext>());
  llvm::transform(ins, std::back_inserter(tx.args),
                  [&](Any in) { return Transaction::Value(in, Type()); });
  return tx;
}

/// todo: this seems general... Will it also look like this for more quirky IRs?
void HandshakeInstanceOpTransactor::decode(std::vector<Any> &outs, Operation *,
                                           Transaction &transaction) {
  outs.clear();
  llvm::transform(transaction.results, std::back_inserter(outs),
                  [&](Transaction::Value res) { return res.value; });
}

//===----------------------------------------------------------------------===//
// Interpreter
//===----------------------------------------------------------------------===//

LogicalResult HandshakeInterpreter::execute(SimContextPtr &simctx) {
  auto funcOp = dyn_cast<handshake::FuncOp>(simctx->instrIter);
  assert(funcOp && "Expected to be initialized with a handshake::FuncOp");

  HandshakeContextPtr handshakeCtx = std::make_shared<HandshakeContext>();
  simctx->userPtrs[getDialectNamespace()] = handshakeCtx;

  /// Pre-allocate memory.
  for (auto memoryOp : funcOp.getOps<handshake::MemoryOp>())
    handshakeCtx->localMemories[memoryOp.getID()] =
        std::make_shared<SimMemory>(memoryOp.getMemRefType());

  /// Schedule starting from the block arguments.
  for (auto blockArg : funcOp.getBody().front().getArguments())
    handshakeCtx->readylist.scheduleUses(blockArg);

  assert(handshakeCtx->readylist.ops.size() != 0 &&
         "No operations ready to execute?");

  Operation *priorOp = nullptr;
  while (!simctx->yielded()) {
    assert(handshakeCtx->readylist.ops.size() != 0 &&
           "Expected some operation to be ready to execute");
    mlir::Operation *op = &*handshakeCtx->readylist.ops.front();
    handshakeCtx->readylist.ops.pop_front();

    /// Ensure that we're not re-executing an op twice in a row.
    if (priorOp && op == priorOp)
      return op->emitOpError() << "Stuck in loop";
    priorOp = op;

    /// Locate interpreter for the operation
    LLVM_DEBUG(llvm::dbgs() << "Interpreting: " << *op << "\n");
    StringRef opDialect = op->getDialect()->getNamespace();
    auto it = interpreters.find(opDialect);
    if (it == interpreters.end())
      return op->emitOpError()
             << "No interpreter registerred for the operation";

    auto interpreter = it->second;
    if (interpreter->execute(op, simctx).failed())
      return failure();

    /// Check if rescheduling was requested by the operation itself
    if (handshakeCtx->reschedule) {
      LLVM_DEBUG(llvm::dbgs() << "\tRescheduling: " << *op << "\n");
      handshakeCtx->readylist.ops.push_back(op);
      handshakeCtx->reschedule = false;
      continue;
    }

    /// If we executed with the standard interpreter, consume the inputs and
    /// schedule uses of any results that the op generated.
    if (opDialect != getDialectNamespace()) {
      for (auto arg : op->getOperands())
        simctx->eraseValue(arg);
      for (auto res : op->getResults())
        handshakeCtx->readylist.scheduleUses(res);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Interpreter implementation
//===----------------------------------------------------------------------===//

// Returns whether the precondition holds for a general op to execute
static bool isReadyToExecute(ValueRange ins, ValueRange outs,
                             SimContextPtr &simctx) {
  if (!llvm::any_of(ins, [&](auto in) { return simctx->hasValue(in); }))
    return false;

  if (llvm::any_of(outs, [&](auto out) { return simctx->hasValue(out); }))
    return false;

  return true;
}

template <typename HandshakeType>
bool isSpecialReadyToExecute(SimContextPtr &simctx, HandshakeType op);

template <>
bool isSpecialReadyToExecute<handshake::LoadOp>(SimContextPtr &simctx,
                                                handshake::LoadOp op) {
  mlir::Value address = op->getOperand(0);
  mlir::Value data = op->getOperand(1);
  mlir::Value nonce = op->getOperand(2);
  if ((simctx->hasValue(address) && !simctx->hasValue(nonce)) ||
      (!simctx->hasValue(address) && simctx->hasValue(nonce)) ||
      (!simctx->hasValue(address) && !simctx->hasValue(nonce) &&
       !simctx->hasValue(data)))
    return false;

  return true;
}

template <>
bool isSpecialReadyToExecute<handshake::MemoryOp>(SimContextPtr &simctx,
                                                  handshake::MemoryOp op) {
  int opIndex = 0;

  for (unsigned i = 0; i < op.getStCount().getZExtValue(); i++) {
    mlir::Value data = op->getOperand(opIndex++);
    mlir::Value address = op->getOperand(opIndex++);
    if ((!simctx->hasValue(data) || !simctx->hasValue(address)))
      return false;
  }

  for (unsigned i = 0; i < op.getLdCount().getZExtValue(); i++) {
    mlir::Value address = op->getOperand(opIndex++);
    if (!simctx->hasValue(address))
      return false;
  }
  return true;
}

template <>
bool isSpecialReadyToExecute<handshake::ConditionalBranchOp>(
    SimContextPtr &simctx, handshake::ConditionalBranchOp op) {
  mlir::Value control = op->getOperand(0);
  if (!simctx->hasValue(control))
    return false;
  mlir::Value in = op->getOperand(1);
  if (!simctx->hasValue(in))
    return false;
  return true;
}

template <>
bool isSpecialReadyToExecute<handshake::MuxOp>(SimContextPtr &simctx,
                                               handshake::MuxOp op) {
  mlir::Value control = op->getOperand(0);
  if (!simctx->hasValue(control))
    return false;
  mlir::Value in = llvm::any_cast<APInt>(simctx->getValue(control)) == 0
                       ? op->getOperand(1)
                       : op->getOperand(2);
  if (!simctx->hasValue(in))
    return false;
  return true;
}

LogicalResult HandshakeInterpreterImpl::execute(Operation *op,
                                                SimContextPtr &simctx) {
  HandshakeContextPtr handshakeCtx = getHandshakeContext(simctx);
  std::vector<Any> inValues(op->getNumOperands());
  std::vector<Any> outValues(op->getNumResults());
  handshakeCtx->reschedule = false;

  /// Check whether the op is ready to execute.
  bool ready =
      llvm::TypeSwitch<Operation *, bool>(op)
          /// Regular cases; execution invariant defined by isReadyToExecute.
          .Case<handshake::BranchOp, handshake::ReturnOp, handshake::ForkOp,
                handshake::ConstantOp, handshake::StoreOp, handshake::JoinOp>(
              [&](auto op) {
                return isReadyToExecute(op->getOperands(), op->getResults(),
                                        simctx);
              })
          /// Always ready to execute when in the ready list.
          .Case<handshake::MergeOp, handshake::ControlMergeOp,
                handshake::SinkOp>([&](auto) { return true; })
          /// Special cases.
          .Case<handshake::LoadOp, handshake::MemoryOp,
                handshake::ConditionalBranchOp, handshake::MuxOp>(
              [&](auto op) { return isSpecialReadyToExecute(simctx, op); })
          .Default([](auto op) {
            op->emitOpError() << "Unknown operation!" << op->getName();
            assert(false);
            return false;
          });

  /// If the op was not ready to execute, raise the reschedule flag.
  if (!ready) {
    handshakeCtx->reschedule = true;
    return success();
  }

  /// Go execute!
  auto res =
      llvm::TypeSwitch<Operation *, LogicalResult>(op)
          .Case<handshake::BranchOp, handshake::ReturnOp, handshake::ForkOp,
                handshake::ConstantOp, handshake::StoreOp, handshake::JoinOp>(
              [&](auto op) {
                /// Gather input values from the context.
                std::vector<Any> inValues;
                for (auto operand : op->getOperands())
                  inValues.push_back(simctx->getValue(operand));

                if (execute(op, simctx, inValues, outValues).failed())
                  return failure();

                /// Consume the inputs.
                for (mlir::Value in : op->getOperands())
                  simctx->eraseValue(in);

                /// Write output values into the context.
                for (auto [simRes, opRes] :
                     llvm::zip(outValues, op->getResults()))
                  simctx->setValue(opRes, simRes);

                /// Try to schedule users of the result values.
                for (auto out : op->getResults())
                  handshakeCtx->readylist.scheduleUses(out);
                return success();
              })
          /// Special cases.
          .Case<handshake::LoadOp, handshake::MemoryOp,
                handshake::ConditionalBranchOp, handshake::MergeOp,
                handshake::MuxOp, handshake::ControlMergeOp, handshake::SinkOp>(
              [&](auto op) { return execute(op, simctx); })
          .Default([](auto op) {
            return op->emitOpError() << "Unknown operation!";
          });

  if (res.failed())
    return failure();

  return success();
}

LogicalResult HandshakeInterpreterImpl::execute(handshake::ReturnOp,
                                                SimContextPtr &simctx,
                                                std::vector<Any> &ins,
                                                std::vector<Any> &) {
  simctx->yield(ins);
  return success();
}

LogicalResult HandshakeInterpreterImpl::execute(handshake::StoreOp,
                                                SimContextPtr &,
                                                std::vector<Any> &ins,
                                                std::vector<Any> &outs) {
  // Forward the address and data to the memory op.
  outs[0] = ins[0];
  outs[1] = ins[1];
  return success();
}

LogicalResult HandshakeInterpreterImpl::execute(handshake::JoinOp,
                                                SimContextPtr &,
                                                std::vector<Any> &ins,
                                                std::vector<Any> &outs) {
  outs[0] = ins[0];
  return success();
}

LogicalResult HandshakeInterpreterImpl::execute(handshake::BranchOp,
                                                SimContextPtr &,
                                                std::vector<Any> &ins,
                                                std::vector<Any> &outs) {
  outs[0] = ins[0];
  return success();
}

LogicalResult HandshakeInterpreterImpl::execute(handshake::ForkOp,
                                                SimContextPtr &,
                                                std::vector<Any> &ins,
                                                std::vector<Any> &outs) {
  for (auto &out : outs)
    out = ins[0];
  return success();
}

LogicalResult HandshakeInterpreterImpl::execute(handshake::ConstantOp op,
                                                SimContextPtr &,
                                                std::vector<Any> &,
                                                std::vector<Any> &outs) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>("value");
  outs[0] = attr.getValue();
  return success();
}

LogicalResult HandshakeInterpreterImpl::execute(handshake::LoadOp op,
                                                SimContextPtr &simctx) {
  mlir::Value address = op->getOperand(0);
  mlir::Value data = op->getOperand(1);
  mlir::Value nonce = op->getOperand(2);
  mlir::Value addressOut = op->getResult(1);
  mlir::Value dataOut = op->getResult(0);

  auto handshakeCtx = getHandshakeContext(simctx);

  if (simctx->hasValue(address) && simctx->hasValue(nonce)) {
    auto addressValue = simctx->getValue(address);
    auto nonceValue = simctx->getValue(nonce);
    simctx->setValue(addressOut, addressValue);
    handshakeCtx->readylist.scheduleUses(addressOut);
    // Consume the inputs.
    simctx->eraseValue(address);
    simctx->eraseValue(nonce);
  } else if (simctx->hasValue(data)) {
    auto dataValue = simctx->getValue(data);
    handshakeCtx->readylist.scheduleUses(dataOut);
    // Consume the inputs.
    simctx->eraseValue(data);
  } else
    llvm_unreachable("why?");

  return success();
}

LogicalResult HandshakeInterpreterImpl::execute(handshake::MemoryOp op,
                                                SimContextPtr &simctx) {
  auto handshakeCtx = getHandshakeContext(simctx);

  int opIndex = 0;
  unsigned id = op.getID(); // The ID of this memory.
  auto memory = handshakeCtx->localMemories[id];

  /// Perform stores
  for (unsigned i = 0; i < op.getStCount().getZExtValue(); i++) {
    mlir::Value data = op->getOperand(opIndex++);
    mlir::Value address = op->getOperand(opIndex++);
    mlir::Value nonceOut = op->getResult(op.getLdCount().getZExtValue() + i);

    auto addressValue = simctx->getValue(address);
    auto dataValue = simctx->getValue(data);

    unsigned offset = llvm::any_cast<APInt>(addressValue).getZExtValue();
    if (memory->write(offset, dataValue).failed())
      return op.emitOpError() << "Store failed";

    // Implicit none argument
    APInt apnonearg(1, 0);
    simctx->setValue(nonceOut, apnonearg);
    handshakeCtx->readylist.scheduleUses(nonceOut);
    // Consume the inputs.
    simctx->eraseValue(data);
    simctx->eraseValue(address);
  }

  /// Perform loads
  for (unsigned i = 0; i < op.getLdCount().getZExtValue(); i++) {
    mlir::Value address = op->getOperand(opIndex++);
    mlir::Value dataOut = op->getResult(i);
    mlir::Value nonceOut = op->getResult(op.getLdCount().getZExtValue() +
                                         op.getStCount().getZExtValue() + i);

    auto addressValue = simctx->getValue(address);
    unsigned offset = llvm::any_cast<APInt>(addressValue).getZExtValue();

    auto readVal = memory->read(offset);
    if (!readVal.hasValue())
      return op.emitOpError() << "Read failed";

    simctx->setValue(dataOut, readVal.getValue());
    // Implicit none argument
    APInt apnonearg(1, 0);
    simctx->setValue(nonceOut, apnonearg);
    handshakeCtx->readylist.scheduleUses(dataOut);
    handshakeCtx->readylist.scheduleUses(nonceOut);
    // Consume the inputs.
    simctx->eraseValue(address);
  }

  return success();
}

LogicalResult
HandshakeInterpreterImpl::execute(handshake::ConditionalBranchOp op,
                                  SimContextPtr &simctx) {
  auto handshakeCtx = getHandshakeContext(simctx);
  mlir::Value control = op->getOperand(0);
  auto controlValue = simctx->getValue(control);
  mlir::Value in = op->getOperand(1);
  auto inValue = simctx->getValue(in);
  mlir::Value out = llvm::any_cast<APInt>(controlValue) != 0 ? op->getResult(0)
                                                             : op->getResult(1);
  simctx->setValue(out, inValue);
  handshakeCtx->readylist.scheduleUses(out);

  // Consume the inputs.
  simctx->eraseValue(control);
  simctx->eraseValue(in);
  return success();
}

LogicalResult HandshakeInterpreterImpl::execute(handshake::MergeOp op,
                                                SimContextPtr &simctx) {
  auto handshakeCtx = getHandshakeContext(simctx);
  bool found = false;
  for (mlir::Value in : op->getOperands()) {
    if (simctx->hasValue(in)) {
      if (found)
        return op->emitOpError("More than one valid input to Merge!");
      auto t = simctx->getValue(in);
      simctx->setValue(op->getResult(0), t);

      // Consume the inputs.
      simctx->eraseValue(in);
      found = true;
    }
  }

  if (!found)
    return op->emitOpError("No valid input to Merge!");

  handshakeCtx->readylist.scheduleUses(op.getResult());
  return success();
}

LogicalResult HandshakeInterpreterImpl::execute(handshake::MuxOp op,
                                                SimContextPtr &simctx) {
  mlir::Value control = op->getOperand(0);
  auto controlValue = simctx->getValue(control);
  mlir::Value in = llvm::any_cast<APInt>(controlValue) == 0 ? op->getOperand(1)
                                                            : op->getOperand(2);
  simctx->setValue(op->getResult(0), simctx->getValue(in));
  auto handshakeCtx = getHandshakeContext(simctx);

  // Consume the inputs.
  simctx->eraseValue(control);
  simctx->eraseValue(in);
  handshakeCtx->readylist.scheduleUses(op.getResult());
  return success();
}

LogicalResult HandshakeInterpreterImpl::execute(handshake::ControlMergeOp op,
                                                SimContextPtr &simctx) {
  bool found = false;
  int i = 0;
  for (mlir::Value in : op->getOperands()) {
    if (simctx->hasValue(in)) {
      if (found)
        return op->emitOpError("More than one valid input to CMerge!");
      simctx->setValue(op->getResult(0), simctx->getValue(in));
      simctx->setValue(op->getResult(1), APInt(32, i));

      // Consume the inputs.
      simctx->eraseValue(in);
      found = true;
    }
    i++;
  }
  if (!found)
    return op->emitOpError("No valid input to CMerge!");

  auto handshakeCtx = getHandshakeContext(simctx);
  for (auto res : op.getResults())
    handshakeCtx->readylist.scheduleUses(res);

  return success();
}

LogicalResult HandshakeInterpreterImpl::execute(handshake::SinkOp op,
                                                SimContextPtr &simctx) {
  simctx->eraseValue(op.getOperand());
  return success();
}

} // namespace txvf
} // namespace circt
