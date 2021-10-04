#include "DialectSimulator.h"

namespace circt {
namespace txvf {

static void assertTransaction(const Transaction &transaction) {
  for (auto &vec : {transaction.args, transaction.results})
    for (auto &v : vec)
      assert(v.value.hasValue() && "Null values not allowed");
}

LogicalResult DialectSim::call(Operation *sourceOp, StringRef calledSymbol,
                               SimContextPtr &simctx, ArrayRef<Any> ins,
                               std::vector<Any> &outs) {
  if (backplane.call(sourceOp, calledSymbol, simctx, ins, outs).failed())
    return failure();
  return success();
}

#define DEBUG_TYPE "DialectSimBackplane"

/// Returns true if the provided op has a definition. This is intended to be a
/// general function for figuring out whether any call-like symbol (which may be
/// predeclared) has a definition.
static bool isDefinedSymbol(Operation *op) {
  if (op->getNumRegions() == 0)
    return false;
  if (op->getRegion(0).getBlocks().size() == 0)
    return false;
  return true;
}

static LogicalResult compare(Location loc, const Transaction &ref,
                             const Transaction &tst) {
  if (ref.results.size() != tst.results.size())
    return emitError(loc) << "Mismatch in number of result values. Expected "
                          << ref.results.size() << " but got "
                          << tst.results.size();
  return success();
}

LogicalResult DialectSimBackplane::validate(
    Location loc, llvm::DenseMap<StringRef, Transaction> &transactions) {
  Transaction *referenceTransaction = nullptr;

  for (auto &[ns, transaction] : transactions) {
    assertTransaction(transaction);
    if (ns == ref)
      referenceTransaction = &transaction;
  }

  if (referenceTransaction == nullptr)
    return emitError(loc) << "Reference transaction of dialect '" << ref
                          << "' not found";

  for (auto &[ns, transaction] : transactions) {
    if (ns == ref)
      continue;
    LLVM_DEBUG(llvm::dbgs()
               << "Validating " << ns << " against " << ref << "\n");
    if (compare(loc, *referenceTransaction, transaction).failed())
      return emitError(loc)
             << "Failure during validation of result from dialect " << ns;
  }
  return success();
}

LogicalResult DialectSimBackplane::call(Operation *sourceOp, StringRef symbol,
                                        SimContextPtr &sourceContext,
                                        ArrayRef<Any> ins,
                                        std::vector<Any> &outs) {
  auto transactorIt = callTransactors.find(sourceOp->getName().getStringRef());
  if (transactorIt == callTransactors.end())
    return sourceOp->emitOpError() << "No call transactor found for this op";

  auto transactor = transactorIt->second;
  auto enterTransaction = transactor->encode(sourceContext, ins);
  assertTransaction(enterTransaction);

  LLVM_DEBUG(llvm::dbgs() << "Looking up symbol: '" << symbol << "'\n");

  if (auto libfunc = lib.get(symbol); libfunc.hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "Found test library definition\n");
    return libfunc.getValue()(enterTransaction);
  }

  /// Look up symbols in the available modules. Use isDefinedSymbol to avoid
  /// pulling in symbol predeclarations.
  SmallVector<Operation *, 4> targetOps;
  for (auto &mod : modules) {
    Operation *op = mod->lookupSymbol(symbol);
    if (op && isDefinedSymbol(op)) {
      LLVM_DEBUG(llvm::dbgs() << "Found definition " << op->getName() << " at "
                              << mod->getLoc() << "\n");
      targetOps.push_back(op);
    }
  }

  if (targetOps.size() > 1 && ref.empty())
    return sourceOp->emitOpError()
           << "Found multiple definitions of symbol '" << symbol
           << "' but reference model was not set";

  llvm::DenseMap<StringRef, Transaction> exitTransactions;

  /// todo(mortbopet): We should be able to execute the following concurrently.
  for (auto targetOp : targetOps) {
    auto res = enter(targetOp, enterTransaction,
                     exitTransactions[targetOp->getDialect()->getNamespace()]);
    if (res.failed())
      return failure();
  }

  /// Validate the results
  if (validate(sourceOp->getLoc(), exitTransactions).failed())
    return failure();

  /// Decode the result; any exit transaction will do, since validation
  /// validated that all transactions were identical.
  transactor->decode(outs, sourceOp, exitTransactions.begin()->second);

  return success();
}

LogicalResult DialectSimBackplane::enter(Operation *op,
                                         Transaction &enterTransaction,
                                         Transaction &exitTransaction) {

  LLVM_DEBUG(llvm::dbgs() << "Entering: " << op->getName() << " at "
                          << op->getLoc() << "\n");

  /// Locate transactor to decode decode the transaction with.
  auto transactorIt = entryTransactors.find(op->getName().getStringRef());
  if (transactorIt == entryTransactors.end())
    return op->emitOpError() << "No entry transactor found for this op";

  auto transactor = transactorIt->second;
  auto simctx = transactor->decode(op, enterTransaction);
  simctx->instrIter = op->getIterator();

  /// Locate simulator to execute the operation with.
  auto sim = simulators.find(transactor.get());
  if (sim == simulators.end())
    return op->emitOpError()
           << "No simulator loaded for entry transactor for op '"
           << op->getName() << "'";

  /// Go execute.
  if (sim->second->execute(simctx).failed()) {
    return failure();
  }

  /// Encode resulting context through the exit transactor.
  /// todo(mortbopet) is this needed?
  exitTransaction = transactor->encode(simctx, simctx->getReturnValues());

  LLVM_DEBUG(llvm::dbgs() << "Exiting: " << op->getName() << " at "
                          << op->getLoc() << "\n");
  return success();
}

LogicalResult DialectSimBackplane::print(Operation *op,
                                         Transaction &exitTransaction) {
  auto transactorIt = entryTransactors.find(op->getName().getStringRef());
  if (transactorIt == entryTransactors.end())
    return op->emitOpError() << "No entry transactor found for this op";

  return transactorIt->second->print(outs(), op, exitTransaction);
}

} // namespace txvf
} // namespace circt
