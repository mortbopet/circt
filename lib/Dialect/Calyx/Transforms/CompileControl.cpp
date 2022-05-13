//===- CompileControlPass..cpp - Compile Control Pass -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Compile Control pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace calyx;
using namespace mlir;

struct CalyxNamespace : public Namespace {
  CalyxNamespace() {}
  CalyxNamespace(ComponentOp component) { add(component); }

  void add(ComponentOp component) {
    for (SymbolOpInterface op :
         component.getBody()->getOps<SymbolOpInterface>())
      nextIndex.insert({op.getName(), 0});
  }
};

/// Given some number of states, returns the necessary bit width
/// TODO(Calyx): Probably a better built-in operation?
static size_t getNecessaryBitWidth(size_t numStates) {
  APInt apNumStates(64, numStates);
  size_t log2 = apNumStates.ceilLogBase2();
  return log2 > 1 ? log2 : 1;
}

// Ensures that a structural control operation is predicated on either some
// registered value or an external input, and not a combinational value + 'with'
// a combinational group.
template <typename TOp>
static LogicalResult ensureValidStructuralGuard(TOp op) {
  if (op.groupName().hasValue())
    return op.emitOpError() << "expected 'with' groups to be removed before "
                               "this pass (--remove-comb-groups)";

  if (!op.cond().template isa<BlockArgument>() &&
      !isa<RegisterOp>(op.cond().getDefiningOp()))
    return op.emitOpError()
           << "expected structural guard to be a register output";

  return success();
}

struct CCVisitor {
  CCVisitor(CalyxNamespace &ns, ComponentOp component)
      : component(component), ns(ns), ctx(component.getContext()),
        builder(component.getContext()) {
    // Build the TDCC group where all FSM operations will be emitted.
    builder.setInsertionPointToStart(component.getWiresOp().getBody());
    tdccGroup = builder.create<GroupOp>(component.getLoc(), "fsm");
    builder.setInsertionPointToStart(component.getWiresOp().getBody());
  }

  LogicalResult go() {
    auto &topLevelControlOp = component.getControlOp().getBody()->front();
    auto res = dispatch(&topLevelControlOp, component.getGoPort());
    if (failed(res))
      return failure();

    // Connect top-level FSM finished signal to TDCCs group done signal.
    builder.setInsertionPointToEnd(tdccGroup.getBody());
    builder.create<GroupDoneOp>(
        topLevelControlOp.getLoc(),
        createConstant(tdccGroup.getLoc(), builder, component, 1, 1),
        res.getValue());
    topLevelControlOp.erase();
    return success();
  }

  FailureOr<Value> dispatch(Operation *op, Value inGo) {
    return TypeSwitch<Operation *, FailureOr<Value>>(op)
        .template Case<SeqOp, EnableOp, WhileOp, IfOp, ParOp>(
            [&](auto opNode) { return visit(opNode, inGo); })
        .Default([&](auto) {
          return op->emitError() << "Operation '" << op->getName()
                                 << "' not supported for control compilation";
        });
  }

  /// Each 'visit' statement compiles a separate distinct FSM, based on the type
  /// of the control reached. A 'visit' function has the signature:
  ///   FSMDoneValue visit(T op, inGoValue)
  ///
  /// - FSMDoneValue: A value that is asserted whenever the given FSM has
  ///   finished.
  /// - inGoValue: A value that is asserted whenever the sub-FSM is to be
  ///   active.
  FailureOr<Value> visit(SeqOp seqOp, Value inGo);
  FailureOr<Value> visit(ParOp parOp, Value inGo);
  FailureOr<Value> visit(EnableOp enableOp, Value inGo);
  FailureOr<Value> visit(IfOp ifOp, Value inGo);
  FailureOr<Value> visit(WhileOp whileOp, Value inGo);

  ComponentOp component;
  CalyxNamespace &ns;
  MLIRContext *ctx;
  OpBuilder builder;
  GroupOp tdccGroup;
  SmallVector<Attribute, 8> compiledGroups;
};

/// Writes the 'go' signal of the given group, and returns signalling for
/// indicating that this group is done.
FailureOr<Value> CCVisitor::visit(EnableOp enOp, Value inGo) {
  auto loc = enOp.getLoc();
  GroupOp enabledGroup =
      component.getWiresOp().lookupSymbol<GroupOp>(enOp.groupName());
  assert(enabledGroup);
  auto oneConstant = createConstant(loc, builder, component, 1, 1);

  // Directly update the GroupGoOp of the current group being walked.
  // We here OR the inGo op with the current value of the go op source (since a
  // group may be activated from multiple places in the control nshedule).
  auto goOp = enabledGroup.getGoOp();
  if (!goOp)
    return enabledGroup.emitOpError()
           << "The Go Insertion pass should be run before control compilation.";
  Value currentGoGuard = goOp.guard();
  Value goValue = inGo;
  if (currentGoGuard && !currentGoGuard.isa<BlockArgument>() &&
      !isa<UndefinedOp>(currentGoGuard.getDefiningOp())) {
    auto goGuardOrOp = builder.create<comb::OrOp>(
        loc, SmallVector<Value>({currentGoGuard, inGo}));
    goValue = goGuardOrOp;
  }
  goOp->setOperands({oneConstant, goValue});

  // Get the group done condition.
  Value guard = enabledGroup.getDoneOp().guard();
  Value source = enabledGroup.getDoneOp().src();
  Value doneOpValue =
      !guard ? source : builder.create<comb::AndOp>(loc, guard, source);

  // Append this group to the set of compiled groups.
  compiledGroups.push_back(
      SymbolRefAttr::get(builder.getContext(), enOp.groupName()));

  return doneOpValue;
}

FailureOr<Value> CCVisitor::visit(WhileOp whileOp, Value inGo) {
  if (failed(ensureValidStructuralGuard(whileOp)))
    return failure();

  auto loc = whileOp.getLoc();

  // Loop continuation carries a strong assumption on the whileOp.cond() value
  // being a registered output and active
  // 1: upon loop entry
  // 2: only written in the loop latch (final loop group).
  Value continueLoop = builder.create<comb::AndOp>(
      loc, builder.getI1Type(), SmallVector<Value>({whileOp.cond(), inGo}));

  // Recurse into handling the loop body FSM. Dinsard the return value, since we
  // only care about loop continuation.
  auto res = dispatch(&whileOp.getBody()->front(), continueLoop);
  if (failed(res))
    return res;

  // Continue after the while loop once the loop is not taken.
  return comb::createOrFoldNot(loc, continueLoop, builder);
}

FailureOr<Value> CCVisitor::visit(IfOp ifOp, Value inGo) {
  if (failed(ensureValidStructuralGuard(ifOp)))
    return failure();
  BackedgeBuilder bb(builder, component.getLoc());

  auto loc = ifOp.getLoc();

  auto lowerBranch = [&](Block *branchBlock,
                         Value branchTaken) -> FailureOr<Value> {
    auto branchDone = bb.get(builder.getI1Type());
    auto branchNotDone = comb::createOrFoldNot(loc, branchDone, builder);

    // Trigger then branch upon incoming control & branch
    // not done & branch taken.
    auto branchGoGuard = builder.create<comb::AndOp>(
        loc, builder.getI1Type(),
        SmallVector<Value>({inGo, branchTaken, branchNotDone}));

    // Recurse into handling the branch sub-FSM
    assert(branchBlock->getOperations().size() == 1 &&
           "Expected a single operation inside the branch block");
    auto res = dispatch(&branchBlock->front(), branchGoGuard);
    if (failed(res))
      return res;
    branchDone.setValue(res.getValue());
    return res.getValue();
  };

  // Then branch
  auto thenRes = lowerBranch(ifOp.getThenBody(), ifOp.cond());
  if (failed(thenRes))
    return failure();

  // Else branch
  FailureOr<Value> elseRes;
  if (ifOp.elseBodyExists()) {
    elseRes = lowerBranch(ifOp.getElseBody(),
                          comb::createOrFoldNot(loc, ifOp.cond(), builder));
    if (failed(elseRes))
      return failure();
  }

  // The IfOp FSM is finished when either branch reports done (mutually
  // exclusive).
  auto ifOpDone = thenRes.getValue();
  if (ifOp.elseBodyExists())
    ifOpDone = builder.create<comb::OrOp>(loc, ifOpDone, elseRes.getValue());
  return ifOpDone;
}

FailureOr<Value> CCVisitor::visit(ParOp parOp, Value inGo) {
  Location loc = parOp.getLoc();
  BackedgeBuilder bb(builder, component.getLoc());
  auto oneConstant = createConstant(loc, builder, component, 1, 1);

  // Registers to save the done signal from each child.
  SmallVector<Value> doneRegOuts;

  // For each child, build the enabling logic.
  for (auto &child : *parOp.getBody()) {
    // Save the done condition in a register.
    RegisterOp doneReg = createRegister(loc, builder, component, 1,
                                        ns.newName("par_done", "reg"));
    doneRegOuts.push_back(doneReg.out());

    // Build the Guard for the enablement of the sub-FSM.
    // The group should begin when:
    // (1) inGo is high
    // (2) the done signal of this group is not high.
    auto currentOpDone = bb.get(builder.getI1Type());
    auto notDone = comb::createOrFoldNot(loc, currentOpDone, builder);
    auto groupGoGuard = builder.create<comb::AndOp>(loc, inGo, notDone);
    // Recurse into handling the current sub-FSM
    auto dispatchRes = dispatch(&child, groupGoGuard);
    if (failed(dispatchRes)) {
      // Set the value of the backedge before returning the error; this avoids
      // an assert in the BackedgeBuilder upon destruction due to not having
      // set the backedge.
      currentOpDone.setValue(oneConstant);
      return failure();
    }
    Value subFSMDone = dispatchRes.getValue();
    currentOpDone.setValue(subFSMDone);

    // Add guarded assignments to the done register `in` and `write_en` ports.
    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(tdccGroup.getBody());
      builder.create<AssignOp>(loc, doneReg.in(), oneConstant, subFSMDone);
      builder.create<AssignOp>(loc, doneReg.write_en(), oneConstant,
                               subFSMDone);
    }
  };

  // This Par is finished when all children are finished
  return builder.create<comb::AndOp>(loc, builder.getI1Type(), doneRegOuts)
      .getResult();
}

FailureOr<Value> CCVisitor::visit(SeqOp seq, Value inGo) {
  auto loc = seq.getLoc();
  auto oneConstant = createConstant(loc, builder, component, 1, 1);
  BackedgeBuilder bb(builder, component.getLoc());

  // Enumerate the number of states required for this FSM.
  auto wires = component.getWiresOp();
  auto &seqOps = seq.getBody()->getOperations();

  unsigned nSeqOps = seqOps.size();
  if (nSeqOps == 1) {
    // No state, simply a nested activation inside a Seq. Simply dispatch the
    // internal group and return its done value.
    return dispatch(&seqOps.front(), inGo);
  } else {
    // This should be the number of enable statements + 1 since this is the
    // maximum value the FSM register will reach.
    unsigned nStates = seqOps.size() + 1;
    size_t fsmBitWidth = getNecessaryBitWidth(nStates);
    auto fsmName = ns.newName("fsm", "reg");

    // Build FSM register
    auto fsmRegister =
        createRegister(seq.getLoc(), builder, component, fsmBitWidth, fsmName);
    Value fsmIn = fsmRegister.in();
    Value fsmWriteEn = fsmRegister.write_en();
    Value fsmOut = fsmRegister.out();
    Value fsmNextState;

    // Iterate over each sub op in this Seq FSM
    for (auto &it : llvm::enumerate(seqOps)) {
      size_t fsmIdx = it.index();
      Operation &currentOp = it.value();

      auto fsmCurrentState = createConstant(wires->getLoc(), builder, component,
                                            fsmBitWidth, fsmIdx);

      // Build the Guard for the enablement of the sub-FSM.
      // The group should begin when:
      // (1) the current step in the fsm is reached, and
      // (2) the done signal of this group is not high.
      auto currentOpDone = bb.get(builder.getI1Type());
      auto eqCmp = builder.create<comb::ICmpOp>(
          wires->getLoc(), comb::ICmpPredicate::eq, fsmOut, fsmCurrentState);
      auto notDone =
          comb::createOrFoldNot(wires->getLoc(), currentOpDone, builder);
      auto groupGoGuard =
          builder.create<comb::AndOp>(wires->getLoc(), eqCmp, notDone);

      // Recurse into handling the current sub-FSM
      auto dispatchRes = dispatch(&currentOp, groupGoGuard);
      if (failed(dispatchRes)) {
        // Set the value of the backedge before returning the error; this avoids
        // an assert in the BackedgeBuilder upon destruction due to not having
        // set the backedge.
        currentOpDone.setValue(oneConstant);
        return failure();
      }
      Value subFSMDone = dispatchRes.getValue();
      currentOpDone.setValue(subFSMDone);

      // Guard for the `in` and `write_en` signal of the fsm register. These are
      // driven when the group has completed.
      auto groupDoneGuard =
          builder.create<comb::AndOp>(wires->getLoc(), eqCmp, subFSMDone);

      // Add guarded assignments to the fsm register `in` and `write_en` ports.
      fsmNextState = createConstant(wires->getLoc(), builder, component,
                                    fsmBitWidth, fsmIdx + 1);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(tdccGroup.getBody());
        builder.create<AssignOp>(wires->getLoc(), fsmIn, fsmNextState,
                                 groupDoneGuard);
        builder.create<AssignOp>(wires->getLoc(), fsmWriteEn, oneConstant,
                                 groupDoneGuard);
      }
    }

    // Build the final guard for the GroupDoneOp. This is
    // defined by the fsm's final state.
    auto isFinalState = builder.create<comb::ICmpOp>(
        wires->getLoc(), comb::ICmpPredicate::eq, fsmOut, fsmNextState);

    // Add continuous wires to reset the `in` and `write_en` ports of the fsm
    // when the SeqGroup is finished executing.
    auto zeroConstant = createConstant(loc, builder, component, fsmBitWidth, 0);
    {
      OpBuilder::InsertionGuard g(builder);
      builder.create<AssignOp>(wires->getLoc(), fsmIn, zeroConstant,
                               isFinalState);
      builder.create<AssignOp>(wires->getLoc(), fsmWriteEn, oneConstant,
                               isFinalState);
    }

    return isFinalState.result();
  }
}

namespace {

struct CompileControlPass : public CompileControlBase<CompileControlPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void CompileControlPass::runOnOperation() {
  ComponentOp component = getOperation();

  if (component.getControlOp().getBody()->empty())
    return; // Nothing to do

  // This pass places strong assumptions on all comb logic has been hoisted to
  // wire nsope.
  // TODO: run GICM

  CalyxNamespace ns;
  ns.add(component);
  CCVisitor visitor(ns, component);
  if (failed(visitor.go()))
    signalPassFailure();

  // Replace the control nshedule with a single instance of the TDCC group.
  OpBuilder builder(&getContext());
  builder.setInsertionPointToStart(component.getControlOp().getBody());
  builder.create<EnableOp>(
      component.getLoc(), visitor.tdccGroup.getName(),
      ArrayAttr::get(builder.getContext(), visitor.compiledGroups));
}

std::unique_ptr<mlir::Pass> circt::calyx::createCompileControlPass() {
  return std::make_unique<CompileControlPass>();
}
