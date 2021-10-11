//===- HandshakeToAsync.cpp - Translate Handshake into Async --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Handshake to Async Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

// @todo: after all functions have been built, they probably need to be inlined
// into structured control flow... Else we're just going to create a million
// stack frames.

#include "circt/Conversion/HandshakeToAsync.h"
#include "../PassDetail.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/Visitor.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <list>
#include <variant>

using namespace mlir;
using namespace circt;

namespace circt {

using OpMapping = DenseMap<Operation *, FuncOp>;

static std::vector<std::string> SVGNames = {"aliceblue",
                                            "antiquewhite",
                                            "aqua",
                                            "aquamarine",
                                            "azure",
                                            "beige",
                                            "bisque",
                                            "black",
                                            "blanchedalmond",
                                            "blue",
                                            "blueviolet",
                                            "brown",
                                            "burlywood",
                                            "cadetblue",
                                            "chartreuse",
                                            "chocolate",
                                            "coral",
                                            "cornflowerblue",
                                            "cornsilk",
                                            "crimson",
                                            "cyan",
                                            "darkblue",
                                            "darkcyan",
                                            "darkgoldenrod",
                                            "darkgray",
                                            "darkgreen",
                                            "darkgrey",
                                            "darkkhaki",
                                            "darkmagenta",
                                            "darkolivegreen",
                                            "darkorange",
                                            "darkorchid",
                                            "darkred",
                                            "darksalmon",
                                            "darkseagreen",
                                            "darkslateblue",
                                            "darkslategray",
                                            "darkslategrey",
                                            "darkturquoise",
                                            "darkviolet",
                                            "deeppink",
                                            "deepskyblue",
                                            "dimgray",
                                            "dimgrey",
                                            "dodgerblue",
                                            "firebrick",
                                            "floralwhite",
                                            "forestgreen",
                                            "fuchsia",
                                            "gainsboro",
                                            "ghostwhite",
                                            "gold",
                                            "goldenrod",
                                            "gray",
                                            "grey",
                                            "green",
                                            "greenyellow",
                                            "honeydew",
                                            "hotpink",
                                            "indianred",
                                            "indigo",
                                            "ivory",
                                            "khaki",
                                            "lavender",
                                            "lavenderblush",
                                            "lawngreen",
                                            "lemonchiffon",
                                            "lightblue",
                                            "lightcoral",
                                            "lightcyan",
                                            "lightgoldenrodyellow",
                                            "lightgray",
                                            "lightgreen",
                                            "lightgrey",
                                            "lightpink",
                                            "lightsalmon",
                                            "lightseagreen",
                                            "lightskyblue",
                                            "lightslategray",
                                            "lightslategrey",
                                            "lightsteelblue",
                                            "lightyellow",
                                            "lime",
                                            "limegreen",
                                            "linen",
                                            "magenta",
                                            "maroon",
                                            "mediumaquamarine",
                                            "mediumblue",
                                            "mediumorchid",
                                            "mediumpurple",
                                            "mediumseagreen",
                                            "mediumslateblue",
                                            "mediumspringgreen",
                                            "mediumturquoise",
                                            "mediumvioletred",
                                            "midnightblue",
                                            "mintcream",
                                            "mistyrose",
                                            "moccasin",
                                            "navajowhite",
                                            "navy",
                                            "oldlace",
                                            "olive",
                                            "olivedrab",
                                            "orange",
                                            "orangered",
                                            "orchid",
                                            "palegoldenrod",
                                            "palegreen",
                                            "paleturquoise",
                                            "palevioletred",
                                            "papayawhip",
                                            "peachpuff",
                                            "peru",
                                            "pink",
                                            "plum",
                                            "powderblue",
                                            "purple",
                                            "red",
                                            "rosybrown",
                                            "royalblue",
                                            "saddlebrown",
                                            "salmon",
                                            "sandybrown",
                                            "seagreen",
                                            "seashell",
                                            "sienna",
                                            "silver",
                                            "skyblue",
                                            "slateblue",
                                            "slategray",
                                            "slategrey",
                                            "snow",
                                            "springgreen",
                                            "steelblue",
                                            "tan",
                                            "teal",
                                            "thistle",
                                            "tomato",
                                            "turquoise",
                                            "violet",
                                            "wheat",
                                            "white",
                                            "whitesmoke",
                                            "yellow",
                                            "yellowgreen"

};

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

struct AsyncConvRes {
  ValueRange values;
  Value token;
  Operation *yieldOp;
};

static AsyncConvRes syncToAsyncValues(OpBuilder &builder, Location loc,
                                      ValueRange inputs) {
  Operation *yieldOp = nullptr;
  auto executeBodyBuilder = [&](OpBuilder &executeBuilder, Location executeLoc,
                                ValueRange /*executeArgs*/) {
    yieldOp = executeBuilder.create<async::YieldOp>(executeLoc, inputs);
  };

  auto execOp = builder.create<async::ExecuteOp>(
      loc, inputs.getTypes(), ValueRange(), ValueRange(), executeBodyBuilder);

  // Replace arguments with async arguments. The first result of the ExecuteOp
  // is an async.token, so skip it - we only care about the values.
  AsyncConvRes res;
  res.values = execOp.getResults().drop_front(1);
  res.token = execOp.getResult(0);
  res.yieldOp = yieldOp;
  return res;
}

//===----------------------------------------------------------------------===//
// Partial lowering infrastructure
//===----------------------------------------------------------------------===//

/// Base class for partial lowering passes. A partial lowering pass
/// modifies the root operation in place, but does not replace the root
/// operation.
/// The RewritePatternType template parameter allows for using both
/// OpRewritePattern (default) or OpInterfaceRewritePattern.
template <class OpType,
          template <class> class RewritePatternType = OpRewritePattern>
class PartialLoweringPattern : public RewritePatternType<OpType> {
public:
  using RewritePatternType<OpType>::RewritePatternType;
  PartialLoweringPattern(MLIRContext *ctx, LogicalResult &resRef)
      : RewritePatternType<OpType>(ctx), partialPatternRes(resRef) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(
        op, [&] { partialPatternRes = partiallyLower(op, rewriter); });
    return partialPatternRes;
  }

  virtual LogicalResult partiallyLower(OpType op,
                                       PatternRewriter &rewriter) const = 0;

private:
  LogicalResult &partialPatternRes;
};

//===----------------------------------------------------------------------===//
// Partial lowerings
//===----------------------------------------------------------------------===//

template <typename TokenRange>
static void awaitAllTokens(OpBuilder &builder, Location loc,
                           TokenRange tokens) {
  // Await tokens
  if (tokens.size() > 0) {
    if (tokens.size() == 1) {
      builder.create<async::AwaitOp>(loc, tokens[0]);
    } else {
      auto groupOp = builder.create<async::CreateGroupOp>(
          loc, async::TokenType(),
          builder.create<ConstantOp>(
              loc, IntegerAttr::get(builder.getI32Type(), tokens.size())));
      for (auto token : tokens)
        builder.create<async::AddToGroupOp>(loc, async::TokenType(), token,
                                            groupOp);
      builder.create<async::AwaitAllOp>(loc, groupOp.getResult());
    }
  }
}

/// Value must be an async::TokenType
using Tokens = SmallVector<Value>;
static llvm::SmallVector<std::function<Tokens(Tokens)>> outs;

/*

template <typename TOp>
class ConvertHandshakeOp : public OpRewritePattern<TOp> {
public:
  LogicalResult matchAndRewrite(TOp op,
                                PatternRewriter &rewriter) const override {
    return build(rewriter, op);
  }
  void build(PatternRewriter &rewriter, TOp op);
};

template <>
void ConvertHandshakeOp<handshake::ForkOp>::build(PatternRewriter &rewriter,
                                                  handshake::ForkOp op) {
  outs.push_back([&](Tokens tokens) {
    awaitAllTokens(rewriter, op.getLoc(), tokens);
    Tokens outTokens;
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      auto execOp = rewriter.replaceOpWithNewOp<async::ExecuteOp>(op);
      outTokens.push_back(execOp.getResult(0));
    };
    return outTokens;
  });
}

template <>
void ConvertHandshakeOp<handshake::ReturnOp>::build(PatternRewriter &rewriter,
                                                    handshake::ReturnOp op) {
  outs.push_back([&](Tokens tokens) {
    awaitAllTokens(rewriter, op.getLoc(), tokens);
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
    return SmallVector<Value>();
  });
}
*/

class BuildOpFunctionSigs {
public:
  explicit BuildOpFunctionSigs(handshake::FuncOp topLevelFuncOp,
                               OpMapping &opMapping)
      : opMapping(opMapping) {
    OpBuilder builder(topLevelFuncOp.getContext());
    topLevelFuncOp.walk([&](Operation *op) {
      llvm::dbgs() << "Building signature for " << *op << "\n";
      (void)llvm::TypeSwitch<Operation *, LogicalResult>(op)
          .Case<handshake::ForkOp, handshake::ReturnOp, handshake::BranchOp,
                handshake::JoinOp, handshake::ConstantOp, handshake::SinkOp,
                handshake::ConditionalBranchOp, handshake::ControlMergeOp>(
              [&](auto op) -> LogicalResult {
                builder.setInsertionPoint(topLevelFuncOp);
                auto funcOp = buildFunction(builder, op);
                setMapping(op, funcOp);
                return success();
              })
          .Case([&](handshake::FuncOp) {
            return success(); // ignore
          })
          .Default([&](auto op) {
            return op->emitOpError() << "Unknown operation";
          });
    });
  }

  FuncOp buildFunction(OpBuilder &builder, Operation *op);
  SmallVector<Type> getInputTypes(Operation *op);
  SmallVector<Type> getOutputTypes(Operation *op);

private:
  LogicalResult setMapping(Operation *src, FuncOp func);
  OpMapping &opMapping;
};

static std::string unique(StringRef str) {
  static std::map<std::string, unsigned> cntr;
  return (str + "_" + std::to_string(cntr[str.str()]++)).str();
}

LogicalResult BuildOpFunctionSigs::setMapping(Operation *src, FuncOp func) {
  if (opMapping.count(src))
    return src->emitOpError() << "Multiple mappings registered for op";

  opMapping[src] = func;
  return success();
}

SmallVector<Type> BuildOpFunctionSigs::getInputTypes(Operation *op) {
  SmallVector<Type> inputs;
  llvm::TypeSwitch<Operation *, void>(op)
      .Case([&](handshake::ConstantOp) {
        inputs.push_back(async::TokenType::get(op->getContext()));
      })
      .Default([&](auto op) {
        for (auto operand : op->getOperandTypes())
          inputs.push_back(async::ValueType::get(operand));
      });
  return inputs;
}

SmallVector<Type> BuildOpFunctionSigs::getOutputTypes(Operation *op) {
  SmallVector<Type> outputs;
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::ReturnOp>(
          [&](auto op) { llvm::append_range(outputs, op->getOperandTypes()); })
      .Default([&](auto op) {
        for (auto result : op->getResultTypes())
          outputs.push_back(async::ValueType::get(result));
      });
  return outputs;
}

SmallVector<Type> convertNoneTypes(ArrayRef<Type> types) {
  SmallVector<Type> out;
  llvm::transform(types, std::back_inserter(out), [](Type t) -> Type {
    if (t.isa<NoneType>())
      return async::TokenType();
    return t;
  });
  return out;
}

FuncOp BuildOpFunctionSigs::buildFunction(OpBuilder &builder, Operation *op) {
  SmallVector<Type> inputs = getInputTypes(op);
  SmallVector<Type> outputs = getOutputTypes(op);
  auto funcType = builder.getFunctionType(inputs, outputs);
  auto funcOp = builder.create<FuncOp>(
      op->getLoc(), unique(op->getName().getIdentifier()), funcType);
  return funcOp;
}

/// The IR graph is walked based on a ready list of operation. An operation is
/// added to the ready list when any of its dependencies have been built. An
/// operation is built whenever all of its dependencies have been built...
/// maybe?
class BuildOpFunctionBodies {
public:
  explicit BuildOpFunctionBodies(handshake::FuncOp topLevelFuncOp,
                                 OpMapping &opMapping) {
    OpBuilder builder(topLevelFuncOp.getContext());

    std::list<Operation *> readyList;

    for (auto &op : topLevelFuncOp.getOps()) {
      if (op.getNumSuccessors() == 0)
        readyList.push_back(&op);
    }

    topLevelFuncOp.walk([&](Operation *op) {
      (void)llvm::TypeSwitch<Operation *, LogicalResult>(op)
          .Case<handshake::ForkOp, handshake::ReturnOp, handshake::BranchOp,
                handshake::JoinOp, handshake::ConstantOp, handshake::SinkOp,
                handshake::ConditionalBranchOp>([&](auto op) -> LogicalResult {
            auto funcOp = opMapping.find(op);
            assert(funcOp != opMapping.end() &&
                   "No function signature available for op");
            builder.setInsertionPointToStart(funcOp->second.addEntryBlock());
            if (build(builder, funcOp->second, op).failed())
              return op->emitOpError() << "Failure during function conversion";
            return success();
          })
          .Case([&](handshake::FuncOp) {
            return success(); // ignore
          })
          .Default([&](auto op) {
            return op->emitOpError() << "Unknown operation";
          });
    });
  }

  LogicalResult build(OpBuilder &builder, FuncOp funcOp, handshake::ForkOp);
  LogicalResult build(OpBuilder &builder, FuncOp funcOp, handshake::ReturnOp);
  LogicalResult build(OpBuilder &builder, FuncOp funcOp, handshake::BranchOp);
  LogicalResult build(OpBuilder &builder, FuncOp funcOp, handshake::JoinOp);
  LogicalResult build(OpBuilder &builder, FuncOp funcOp, handshake::ConstantOp);
  LogicalResult build(OpBuilder &builder, FuncOp funcOp, handshake::SinkOp);
  LogicalResult build(OpBuilder &builder, FuncOp funcOp,
                      handshake::ConditionalBranchOp);
};

LogicalResult BuildOpFunctionBodies::build(OpBuilder &builder, FuncOp funcOp,
                                           handshake::BranchOp op) {
  builder.create<ReturnOp>(op.getLoc(), funcOp.getArgument(0));
  return success();
}

LogicalResult BuildOpFunctionBodies::build(OpBuilder &builder, FuncOp funcOp,
                                           handshake::ConstantOp op) {
  awaitAllTokens(builder, op.getLoc(), funcOp->getOperands());

  auto constantValue = builder.create<ConstantOp>(op.getLoc(), op.getValue());
  auto asyncConvRes = syncToAsyncValues(builder, op.getLoc(),
                                        ValueRange{constantValue.getResult()});
  builder.create<ReturnOp>(op.getLoc(), asyncConvRes.values);
  return success();
}

LogicalResult BuildOpFunctionBodies::build(OpBuilder &builder, FuncOp funcOp,
                                           handshake::ForkOp op) {
  awaitAllTokens(builder, op.getLoc(), funcOp->getOperands());
  Tokens outTokens;
  for (unsigned i = 0; i < op.getNumResults(); ++i) {
  }
  outTokens.push_back(funcOp.getArgument(0));
  builder.create<ReturnOp>(op.getLoc(), outTokens);
  return success();
}

LogicalResult BuildOpFunctionBodies::build(OpBuilder &builder, FuncOp funcOp,
                                           handshake::JoinOp op) {
  awaitAllTokens(builder, op.getLoc(), funcOp->getOperands());
  builder.create<ReturnOp>(op.getLoc(), funcOp->getOperand(0));
  return success();
}

LogicalResult BuildOpFunctionBodies::build(OpBuilder &builder, FuncOp funcOp,
                                           handshake::SinkOp op) {
  // todo(mortbopet): Sink ops should just be removed, no reason to have them in
  // this execution model.
  awaitAllTokens(builder, op.getLoc(), funcOp->getOperands());
  return success();
}

LogicalResult BuildOpFunctionBodies::build(OpBuilder &builder, FuncOp funcOp,
                                           handshake::ReturnOp op) {
  awaitAllTokens(builder, op.getLoc(), funcOp->getOperands());
  SmallVector<Value> returnArgs;
  Location loc = op.getLoc();

  // await each input operand, which unwraps the values, and return all of the
  // unwrapped values.
  for (auto arg : funcOp.getArguments()) {
    auto awaitOp = builder.create<async::AwaitOp>(loc, arg);
    returnArgs.push_back(awaitOp.getResult(0));
  }
  builder.create<ReturnOp>(loc, returnArgs);
  return success();
}

LogicalResult BuildOpFunctionBodies::build(OpBuilder &builder, FuncOp funcOp,
                                           handshake::ConditionalBranchOp op) {
  auto loc = op.getLoc();
  auto ctrl = builder.create<async::AwaitOp>(loc, funcOp.getArgument(0));
  auto data = funcOp.getArgument(1);

  auto thenBuilder = [&](OpBuilder &builder, Location loc) {
    SmallVector<Value> returnArgs;
    builder.create<ReturnOp>(loc, returnArgs);
  };

  auto ifOp =
      builder.create<scf::IfOp>(loc, ctrl.getResult(0), /*withElse=*/true);

  return success();
}

static void createAsyncArgs(handshake::FuncOp topLevelFuncOp) {
  OpBuilder builder(topLevelFuncOp.getContext());
  builder.setInsertionPointToStart(&topLevelFuncOp.front());
  Location loc = topLevelFuncOp.getLoc();

  auto asyncConvRes =
      syncToAsyncValues(builder, loc, topLevelFuncOp.getArguments());

  // Replace arguments with async arguments. The first result of the ExecuteOp
  // is an async.token, so skip it - we only care about the values.
  for (auto asyncOp : llvm::enumerate(asyncConvRes.values))
    topLevelFuncOp.getArgument(asyncOp.index())
        .replaceAllUsesExcept(asyncOp.value(), asyncConvRes.yieldOp);
}

static LogicalResult replaceOps(handshake::FuncOp topLevelFuncOp,
                                OpMapping &mapping) {
  OpBuilder builder(topLevelFuncOp.getContext());
  for (auto &op : llvm::make_early_inc_range(topLevelFuncOp.front())) {
    if (&op == topLevelFuncOp)
      continue;
    if (op.getDialect()->getNamespace() == "async")
      continue;

    auto callee = mapping.find(&op);
    if (callee == mapping.end())
      return op.emitOpError() << "No mapping found for op!";
    builder.setInsertionPoint(&op);
    auto callOp =
        builder.create<CallOp>(op.getLoc(), callee->second, op.getOperands());

    llvm::TypeSwitch<Operation *, void>(&op)
        .Case([&](handshake::ReturnOp op) {
          builder.create<ReturnOp>(op.getLoc(), callOp.getResults());
        })
        .Default([&](auto op) { op->replaceAllUsesWith(callOp); });
    op.erase();
  };
  return success();
}

class FuncOpLowering : public OpRewritePattern<handshake::FuncOp> {
  using OpRewritePattern<handshake::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::FuncOp handshakeFuncOp,
                                PatternRewriter &rewriter) const override {
    // Create a new func op

    rewriter.setInsertionPoint(handshakeFuncOp);
    auto funcOp = rewriter.create<FuncOp>(
        handshakeFuncOp.getLoc(), (handshakeFuncOp.getName() + "_async").str(),
        handshakeFuncOp.getType());
    auto entryBlock = funcOp.addEntryBlock();
    rewriter.mergeBlocks(&handshakeFuncOp.getRegion().front(), entryBlock,
                         entryBlock->getArguments());
    rewriter.eraseOp(handshakeFuncOp);
    return success();
  }
};

} // namespace circt

static bool hasGroupAttr(Operation *op) { return op->hasAttr("group"); }

static unsigned getGroup(Operation *op) {
  auto group = op->getAttrOfType<IntegerAttr>("group");
  return group.getValue().getZExtValue();
}

template <typename FuncOp>
void dotPrint(FuncOp f, StringRef name) {
  // Prints DOT representation of the dataflow graph, used for debugging.
  DenseMap<Block *, unsigned> blockIDs;
  DenseMap<Operation *, unsigned> opIDs;
  unsigned i = 0;
  unsigned j = 0;

  for (Block &block : f) {
    blockIDs[&block] = i++;
    for (Operation &op : block)
      opIDs[&op] = j++;
  }

  std::error_code ec;
  llvm::raw_fd_ostream outfile(name.str() + ".dot", ec);

  outfile << "Digraph G {\n\tsplines=spline;\n";

  for (Block &block : f) {
    outfile << "\tsubgraph cluster_" + std::to_string(blockIDs[&block]) +
                   " {\n";
    outfile << "\tcolor = \"darkgreen\";\n";
    outfile << "\t\tlabel = \" block " + std::to_string(blockIDs[&block]) +
                   "\";\n";

    std::map<unsigned, SmallVector<Operation *>> groups;
    SmallVector<Operation *> ungrouped;
    for (auto &op : block) {
      if (hasGroupAttr(&op))
        groups[getGroup(&op)].push_back(&op);
    }

    auto emitOp = [&](Operation &op) {
      outfile << "\t\t";
      outfile << "\"" + op.getName().getStringRef().str() + "_" +
                     std::to_string(opIDs[&op]) + "\"";
      outfile << " [";
      llvm::SmallVector<std::string> style;
      if (op.hasAttr("group")) {
        auto group = op.getAttrOfType<IntegerAttr>("group");
        outfile << "fillcolor=" << SVGNames[group.getValue().getZExtValue()]
                << ",";
        style.push_back("filled");
      }
      if (op.hasAttr("control")) {
        if (op.getAttrOfType<BoolAttr>("control").getValue())
          style.push_back("dotted");
      }

      if (!style.empty()) {
        outfile << "style=\"";
        llvm::interleaveComma(style, outfile);
        outfile << "\"";
      }

      outfile << "]\n";
      outfile << "\n";
    };

    for (auto group : groups) {
      outfile << "\t";
      outfile << "subgraph cluster_" << std::to_string(group.first) << "{\n";
      outfile << "\t\tlabel=\"group_" << std::to_string(group.first) << "\";\n";
      for (auto groupOp : group.second)
        emitOp(*groupOp);
      outfile << "}\n";
    }
    for (auto ungroupedOp : ungrouped)
      emitOp(*ungroupedOp);

    for (Operation &op : block) {
      if (op.getNumResults() == 0)
        continue;

      for (auto result : op.getResults()) {
        for (auto &u : result.getUses()) {
          Operation *useOp = u.getOwner();
          if (useOp->getBlock() == &block) {
            outfile << "\t\t";
            outfile << "\"" + op.getName().getStringRef().str() + "_" +
                           std::to_string(opIDs[&op]) + "\"";
            outfile << " -> ";
            outfile << "\"" + useOp->getName().getStringRef().str() + "_" +
                           std::to_string(opIDs[useOp]) + "\"";
            outfile << "\n";
          }
        }
      }
    }

    outfile << "\t}\n";

    for (Operation &op : block) {
      if (op.getNumResults() == 0)
        continue;

      for (auto result : op.getResults()) {
        for (auto &u : result.getUses()) {
          Operation *useOp = u.getOwner();
          if (useOp->getBlock() != &block) {
            outfile << "\t\t";
            outfile << "\"" + op.getName().getStringRef().str() + "_" +
                           std::to_string(opIDs[&op]) + "\"";
            outfile << " -> ";
            outfile << "\"" + useOp->getName().getStringRef().str() + "_" +
                           std::to_string(opIDs[useOp]) + "\"";
            outfile << " [minlen = 3]\n";
          }
        }
      }
    }
  }

  outfile << "\t}\n";
  outfile.close();
}

struct WorkListDep {
  WorkListDep() {}
  WorkListDep(ArrayRef<Operation *> _anyDep, ArrayRef<Operation *> _allDep) {
    llvm::append_range(anyDep, _anyDep);
    llvm::append_range(allDep, _allDep);
  }
  llvm::SmallVector<Operation *> anyDep;
  llvm::SmallVector<Operation *> allDep;
};

struct WorkList {
  void add(Operation *op, WorkListDep dep) {
    assert(!hasGroupAttr(op) &&
           "Adding operation which was already analyzed to worklist?");
    list[op] = dep;
  }
  std::map<Operation *, WorkListDep> list;
};

static bool canAnnotate(Operation *op, WorkList &worklist) {
  if (hasGroupAttr(op))
    return false;

  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<handshake::ControlMergeOp>([&](auto op) {
        // Always assign new Id to control merge due to them
        // being landing pads.
        return true;
      })
      .Case<handshake::MuxOp>([&](handshake::MuxOp op) {
        // Depend on the control-value generating op
        auto selectOp = op.selectOperand().getDefiningOp();
        if (hasGroupAttr(selectOp))
          return true;
        else {
          worklist.add(op,
                       WorkListDep({}, {op.selectOperand().getDefiningOp()}));
          return false;
        }
      })
      .Default([&](auto op) {
        llvm::SmallVector<Operation *> dependencies;
        for (auto operand : op->getOperands()) {
          if (operand.template dyn_cast<BlockArgument>())
            continue;

          // Depend on all
          if (!hasGroupAttr(operand.getDefiningOp()))
            dependencies.push_back(operand.getDefiningOp());
        }

        if (dependencies.size() != 0) {
          // Add to worklist
          worklist.add(op, WorkListDep({}, dependencies));
          return false;
        }
        return true;
      });
}

static void annotateGroupsRec(Operation *op, WorkList &worklist, int id,
                              int &cntr) {
  assert(!hasGroupAttr(op) && "Annotating op which is already annotated!");

  // Sourcing; these ops will always start a new group
  if (isa<handshake::ControlMergeOp>(op)) {
    id = ++cntr;
  }

  llvm::dbgs() << "Annotating " << *op << " with " << id << "\n";
  op->setAttr("group", IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                        APInt(32, id)));
  worklist.list.erase(op);

  // Sinking
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::ForkOp>([&](auto op) {
        for (auto succ : op.getResults())
          for (auto user : succ.getUsers())
            if (canAnnotate(user, worklist)) {
              cntr++;
              annotateGroupsRec(user, worklist, cntr, cntr);
            }
      })
      .Default([&](auto op) {
        for (auto succ : op->getResults())
          for (auto user : succ.getUsers())
            if (canAnnotate(user, worklist))
              annotateGroupsRec(user, worklist, id, cntr);
      });
}

static void annotateGroups(handshake::FuncOp funcOp) {
  std::list<Operation *> readyList;
  WorkList worklist;

  int cntr = 0;
  for (auto arg : funcOp.getArguments()) {
    for (auto user : arg.getUsers()) {
      if (user->getNumOperands() == 1) {
        worklist.add(user, {});
      }
    }
  }

  bool continueAnalysis = true;
  while (continueAnalysis) {
    // Locate an operation ready to be analyzed.
    continueAnalysis = false;
    for (auto it : worklist.list) {
      bool anyOk = it.second.anyDep.empty() ||
                   llvm::any_of(it.second.anyDep, hasGroupAttr);
      bool allOk = it.second.allDep.empty() ||
                   llvm::all_of(it.second.allDep, hasGroupAttr);
      if (anyOk && allOk) {
        annotateGroupsRec(it.first, worklist, cntr, cntr);
        continueAnalysis = true;
        break;
      }
    }
  }
}

namespace {
class HandshakeToAsyncPass : public HandshakeToAsyncBase<HandshakeToAsyncPass> {
public:
  void runOnOperation() override {
    getOperation().walk([&](handshake::FuncOp funcOp) {
      annotateGroups(funcOp);
      dotPrint(funcOp, funcOp.getName().str());

      OpMapping opMapping;

      { BuildOpFunctionSigs pass(funcOp, opMapping); }

      // Create function signatures
      { BuildOpFunctionBodies pass(funcOp, opMapping); }

      // // Create futures of the input arguments
      // createAsyncArgs(funcOp);
      //
      // // Replace calls
      // replaceOps(funcOp, opMapping);
    });

    // ...
    ConversionTarget target(getContext());
    target.addLegalDialect<async::AsyncDialect>();
    target.addIllegalDialect<handshake::HandshakeDialect>();
    RewritePatternSet patterns(getOperation().getContext());
    patterns.insert<FuncOpLowering>(getOperation().getContext());
    for (auto funcOp : llvm::make_early_inc_range(
             getOperation().getOps<handshake::FuncOp>())) {
      if (failed(applyOpPatternsAndFold(funcOp, std::move(patterns))))
        signalPassFailure();
    }

    getOperation().dump();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::createHandshakeToAsyncPass() {
  return std::make_unique<HandshakeToAsyncPass>();
}
