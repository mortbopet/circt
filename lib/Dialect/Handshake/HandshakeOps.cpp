//===- HandshakeOps.cpp - Handshake MLIR Operations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the SlotMapping struct.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace circt::handshake;

#define INDEX_WIDTH 32

namespace circt {
namespace handshake {
#include "circt/Dialect/Handshake/HandshakeCanonicalization.h.inc"
}
} // namespace circt

// Convert ValueRange to vectors
std::vector<mlir::Value> toVector(mlir::ValueRange range) {
  return std::vector<mlir::Value>(range.begin(), range.end());
}

// Returns whether the precondition holds for a general op to execute
bool isReadyToExecute(ArrayRef<mlir::Value> ins, ArrayRef<mlir::Value> outs,
                      llvm::DenseMap<mlir::Value, llvm::Any> &valueMap) {
  for (auto in : ins)
    if (valueMap.count(in) == 0)
      return false;

  for (auto out : outs)
    if (valueMap.count(out) > 0)
      return false;

  return true;
}

static std::string defaultOperandName(unsigned int idx) {
  return "in" + std::to_string(idx);
}

// Fetch values from the value map and consume them
std::vector<llvm::Any>
fetchValues(ArrayRef<mlir::Value> values,
            llvm::DenseMap<mlir::Value, llvm::Any> &valueMap) {
  std::vector<llvm::Any> ins;
  for (auto &value : values) {
    assert(valueMap[value].hasValue());
    ins.push_back(valueMap[value]);
    valueMap.erase(value);
  }
  return ins;
}

// Store values to the value map
void storeValues(std::vector<llvm::Any> &values, ArrayRef<mlir::Value> outs,
                 llvm::DenseMap<mlir::Value, llvm::Any> &valueMap) {
  assert(values.size() == outs.size());
  for (unsigned long i = 0; i < outs.size(); ++i)
    valueMap[outs[i]] = values[i];
}

// Update the time map after the execution
void updateTime(ArrayRef<mlir::Value> ins, ArrayRef<mlir::Value> outs,
                llvm::DenseMap<mlir::Value, double> &timeMap, double latency) {
  double time = 0;
  for (auto &in : ins)
    time = std::max(time, timeMap[in]);
  time += latency;
  for (auto &out : outs)
    timeMap[out] = time;
}

bool tryToExecute(Operation *op,
                  llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                  llvm::DenseMap<mlir::Value, double> &timeMap,
                  std::vector<mlir::Value> &scheduleList, double latency) {
  auto ins = toVector(op->getOperands());
  auto outs = toVector(op->getResults());

  if (isReadyToExecute(ins, outs, valueMap)) {
    auto in = fetchValues(ins, valueMap);
    std::vector<llvm::Any> out(outs.size());
    auto generalOp = dyn_cast<handshake::GeneralOpInterface>(op);
    if (!generalOp)
      op->emitError("Undefined execution for the current op");
    generalOp.execute(in, out);
    storeValues(out, outs, valueMap);
    updateTime(ins, outs, timeMap, latency);
    scheduleList = outs;
    return true;
  } else
    return false;
}

void ForkOp::build(OpBuilder &builder, OperationState &result, Value operand,
                   int outputs) {

  auto type = operand.getType();

  // Fork has results as many as there are successor ops
  result.types.append(outputs, type);

  // Single operand
  result.addOperands(operand);

  // Fork is control-only if it has NoneType. This includes the no-data output
  // of a ControlMerge or a StartOp, as well as control values from MemoryOps.
  bool isControl = operand.getType().isa<NoneType>() ? true : false;
  result.addAttribute("control", builder.getBoolAttr(isControl));
}

static ::mlir::ParseResult parseForkOp(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> resultTypes;
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseLSquare())
    return ::mlir::failure();
  int size;
  if (parser.parseInteger(size))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(operandRawTypes[0]))
    return ::mlir::failure();
  // Add types for the variadic result  
  resultTypes.assign(size, operandRawTypes[0]);
  result.addTypes(resultTypes);
  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

static void printForkOp(::mlir::OpAsmPrinter &p, ForkOp op) {
  p << '[' << op.result().size() << ']';
  p << ' ';
  p << op.getOperation()->getOperands();
  p.printOptionalAttrDict((op)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(op.operand().getType());
}

void handshake::ForkOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.insert<circt::handshake::EliminateSimpleForksPattern>(context);
}

void handshake::ForkOp::execute(std::vector<llvm::Any> &ins,
                                std::vector<llvm::Any> &outs) {
  for (auto &out : outs)
    out = ins[0];
}

bool handshake::ForkOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 1);
}

void LazyForkOp::build(OpBuilder &builder, OperationState &result,
                       Value operand, int outputs) {

  auto type = operand.getType();

  // Fork has results as many as there are successor ops
  result.types.append(outputs, type);

  // Single operand
  result.addOperands(operand);

  // Fork is control-only if it is the no-data output of a ControlMerge or a
  // StartOp
  auto *op = operand.getDefiningOp();
  bool isControl = ((dyn_cast<ControlMergeOp>(op) || dyn_cast<StartOp>(op)) &&
                    operand == op->getResult(0))
                       ? true
                       : false;
  result.addAttribute("control", builder.getBoolAttr(isControl));
}

static ::mlir::ParseResult parseLazyForkOp(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return parseForkOp(parser, result);
 }

static void printLazyForkOp(::mlir::OpAsmPrinter &p, LazyForkOp op) {
  p << '[' << op.result().size() << ']';
  p << ' ';
  p << op.getOperation()->getOperands();
  p.printOptionalAttrDict((op)->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(op.operand().getType());
}

void MergeOp::build(OpBuilder &builder, OperationState &result, Value operand,
                    int inputs) {

  auto type = operand.getType();
  result.types.push_back(type);

  // Operand to keep defining value (used when connecting merges)
  // Removed afterwards
  result.addOperands(operand);

  // Operands from predecessor blocks
  for (int i = 0, e = inputs; i < e; ++i)
    result.addOperands(operand);
}

static ::mlir::ParseResult parseMergeOp(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> dataOperandsTypes;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseLSquare())
    return ::mlir::failure();
  int size;
  if (parser.parseInteger(size))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  // Add types for the variadic operands
  dataOperandsTypes.assign(size, resultRawTypes[0]);
  result.addTypes(resultTypes);
  if (parser.resolveOperands(allOperands, dataOperandsTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void printMergeOp(::mlir::OpAsmPrinter &p, MergeOp op) {
  p << '[' << op.dataOperands().size() << ']';
  p << ' ';
  p << op.getOperation()->getOperands();
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(op.result().getType());
}

void MergeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<circt::handshake::EliminateSimpleMergesPattern>(context);
}

bool handshake::MergeOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  auto op = getOperation();
  bool found = false;
  int i = 0;
  for (mlir::Value in : op->getOperands()) {
    if (valueMap.count(in) == 1) {
      if (found)
        op->emitError("More than one valid input to Merge!");
      auto t = valueMap[in];
      valueMap[op->getResult(0)] = t;
      timeMap[op->getResult(0)] = timeMap[in];
      // Consume the inputs.
      valueMap.erase(in);
      found = true;
    }
    i++;
  }
  if (!found)
    op->emitError("No valid input to Merge!");
  scheduleList.push_back(getResult());
  return true;
}

void MuxOp::build(OpBuilder &builder, OperationState &result, Value operand,
                  int inputs) {

  auto type = operand.getType();
  result.types.push_back(type);

  // Operand connected to ControlMerge from same block
  result.addOperands(operand);

  // Operands from predecessor blocks
  for (int i = 0, e = inputs; i < e; ++i)
    result.addOperands(operand);
}

std::string handshake::MuxOp::getOperandName(unsigned int idx) {
  return idx == 0 ? "select" : defaultOperandName(idx - 1);
}

static ::mlir::ParseResult parseMuxOp(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> dataOperandsTypes;
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseLSquare())
    return ::mlir::failure();
  int size;
  if (parser.parseInteger(size))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  // Add types for the variadic operands
  dataOperandsTypes.assign(size, resultRawTypes[0]);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(allOperands, ::llvm::concat<const Type>(::llvm::ArrayRef<::mlir::Type>(odsBuildableType0), ::llvm::ArrayRef<::mlir::Type>(dataOperandsTypes)), allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

static void printMuxOp(::mlir::OpAsmPrinter &p, MuxOp op) {
  p << '[' << op.dataOperands().size() << ']';
  p << ' ';
  p << op.getOperation()->getOperands();
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(op.result().getType());
}

bool handshake::MuxOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  auto op = getOperation();
  mlir::Value control = op->getOperand(0);
  if (valueMap.count(control) == 0)
    return false;
  auto controlValue = valueMap[control];
  auto controlTime = timeMap[control];
  mlir::Value in = llvm::any_cast<APInt>(controlValue) == 0 ? op->getOperand(1)
                                                            : op->getOperand(2);
  if (valueMap.count(in) == 0)
    return false;
  auto inValue = valueMap[in];
  auto inTime = timeMap[in];
  double time = std::max(controlTime, inTime);
  valueMap[op->getResult(0)] = inValue;
  timeMap[op->getResult(0)] = time;

  // Consume the inputs.
  valueMap.erase(control);
  valueMap.erase(in);
  scheduleList.push_back(getResult());
  return true;
}

static LogicalResult verify(MuxOp op) {
  unsigned numDataOperands = static_cast<int>(op.dataOperands().size());
  if (numDataOperands < 2)
    return op.emitError("need at least two inputs to mux");

  auto selectType = op.selectOperand().getType();

  unsigned selectBits;
  if (auto integerType = selectType.dyn_cast<IntegerType>())
    selectBits = integerType.getWidth();
  else if (selectType.isIndex())
    selectBits = IndexType::kInternalStorageBitWidth;
  else
    return op.emitError("unsupported type for select operand: ") << selectType;

  double maxDataOperands = std::pow(2, selectBits);
  if (numDataOperands > maxDataOperands)
    return op.emitError("select bitwidth was ")
           << selectBits << ", which can mux "
           << static_cast<int64_t>(maxDataOperands) << " operands, but found "
           << numDataOperands << " operands";

  return success();
}

std::string handshake::ControlMergeOp::getResultName(unsigned int idx) {
  assert(idx == 0 || idx == 1);
  return idx == 0 ? "dataOut" : "index";
}

void ControlMergeOp::build(OpBuilder &builder, OperationState &result,
                           Value operand, int inputs) {

  auto type = operand.getType();
  result.types.push_back(type);
  // Second result gives the input index to the muxes
  // Number of bits depends on encoding (log2/1-hot)
  result.types.push_back(builder.getIndexType());

  // Operand to keep defining value (used when connecting merges)
  // Removed afterwards
  result.addOperands(operand);

  // Operands from predecessor blocks
  for (int i = 0, e = inputs; i < e; ++i)
    result.addOperands(operand);

  result.addAttribute("control", builder.getBoolAttr(true));
}

static ::mlir::ParseResult parseControlMergeOp(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::Type resultType;
  ::mlir::SmallVector<::mlir::Type, 1> dataOperandsTypes;
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseLSquare())
    return ::mlir::failure();
  int size;
  if (parser.parseInteger(size))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultType))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  // Add types for the variadic operands
  dataOperandsTypes.assign(size, resultType);
  result.addTypes(resultType);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(allOperands, dataOperandsTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void printControlMergeOp(::mlir::OpAsmPrinter &p, ControlMergeOp op) {
  p << '[' << op.dataOperands().size() << ']';
  p << ' ';
  p << op.getOperation()->getOperands();
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << ::llvm::ArrayRef<::mlir::Type>(op.result().getType());
}

static ParseResult verifyFuncOp(handshake::FuncOp op) {
  // If this function is external there is nothing to do.
  if (op.isExternal())
    return success();

  // Verify that the argument list of the function and the arg list of the
  // entry block line up.  The trait already verified that the number of
  // arguments is the same between the signature and the block.
  auto fnInputTypes = op.getType().getInputs();
  Block &entryBlock = op.front();

  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return op.emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';

  // Verify that we have a name for each argument and result of this function.
  auto verifyPortNameAttr = [&](StringRef attrName,
                                unsigned numIOs) -> LogicalResult {
    auto portNamesAttr = op->getAttrOfType<ArrayAttr>(attrName);

    if (!portNamesAttr)
      return op.emitOpError() << "expected attribute '" << attrName << "'.";

    auto portNames = portNamesAttr.getValue();
    if (portNames.size() != numIOs)
      return op.emitOpError()
             << "attribute '" << attrName << "' has " << portNames.size()
             << " entries but is expected to have " << numIOs << ".";

    if (llvm::any_of(portNames,
                     [&](Attribute attr) { return !attr.isa<StringAttr>(); }))
      return op.emitOpError() << "expected all entries in attribute '"
                              << attrName << "' to be strings.";

    return success();
  };
  if (failed(verifyPortNameAttr("argNames", op.getNumArguments())))
    return failure();
  if (failed(verifyPortNameAttr("resNames", op.getNumResults())))
    return failure();

  return success();
}

/// Parses a FuncOp signature using
/// mlir::function_like_impl::parseFunctionSignature while getting access to the
/// parsed SSA names to store as attributes.
static ParseResult parseFuncOpArgs(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::OperandType> &entryArgs,
    SmallVectorImpl<Type> &argTypes, SmallVectorImpl<Attribute> &argNames,
    SmallVectorImpl<NamedAttrList> &argAttrs, SmallVectorImpl<Type> &resTypes,
    SmallVectorImpl<NamedAttrList> &resAttrs) {
  auto *context = parser.getContext();

  bool isVariadic;
  if (mlir::function_like_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/true, entryArgs, argTypes, argAttrs,
          isVariadic, resTypes, resAttrs)
          .failed())
    return failure();

  llvm::transform(entryArgs, std::back_inserter(argNames), [&](auto arg) {
    return StringAttr::get(context, arg.name.drop_front());
  });

  return success();
}

/// Generates names for a handshake.func input and output arguments, based on
/// the number of args as well as a prefix.
static SmallVector<Attribute> getFuncOpNames(Builder &builder, TypeRange types,
                                             StringRef prefix) {
  SmallVector<Attribute> resNames;
  llvm::transform(
      llvm::enumerate(types), std::back_inserter(resNames), [&](auto it) {
        bool lastOperand = it.index() == types.size() - 1;
        std::string suffix = lastOperand && it.value().template isa<NoneType>()
                                 ? "Ctrl"
                                 : std::to_string(it.index());
        return builder.getStringAttr(prefix + suffix);
      });
  return resNames;
}

void handshake::FuncOp::build(OpBuilder &builder, OperationState &state,
                              StringRef name, FunctionType type,
                              ArrayRef<NamedAttribute> attrs) {

  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());

  if (const auto *argNamesAttrIt = llvm::find_if(
          attrs, [&](auto attr) { return attr.first == "argNames"; });
      argNamesAttrIt == attrs.end())
    state.addAttribute("argNames", builder.getArrayAttr({}));

  if (llvm::find_if(attrs, [&](auto attr) {
        return attr.first == "resNames";
      }) == attrs.end())
    state.addAttribute("resNames", builder.getArrayAttr({}));

  state.addRegion();
}

/// Helper function for appending a string to an array attribute, and
/// rewriting the attribute back to the operation.
static void addStringToStringArrayAttr(Builder &builder, Operation *op,
                                       StringRef attrName, StringAttr str) {
  llvm::SmallVector<Attribute> attrs;
  llvm::copy(op->getAttrOfType<ArrayAttr>(attrName).getValue(),
             std::back_inserter(attrs));
  attrs.push_back(str);
  op->setAttr(attrName, builder.getArrayAttr(attrs));
}

void handshake::FuncOp::resolveArgAndResNames() {
  auto type = getType();
  Builder builder(getContext());

  /// Generate a set of fallback names. These are used in case names are
  /// missing from the currently set arg- and res name attributes.
  auto fallbackArgNames = getFuncOpNames(builder, type.getInputs(), "in");
  auto fallbackResNames = getFuncOpNames(builder, type.getResults(), "out");
  auto argNames = getArgNames().getValue();
  auto resNames = getResNames().getValue();

  /// Use fallback names where actual names are missing.
  auto resolveNames = [&](auto &fallbackNames, auto &actualNames,
                          StringRef attrName) {
    for (auto fallbackName : llvm::enumerate(fallbackNames)) {
      if (actualNames.size() <= fallbackName.index())
        addStringToStringArrayAttr(
            builder, this->getOperation(), attrName,
            fallbackName.value().template cast<StringAttr>());
    }
  };
  resolveNames(fallbackArgNames, argNames, "argNames");
  resolveNames(fallbackResNames, resNames, "resNames");
}

static ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  StringAttr nameAttr;
  SmallVector<OpAsmParser::OperandType, 4> args;
  SmallVector<Type, 4> argTypes, resTypes;
  SmallVector<NamedAttrList, 4> argAttributes, resAttributes;
  SmallVector<Attribute> argNames;

  // Parse signature
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parseFuncOpArgs(parser, args, argTypes, argNames, argAttributes, resTypes,
                      resAttributes))
    return failure();
  mlir::function_like_impl::addArgAndResultAttrs(builder, result, argAttributes,
                                                 resAttributes);

  // Set function type
  result.addAttribute(
      handshake::FuncOp::getTypeAttrName(),
      TypeAttr::get(builder.getFunctionType(argTypes, resTypes)));

  // Parse attributes
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  // If argNames and resNames wasn't provided manually, infer argNames attribute
  // from the parsed SSA names and resNames from our naming convention.
  if (!result.attributes.get("argNames"))
    result.addAttribute("argNames", builder.getArrayAttr(argNames));
  if (!result.attributes.get("resNames")) {
    auto resNames = getFuncOpNames(builder, resTypes, "out");
    result.addAttribute("resNames", builder.getArrayAttr(resNames));
  }

  // Parse region
  auto *body = result.addRegion();
  return parser.parseRegion(*body, args, argTypes);
}

static void printFuncOp(OpAsmPrinter &p, handshake::FuncOp op) {
  FunctionType fnType = op.getType();
  mlir::function_like_impl::printFunctionLikeOp(p, op, fnType.getInputs(),
                                                /*isVariadic=*/true,
                                                fnType.getResults());
}

namespace {
struct EliminateSimpleControlMergesPattern
    : mlir::OpRewritePattern<ControlMergeOp> {
  using mlir::OpRewritePattern<ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ControlMergeOp op,
                                PatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult EliminateSimpleControlMergesPattern::matchAndRewrite(
    ControlMergeOp op, PatternRewriter &rewriter) const {
  auto dataResult = op.getResult(0);
  auto choiceResult = op.getResult(1);
  auto choiceUnused = choiceResult.use_empty();
  if (!choiceUnused && !choiceResult.hasOneUse())
    return failure();

  Operation *choiceUser;
  if (choiceResult.hasOneUse()) {
    choiceUser = choiceResult.getUses().begin().getUser();
    if (!isa<SinkOp>(choiceUser))
      return failure();
  }

  auto merge = rewriter.create<MergeOp>(op.getLoc(), dataResult.getType(),
                                        op.dataOperands());

  for (auto &use : dataResult.getUses()) {
    auto *user = use.getOwner();
    rewriter.updateRootInPlace(
        user, [&]() { user->setOperand(use.getOperandNumber(), merge); });
  }

  if (choiceUnused) {
    rewriter.eraseOp(op);
    return success();
  }

  rewriter.eraseOp(choiceUser);
  rewriter.eraseOp(op);
  return success();
}

void ControlMergeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<EliminateSimpleControlMergesPattern>(context);
}

bool handshake::ControlMergeOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  auto op = getOperation();
  bool found = false;
  int i = 0;
  for (mlir::Value in : op->getOperands()) {
    if (valueMap.count(in) == 1) {
      if (found)
        op->emitError("More than one valid input to CMerge!");
      auto t = valueMap[in];
      valueMap[op->getResult(0)] = t;
      timeMap[op->getResult(0)] = timeMap[in];

      valueMap[op->getResult(1)] = APInt(INDEX_WIDTH, i);
      timeMap[op->getResult(1)] = timeMap[in];

      // Consume the inputs.
      valueMap.erase(in);

      found = true;
    }
    i++;
  }
  if (!found)
    op->emitError("No valid input to CMerge!");
  scheduleList = toVector(op->getResults());
  return true;
}

void handshake::BranchOp::build(OpBuilder &builder, OperationState &result,
                                Value dataOperand) {
  auto type = dataOperand.getType();
  result.types.push_back(type);
  result.addOperands(dataOperand);

  // Branch is control-only if it is the no-data output of a ControlMerge or a
  // StartOp This holds because Branches are inserted before Forks
  auto *op = dataOperand.getDefiningOp();
  bool isControl = ((dyn_cast<ControlMergeOp>(op) || dyn_cast<StartOp>(op)) &&
                    dataOperand == op->getResult(0))
                       ? true
                       : false;
  result.addAttribute("control", builder.getBoolAttr(isControl));
}

void handshake::BranchOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<circt::handshake::EliminateSimpleBranchesPattern>(context);
}

void handshake::BranchOp::execute(std::vector<llvm::Any> &ins,
                                  std::vector<llvm::Any> &outs) {
  outs[0] = ins[0];
}

bool handshake::BranchOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 0);
}

std::string handshake::ConditionalBranchOp::getOperandName(unsigned int idx) {
  assert(idx == 0 || idx == 1);
  return idx == 0 ? "cond" : "data";
}

std::string handshake::ConditionalBranchOp::getResultName(unsigned int idx) {
  assert(idx == 0 || idx == 1);
  return idx == ConditionalBranchOp::falseIndex ? "outFalse" : "outTrue";
}

void handshake::ConditionalBranchOp::build(OpBuilder &builder,
                                           OperationState &result,
                                           Value condOperand,
                                           Value dataOperand) {
  auto type = dataOperand.getType();
  result.types.append(2, type);
  result.addOperands(condOperand);
  result.addOperands(dataOperand);

  // Branch is control-only if it is the no-data output of a ControlMerge or a
  // StartOp This holds because Branches are inserted before Forks
  auto *op = dataOperand.getDefiningOp();
  bool isControl = ((dyn_cast<ControlMergeOp>(op) || dyn_cast<StartOp>(op)) &&
                    dataOperand == op->getResult(0))
                       ? true
                       : false;
  result.addAttribute("control", builder.getBoolAttr(isControl));
}

bool handshake::ConditionalBranchOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  auto op = getOperation();
  mlir::Value control = op->getOperand(0);
  if (valueMap.count(control) == 0)
    return false;
  auto controlValue = valueMap[control];
  auto controlTime = timeMap[control];
  mlir::Value in = op->getOperand(1);
  if (valueMap.count(in) == 0)
    return false;
  auto inValue = valueMap[in];
  auto inTime = timeMap[in];
  mlir::Value out = llvm::any_cast<APInt>(controlValue) != 0 ? op->getResult(0)
                                                             : op->getResult(1);
  double time = std::max(controlTime, inTime);
  valueMap[out] = inValue;
  timeMap[out] = time;
  scheduleList.push_back(out);

  // Consume the inputs.
  valueMap.erase(control);
  valueMap.erase(in);
  return true;
}

void StartOp::build(OpBuilder &builder, OperationState &result) {
  // Control-only output, has no type
  auto type = builder.getNoneType();
  result.types.push_back(type);
  result.addAttribute("control", builder.getBoolAttr(true));
}

bool handshake::StartOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  return true;
}

void EndOp::build(OpBuilder &builder, OperationState &result, Value operand) {
  result.addOperands(operand);
}

bool handshake::EndOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  return true;
}

void handshake::ReturnOp::build(OpBuilder &builder, OperationState &result,
                                ArrayRef<Value> operands) {
  result.addOperands(operands);
}

void SinkOp::build(OpBuilder &builder, OperationState &result, Value operand) {
  result.addOperands(operand);
}

bool handshake::SinkOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  valueMap.erase(getOperand());
  return true;
}

std::string handshake::ConstantOp::getOperandName(unsigned int idx) {
  assert(idx == 0);
  return "ctrl";
}

void handshake::ConstantOp::build(OpBuilder &builder, OperationState &result,
                                  Attribute value, Value operand) {
  result.addOperands(operand);

  auto type = value.getType();
  result.types.push_back(type);

  result.addAttribute("value", value);
}

void handshake::ConstantOp::execute(std::vector<llvm::Any> &ins,
                                    std::vector<llvm::Any> &outs) {
  auto attr = (*this)->getAttrOfType<mlir::IntegerAttr>("value");
  outs[0] = attr.getValue();
}

bool handshake::ConstantOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 0);
}

void handshake::ConstantOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<circt::handshake::EliminateSunkConstantsPattern>(context);
}

void handshake::TerminatorOp::build(OpBuilder &builder, OperationState &result,
                                    ArrayRef<Block *> successors) {
  // Add all the successor blocks of the block which contains this terminator
  result.addSuccessors(successors);
  // for (auto &succ : successors)
  //   result.addSuccessor(succ, {});
}

static std::string getMemoryOperandName(unsigned nStores, unsigned idx) {
  std::string name;
  if (idx < nStores * 2) {
    bool isData = idx % 2 == 0;
    name = isData ? "stData" + std::to_string(idx / 2)
                  : "stAddr" + std::to_string(idx / 2);
  } else {
    idx -= 2 * nStores;
    name = "ldAddr" + std::to_string(idx);
  }
  return name;
}

std::string handshake::MemoryOp::getOperandName(unsigned int idx) {
  return getMemoryOperandName(getStCount().getZExtValue(), idx);
}

static std::string getMemoryResultName(unsigned nLoads, unsigned nStores,
                                       unsigned idx) {
  std::string name;
  if (idx < nLoads)
    name = "lddata" + std::to_string(idx);
  else if (idx < nLoads + nStores)
    name = "stDone" + std::to_string(idx - nLoads);
  else
    name = "ldDone" + std::to_string(idx - nLoads - nStores);
  return name;
}

std::string handshake::MemoryOp::getResultName(unsigned int idx) {
  return getMemoryResultName(getLdCount().getZExtValue(),
                             getStCount().getZExtValue(), idx);
}

static LogicalResult verifyMemoryOp(handshake::MemoryOp op) {
  auto memrefType = op.getMemRefType();

  if (memrefType.getNumDynamicDims() != 0)
    return op.emitOpError()
           << "memref dimensions for handshake.memory must be static.";
  if (memrefType.getShape().size() != 1)
    return op.emitOpError() << "memref must have only a single dimension.";

  unsigned st_count = op.st_count();
  unsigned ld_count = op.ld_count();
  int addressCount = memrefType.getShape().size();

  auto inputType = op.inputs().getType();
  auto outputType = op.outputs().getType();
  Type dataType = memrefType.getElementType();
  
  unsigned numOperands = static_cast<int>(op.inputs().size());
  unsigned numResults = static_cast<int>(op.outputs().size());
  if(numOperands != (1+addressCount)*st_count + addressCount*ld_count)
    return op.emitOpError("number of operands ") << numOperands << " does not match number expected of " << 
    2*st_count + ld_count << " with " << addressCount << " address inputs per port";

  if(numResults != st_count + 2*ld_count)
    return op.emitOpError("number of results ") << numResults << " does not match number expected of " << 
    st_count + 2*ld_count << " with " << addressCount << " address inputs per port";

  Type addressType = st_count > 0 ? inputType[1] : inputType[0];

  // Add types for the variadic operands
  for(unsigned i = 0; i < st_count; i++) {
    if(inputType[2*i] != dataType)
      return op.emitOpError("data type for store port ") << i << ":" << inputType[2*i] <<
      " doesn't match memory type " << dataType;
    if(inputType[2*i+1] != addressType)
      return op.emitOpError("address type for store port ") << i << ":" << inputType[2*i+1] <<
      " doesn't match address type " << addressType;
  }
  for(unsigned i = 0; i < ld_count; i++) {
    Type ldAddressType = inputType[2*st_count + i];
    if(ldAddressType != addressType)
      return op.emitOpError("address type for load port ") << i << ":" << ldAddressType <<
      " doesn't match address type " << addressType;
  }
  for(unsigned i = 0; i < ld_count; i++) {
    if(outputType[i] != dataType)
      return op.emitOpError("data type for load port ") << i << ":" << outputType[i] <<
      " doesn't match memory type " << dataType;
  }
  for(unsigned i = 0; i < st_count; i++) {
    Type syncType = outputType[ld_count+i];
    if(!syncType.isa<::mlir::NoneType>())
      return op.emitOpError("data type for sync port for store port ") << i << ":" << syncType <<
      " is not 'none'";
  }
  for(unsigned i = 0; i < ld_count; i++) {
    Type syncType = outputType[ld_count+st_count+i];
    if(!syncType.isa<::mlir::NoneType>())
      return op.emitOpError("data type for sync port for load port ") << i << ":" << syncType <<
      " is not 'none'";
  }

  return success();
}

std::string handshake::ExternalMemoryOp::getOperandName(unsigned int idx) {
  if (idx == 0)
    return "extmem";

  return getMemoryOperandName(stCount(), idx - 1);
}

std::string handshake::ExternalMemoryOp::getResultName(unsigned int idx) {
  return getMemoryResultName(ldCount(), stCount(), idx);
}

void ExternalMemoryOp::build(OpBuilder &builder, OperationState &result,
                             Value memref, ArrayRef<Value> inputs, int ldCount,
                             int stCount, int id) {
  SmallVector<Value> ops;
  ops.push_back(memref);
  llvm::append_range(ops, inputs);
  result.addOperands(ops);

  auto memrefType = memref.getType().cast<MemRefType>();

  // Data outputs (get their type from memref)
  result.types.append(ldCount, memrefType.getElementType());

  // Control outputs
  result.types.append(stCount + ldCount, builder.getNoneType());

  // Memory ID (individual ID for each MemoryOp)
  Type i32Type = builder.getIntegerType(32);
  result.addAttribute("id", builder.getIntegerAttr(i32Type, id));
  result.addAttribute("ldCount", builder.getIntegerAttr(i32Type, ldCount));
  result.addAttribute("stCount", builder.getIntegerAttr(i32Type, stCount));
}

bool handshake::ExternalMemoryOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  // todo(mortbopet): implement execution of ExternalMemoryOp's.
  assert(false && "implement me");
  return 0;
}

void MemoryOp::build(OpBuilder &builder, OperationState &result,
                     ArrayRef<Value> operands, int outputs, int control_outputs,
                     bool lsq, int id, Value memref) {
  result.addOperands(operands);

  auto memrefType = memref.getType().cast<MemRefType>();

  // Data outputs (get their type from memref)
  result.types.append(outputs, memrefType.getElementType());

  // Control outputs
  result.types.append(control_outputs, builder.getNoneType());

  // Indicates whether a memory is an LSQ
  result.addAttribute("lsq", builder.getBoolAttr(lsq));

  // Memref info
  result.addAttribute("type", TypeAttr::get(memrefType));

  // Memory ID (individual ID for each MemoryOp)
  Type i32Type = builder.getIntegerType(32);
  result.addAttribute("id", builder.getIntegerAttr(i32Type, id));

  if (!lsq) {

    result.addAttribute("ld_count", builder.getIntegerAttr(i32Type, outputs));
    result.addAttribute(
        "st_count", builder.getIntegerAttr(i32Type, control_outputs - outputs));
  }
}

static ::mlir::ParseResult parseMemoryOp(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> operandTypes;
  ::mlir::SmallVector<::mlir::Type, 1> resultTypes;
  ::mlir::Type addressRawType[1];
  ::mlir::Type dataRawType[1];
  ::llvm::ArrayRef<::mlir::Type> addressType(addressRawType);
  ::llvm::ArrayRef<::mlir::Type> dataType(dataRawType);
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseType(addressRawType[0]))
    return ::mlir::failure();

  int st_count = result.attributes.get("st_count").cast<::mlir::IntegerAttr>().getInt();
  int ld_count = result.attributes.get("ld_count").cast<::mlir::IntegerAttr>().getInt();
  Type type = result.attributes.get("type").cast<::mlir::TypeAttr>().getValue();
  dataRawType[0] = type.cast<::mlir::MemRefType>().getElementType();

  ::mlir::Type noneType = parser.getBuilder().getNoneType();
  // Add types for the variadic operands
  for(int i = 0; i < st_count; i++) {
    operandTypes.push_back(dataRawType[0]);
    operandTypes.push_back(addressRawType[0]);
  }
  for(int i = 0; i < ld_count; i++) {
    operandTypes.push_back(addressRawType[0]);
  }
  for(int i = 0; i < ld_count; i++) {
    resultTypes.push_back(dataRawType[0]);
  }
  for(int i = 0; i < st_count + ld_count; i++) {
    resultTypes.push_back(noneType);
  }
  
  result.addTypes(resultTypes);
  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void printMemoryOp(::mlir::OpAsmPrinter &p, MemoryOp op) {
  p << ' ';
  p << op.getOperation()->getOperands();
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << op.inputs().getType()[1];
}

bool handshake::MemoryOp::allocateMemory(
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<double> &storeTimes) {
  unsigned id = getID();
  if (memoryMap.count(id))
    return false;

  auto type = getMemRefType();
  std::vector<llvm::Any> in;

  ArrayRef<int64_t> shape = type.getShape();
  int allocationSize = 1;
  unsigned count = 0;
  for (int64_t dim : shape) {
    if (dim > 0)
      allocationSize *= dim;
    else {
      assert(count < in.size());
      allocationSize *= llvm::any_cast<APInt>(in[count++]).getSExtValue();
    }
  }
  unsigned ptr = store.size();
  store.resize(ptr + 1);
  storeTimes.resize(ptr + 1);
  store[ptr].resize(allocationSize);
  storeTimes[ptr] = 0.0;
  mlir::Type elementType = type.getElementType();
  int width = elementType.getIntOrFloatBitWidth();
  for (int i = 0; i < allocationSize; i++) {
    if (elementType.isa<mlir::IntegerType>()) {
      store[ptr][i] = APInt(width, 0);
    } else if (elementType.isa<mlir::FloatType>()) {
      store[ptr][i] = APFloat(0.0);
    } else {
      llvm_unreachable("Unknown result type!\n");
    }
  }

  memoryMap[id] = ptr;
  return true;
}

bool handshake::MemoryOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  auto op = getOperation();
  int opIndex = 0;
  bool notReady = false;
  unsigned id = getID(); // The ID of this memory.
  unsigned buffer = memoryMap[id];

  for (unsigned i = 0; i < getStCount().getZExtValue(); i++) {
    mlir::Value data = op->getOperand(opIndex++);
    mlir::Value address = op->getOperand(opIndex++);
    mlir::Value nonceOut = op->getResult(getLdCount().getZExtValue() + i);
    if ((!valueMap.count(data) || !valueMap.count(address))) {
      notReady = true;
      continue;
    }
    auto addressValue = valueMap[address];
    auto addressTime = timeMap[address];
    auto dataValue = valueMap[data];
    auto dataTime = timeMap[data];

    assert(buffer < store.size());
    auto &ref = store[buffer];
    unsigned offset = llvm::any_cast<APInt>(addressValue).getZExtValue();
    assert(offset < ref.size());
    ref[offset] = dataValue;

    // Implicit none argument
    APInt apnonearg(1, 0);
    valueMap[nonceOut] = apnonearg;
    double time = std::max(addressTime, dataTime);
    timeMap[nonceOut] = time;
    scheduleList.push_back(nonceOut);
    // Consume the inputs.
    valueMap.erase(data);
    valueMap.erase(address);
  }

  for (unsigned i = 0; i < getLdCount().getZExtValue(); i++) {
    mlir::Value address = op->getOperand(opIndex++);
    mlir::Value dataOut = op->getResult(i);
    mlir::Value nonceOut = op->getResult(getLdCount().getZExtValue() +
                                         getStCount().getZExtValue() + i);
    if (!valueMap.count(address)) {
      notReady = true;
      continue;
    }
    auto addressValue = valueMap[address];
    auto addressTime = timeMap[address];
    assert(buffer < store.size());
    auto &ref = store[buffer];
    unsigned offset = llvm::any_cast<APInt>(addressValue).getZExtValue();
    assert(offset < ref.size());

    valueMap[dataOut] = ref[offset];
    timeMap[dataOut] = addressTime;
    // Implicit none argument
    APInt apnonearg(1, 0);
    valueMap[nonceOut] = apnonearg;
    timeMap[nonceOut] = addressTime;
    scheduleList.push_back(dataOut);
    scheduleList.push_back(nonceOut);
    // Consume the inputs.
    valueMap.erase(address);
  }
  return (notReady) ? false : true;
}

void handshake::LoadOp::build(OpBuilder &builder, OperationState &result,
                              Value memref, ArrayRef<Value> indices) {
  // Address indices
  // result.addOperands(memref);
  result.addOperands(indices);

  // Data type
  auto memrefType = memref.getType().cast<MemRefType>();

  // Data output (from load to successor ops)
  result.types.push_back(memrefType.getElementType());

  // Address outputs (to lsq)
  result.types.append(indices.size(), builder.getIndexType());
}

static ::mlir::ParseResult parseLoadOp(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> dataOperandsTypes;
  ::mlir::Type addressRawType[1];
  ::mlir::Type dataRawType[1];
  ::llvm::ArrayRef<::mlir::Type> addressType(addressRawType);
  ::llvm::ArrayRef<::mlir::Type> dataType(dataRawType);
  ::mlir::SmallVector<::mlir::Type, 1> addressTypes;
  ::mlir::SmallVector<::mlir::Type, 1> dataTypes;
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseLSquare())
    return ::mlir::failure();
  int size;
  if (parser.parseInteger(size))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseType(addressRawType[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseType(dataRawType[0]))
    return ::mlir::failure();

  ::mlir::Type noneType = parser.getBuilder().getNoneType();
  // Add types for the variadic operands
  dataTypes.assign(size, dataRawType[0]);
  addressTypes.assign(size, addressRawType[0]);

  dataOperandsTypes.assign(size, addressRawType[0]);
  dataOperandsTypes.push_back(dataRawType[0]);
  dataOperandsTypes.push_back(noneType);
  
  result.addTypes(dataTypes);
  result.addTypes(addressType);
  if (parser.resolveOperands(allOperands, dataOperandsTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void printLoadOp(::mlir::OpAsmPrinter &p, LoadOp op) {
  p << '[' << op.address().size() << ']';
  p << ' ';
  p << op.getOperation()->getOperands();
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << op.address().getType();
  p << ", ";
  p << op.data().getType();
}

bool handshake::LoadOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  auto op = getOperation();
  mlir::Value address = op->getOperand(0);
  mlir::Value data = op->getOperand(1);
  mlir::Value nonce = op->getOperand(2);
  mlir::Value addressOut = op->getResult(1);
  mlir::Value dataOut = op->getResult(0);
  if ((valueMap.count(address) && !valueMap.count(nonce)) ||
      (!valueMap.count(address) && valueMap.count(nonce)) ||
      (!valueMap.count(address) && !valueMap.count(nonce) &&
       !valueMap.count(data)))
    return false;
  if (valueMap.count(address) && valueMap.count(nonce)) {
    auto addressValue = valueMap[address];
    auto addressTime = timeMap[address];
    auto nonceValue = valueMap[nonce];
    auto nonceTime = timeMap[nonce];
    valueMap[addressOut] = addressValue;
    double time = std::max(addressTime, nonceTime);
    timeMap[addressOut] = time;
    scheduleList.push_back(addressOut);
    // Consume the inputs.
    valueMap.erase(address);
    valueMap.erase(nonce);
  } else if (valueMap.count(data)) {
    auto dataValue = valueMap[data];
    auto dataTime = timeMap[data];
    valueMap[dataOut] = dataValue;
    timeMap[dataOut] = dataTime;
    scheduleList.push_back(dataOut);
    // Consume the inputs.
    valueMap.erase(data);
  } else {
    llvm_unreachable("why?");
  }
  return true;
}

void handshake::StoreOp::build(OpBuilder &builder, OperationState &result,
                               Value valueToStore, ArrayRef<Value> indices) {
  // Data
  result.addOperands(valueToStore);

  // Address indices
  result.addOperands(indices);

  // Data output (from store to LSQ)
  result.types.push_back(valueToStore.getType());

  // Address outputs (from store to lsq)
  result.types.append(indices.size(), builder.getIndexType());
}

void handshake::StoreOp::execute(std::vector<llvm::Any> &ins,
                                 std::vector<llvm::Any> &outs) {
  // Forward the address and data to the memory op.
  outs[0] = ins[0];
  outs[1] = ins[1];
}

static ::mlir::ParseResult parseStoreOp(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> dataOperandsTypes;
  ::mlir::Type addressRawType[1];
  ::mlir::Type dataRawType[1];
  ::llvm::ArrayRef<::mlir::Type> addressType(addressRawType);
  ::llvm::ArrayRef<::mlir::Type> dataType(dataRawType);
  ::mlir::SmallVector<::mlir::Type, 1> addressTypes;
  ::mlir::SmallVector<::mlir::Type, 1> dataTypes;
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseLSquare())
    return ::mlir::failure();
  int size;
  if (parser.parseInteger(size))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseType(dataRawType[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseType(addressRawType[0]))
    return ::mlir::failure();

  ::mlir::Type noneType = parser.getBuilder().getNoneType();
  // Add types for the variadic operands
  dataTypes.assign(size, dataRawType[0]);
  addressTypes.assign(size, addressRawType[0]);

  dataOperandsTypes.push_back(dataRawType[0]);
  dataOperandsTypes.insert(dataOperandsTypes.end(), size, addressRawType[0]);
//  for(int i = 0; dataOperandsTypes.assign(size, addressRawType[0]);
  dataOperandsTypes.push_back(noneType);
  
  result.addTypes(dataType);
  result.addTypes(addressTypes);
  if (parser.resolveOperands(allOperands, dataOperandsTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void printStoreOp(::mlir::OpAsmPrinter &p, StoreOp op) {
  p << '[' << op.address().size() << ']';
  p << ' ';
  p << op.getOperation()->getOperands();
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{});
  p << ' ' << ":";
  p << ' ';
  p << op.data().getType();
  p << ", ";
  p << op.address().getType();
}

bool handshake::StoreOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 1);
}

void JoinOp::build(OpBuilder &builder, OperationState &result,
                   ArrayRef<Value> operands) {
  auto type = builder.getNoneType();
  result.types.push_back(type);

  result.addOperands(operands);

  result.addAttribute("control", builder.getBoolAttr(true));
}

void handshake::JoinOp::execute(std::vector<llvm::Any> &ins,
                                std::vector<llvm::Any> &outs) {
  outs[0] = ins[0];
}

bool handshake::JoinOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 1);
}

static LogicalResult verifyInstanceOp(handshake::InstanceOp op) {
  if (op->getNumOperands() == 0)
    return op.emitOpError() << "must provide at least a control operand.";

  if (!op.getControl().getType().dyn_cast<NoneType>())
    return op.emitOpError()
           << "last operand must be a control (none-typed) operand.";

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//


static LogicalResult verify(handshake::ReturnOp op) {
  auto *parent = op->getParentOp();
  auto function = dyn_cast<handshake::FuncOp>(parent);
  if (!function)
    return op.emitOpError("must have a handshake.func parent");

  // The operand number and types must match the function signature.
  const auto &results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError("has ")
           << op.getNumOperands()
           << " operands, but enclosing function returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (op.getOperand(i).getType() != results[i])
      return op.emitError()
             << "type of return operand " << i << " ("
             << op.getOperand(i).getType()
             << ") doesn't match function result type (" << results[i] << ")";

  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/Handshake/Handshake.cpp.inc"
