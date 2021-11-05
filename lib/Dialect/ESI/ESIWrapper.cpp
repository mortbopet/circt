//===- ESIPasses.cpp - ESI wrapper generation -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wrap hardware-like operations within an ESI interface.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace esi {
#define GEN_PASS_CLASSES
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace esi
} // namespace circt

using namespace circt;
using namespace circt::comb;
using namespace circt::esi;
using namespace circt::hw;
using namespace circt::sv;

static esi::ChannelPort wrapChannel(OpBuilder &builder, Type type) {
  return esi::ChannelPort::get(builder.getContext(), type);
}

namespace {
/// Run all the physical lowerings.
struct CreateESIWrapper : public CreateESIWrapperBase<CreateESIWrapper> {
  void runOnOperation() override;
  Operation *inferTopOp(ModuleOp module);
  LogicalResult createWrapper(OpBuilder &builder, calyx::ComponentOp op,
                              SmallVectorImpl<Operation *> &opsToKeep);
};

static bool isCalyxInterfacePort(StringRef portName) {
  auto interfacePorts = {"clk", "done", "go", "reset"};
  return llvm::find(interfacePorts, portName) != interfacePorts.end();
}

static Value getInputPort(hw::HWModuleOp module, StringRef name) {
  auto portInfos = module.getPorts().inputs;
  auto it = llvm::find_if(portInfos, [&](auto portInfo) {
    return portInfo.name.getValue() == name;
  });
  assert(it != portInfos.end() && "Input port not found in module");
  return module.getArgument(it->argNum);
}

static Value getOutputPort(hw::HWModuleOp module, StringRef name) {
  auto portInfos = module.getPorts().outputs;
  auto it = llvm::find_if(portInfos, [&](auto portInfo) {
    return portInfo.name.getValue() == name;
  });
  assert(it != portInfos.end() && "Output port not found in module");
  return module->getResult(it->argNum);
}

static StringAttr getCalyxPortName(calyx::ComponentOp op, bool isInput,
                                   unsigned argNum) {

  unsigned nameIdx = argNum;
  if (!isInput)
    nameIdx += op.getInputPortInfo().size();
  return op.portNames()[nameIdx].cast<StringAttr>();
}

static esi::ChannelBufferOptions getChannelBufferOptions(OpBuilder &builder,
                                                         unsigned stages) {
  auto opts = esi::ChannelBufferOptions::get(builder.getI64IntegerAttr(stages),
                                             nullptr, builder.getContext());
  assert(opts);
  return opts;
}

static bool isGoDonePort(StringAttr portNameAttr) {
  auto portName = portNameAttr.getValue();
  return portName == "go" || portName == "done";
}

LogicalResult
CreateESIWrapper::createWrapper(OpBuilder &builder, calyx::ComponentOp op,
                                SmallVectorImpl<Operation *> &opsToKeep) {

  // Build ports for the ESI-wrapped calyx instance.
  SmallVector<hw::PortInfo, 8> ports;
  auto addHWPortInfo = [&](auto portInfos) {
    for (auto portInfo : enumerate(portInfos)) {
      hw::PortInfo hwPortInfo;
      bool isInput = portInfo.value().direction == calyx::Direction::Input;
      hwPortInfo.direction =
          isInput ? hw::PortDirection::INPUT : hw::PortDirection::OUTPUT;

      auto portType = portInfo.value().type;
      hwPortInfo.type = isCalyxInterfacePort(portInfo.value().name.getValue())
                            ? portType
                            : wrapChannel(builder, portType);
      hwPortInfo.argNum = portInfo.index();
      hwPortInfo.name = getCalyxPortName(op, isInput, hwPortInfo.argNum);

      // Expose everything but the go/done ports in the ESI wrapper.
      if (!isGoDonePort(hwPortInfo.name))
        ports.push_back(hwPortInfo);
    }
  };

  addHWPortInfo(op.getInputPortInfo());
  addHWPortInfo(op.getOutputPortInfo());

  // Create the esi wrapper module
  auto hwModule = builder.create<hw::HWModuleOp>(
      op.getLoc(), builder.getStringAttr(op.getName() + "_esi"), ports);
  opsToKeep.push_back(hwModule);

  // Create a reference to the external Calyx verilog module
  auto calyxModule = calyx::getExternHWModule(builder, op);
  opsToKeep.push_back(calyxModule);

  // Unpack some values

  // Build body
  builder.setInsertionPointToStart(hwModule.getBodyBlock());
  ImplicitLocOpBuilder bodyBuilder(hwModule.getLoc(), hwModule.getBody());
  BackedgeBuilder bb(builder, hwModule.getLoc());

  struct ESIPortVRValueMapping {
    Value valid;
    Value ready;
  };

  SmallVector<esi::ChannelBuffer> inputBuffers, outputBuffers;
  SmallVector<ESIPortVRValueMapping> inputVRMap, outputVRMap;
  SmallVector<Value> instanceOperands;

  // Calyx instance 'go' signal. Defined here due to use as 'ready' signal for
  // input buffers.
  auto instanceGo = bb.get(builder.getI1Type());

  auto rst = getInputPort(hwModule, "reset");
  auto rstn = bodyBuilder.create<comb::XorOp>(rst).getResult();
  auto clk = getInputPort(hwModule, "clk");

  // Create an input buffer for each input channel port.
  for (auto inPort : hwModule.getPorts().inputs) {
    if (auto channelPort = inPort.type.dyn_cast<esi::ChannelPort>();
        channelPort) {
      auto channelOp = bodyBuilder.create<esi::ChannelBuffer>(
          channelPort, clk, rstn, hwModule.getArgument(inPort.argNum),
          getChannelBufferOptions(builder, inputBufferSizeOpt.getValue()));
      inputBuffers.push_back(channelOp);
    }
  }

  // Unwrap input buffers
  for (auto inBuffer : inputBuffers) {
    auto unwrapVrOp = bodyBuilder.create<esi::UnwrapValidReady>(
        inBuffer.getResult(), instanceGo);
    ESIPortVRValueMapping valueMapping = {unwrapVrOp.getResult(1), instanceGo};
    inputVRMap.push_back(valueMapping);
    instanceOperands.push_back(unwrapVrOp.getResult(0));
  }

  // Instantiate Calyx component
  instanceOperands.push_back(clk);
  instanceOperands.push_back(rst);
  instanceOperands.push_back(instanceGo);
  auto calyxInstance = bodyBuilder.create<hw::InstanceOp>(
      calyxModule, op.getName(), instanceOperands);

  auto instanceDone = calyxInstance.getResults().back();

  // Wrap calyx instance output into VR interface, and instantiate a buffer for
  // that VR interface. Drop back to skip done signal.
  for (auto instanceOutput :
       llvm::enumerate(calyxInstance.getResults().drop_back())) {
    auto wrapVrValid = instanceDone;
    auto wrapVrOp = bodyBuilder.create<esi::WrapValidReady>(
        instanceOutput.value(), wrapVrValid);

    // Create an output buffer
    auto channelOp = bodyBuilder.create<esi::ChannelBuffer>(
        wrapChannel(builder, instanceOutput.value().getType()), clk, rstn,
        wrapVrOp.getResult(0),
        getChannelBufferOptions(builder, inputBufferSizeOpt.getValue()));
    outputBuffers.push_back(channelOp);

    ESIPortVRValueMapping valueMapping = {wrapVrOp.getResult(1), wrapVrValid};
    outputVRMap.push_back(valueMapping);
  }

  // Build Calyx instance monitor state machine. Could be an FSM but it's so
  // simple we'll just manually elaborate it. A single register run_reg with
  // next state defined as:
  //    run_reg <== (run_reg & !done) | instanceGo
  auto runNext = bb.get(builder.getI1Type());
  auto runReg = bodyBuilder.create<seq::CompRegOp>(
      builder.getI1Type(), runNext, clk, builder.getStringAttr("runReg"), rst,
      bodyBuilder.create<hw::ConstantOp>(builder.getI1Type(), 0));
  runNext.setValue(bodyBuilder.create<comb::OrOp>(
      bodyBuilder.create<comb::AndOp>(
          runReg.getResult(),
          bodyBuilder.create<comb::XorOp>(instanceDone).getResult()),
      instanceGo));

  // Build Calyx instance 'go' logic. Go is asserted when all input buffers are
  // valid, the instance is currently not running and the output buffer is
  // ready.
  auto notRunning =
      bodyBuilder.create<comb::XorOp>(runReg.getResult()).getResult();
  Value andTree = notRunning;
  for (auto inputVR : inputVRMap)
    andTree =
        bodyBuilder.create<comb::AndOp>(andTree, inputVR.valid).getResult();
  for (auto outputVR : outputVRMap)
    andTree =
        bodyBuilder.create<comb::AndOp>(andTree, outputVR.ready).getResult();
  andTree = bodyBuilder.create<comb::AndOp>(andTree, notRunning).getResult();
  instanceGo.setValue(andTree);

  // Return output channels
  llvm::SmallVector<Value> outputChannels;
  llvm::transform(outputBuffers, std::back_inserter(outputChannels),
                  [](auto buffer) { return buffer.getResult(); });
  bodyBuilder.create<hw::OutputOp>(outputChannels);
  // Remove the default built output op.
  hwModule.getBodyBlock()->back().erase();
  return success();
}

static Optional<Operation *> getCalyxTopComponent(ModuleOp moduleOp) {
  auto programOps = moduleOp.getOps<calyx::ProgramOp>();
  if (!programOps.empty()) {
    auto program = *programOps.begin();
    return {program.getEntryPointComponent()};
  }
  return {};
}

Operation *CreateESIWrapper::inferTopOp(ModuleOp module) {
  // todo: How to do this in a pretty way?
  if (auto calyxTop = getCalyxTopComponent(module); calyxTop.hasValue())
    return calyxTop.getValue();

  // if firrtl module...

  // if handshake module...

  // At this point, go by symbol provided by user
  if (topOpt.empty()) {
    signalPassFailure();
    mlir::emitError(module.getLoc()) << "Top module could not be inferred; you "
                                        "must provide a --top level function.";
    return nullptr;
  }

  /// Operations can come in all manners of funky nested structures, so we'll
  /// just be agnostic to how the program is layed out, and simply search for
  /// any symbol equal to the requested top module.
  Operation *top = nullptr;
  auto symbolLookup = [&](Operation *op, bool /*allUsesVisible*/) {
    if (top)
      return;

    mlir::SymbolOpInterface symbol = dyn_cast<mlir::SymbolOpInterface>(op);
    if (symbol && symbol.getName() == topOpt)
      top = op;
  };

  SymbolTable::walkSymbolTables(module, true, symbolLookup);
  if (!top) {
    signalPassFailure();
    mlir::emitError(module.getLoc())
        << "Top-level symbol '" << topOpt << "' not found in module.";
    return nullptr;
  }
  return top;
}

void CreateESIWrapper::runOnOperation() {
  auto module = getOperation();
  auto ctx = &getContext();

  Operation *top = inferTopOp(module);
  if (!top)
    return;

  auto builder = OpBuilder(ctx);
  builder.setInsertionPointToStart(module.getBody());
  SmallVector<Operation *> opsToKeep;

  auto res = llvm::TypeSwitch<Operation *, LogicalResult>(top)
                 .Case<calyx::ComponentOp>([&](auto op) {
                   return createWrapper(builder, op, opsToKeep);
                 })
                 .Default([](auto op) {
                   return op->emitOpError() << "Unsupported operation";
                 });

  // Erase everything which is not the wrapped module
  for (auto &op : llvm::make_early_inc_range(*module.getBody())) {
    auto it = llvm::find(opsToKeep, &op);
    if (it == opsToKeep.end())
      op.erase();
  }

  if (res.failed())
    signalPassFailure();
}

} // anonymous namespace

namespace circt {
namespace esi {

std::unique_ptr<OperationPass<ModuleOp>> createESIWrapperPass() {
  return std::make_unique<CreateESIWrapper>();
}

} // namespace esi
} // namespace circt
