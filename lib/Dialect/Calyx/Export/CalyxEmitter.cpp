//===- CalyxEmitter.cpp - Calyx dialect to .futil emitter -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements an emitter for the native Calyx language, which uses
// .futil as an alias.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxEmitter.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Translation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace calyx;

namespace {

static constexpr std::string_view LSquare() { return "["; }
static constexpr std::string_view RSquare() { return "]"; }
static constexpr std::string_view LParen() { return "("; }
static constexpr std::string_view RParen() { return ")"; }
static constexpr std::string_view colon() { return ": "; }
static constexpr std::string_view space() { return " "; }
static constexpr std::string_view period() { return "."; }
static constexpr std::string_view questionMark() { return " ? "; }
static constexpr std::string_view exclamationMark() { return "!"; }
static constexpr std::string_view equals() { return " = "; }
static constexpr std::string_view comma() { return ", "; }
static constexpr std::string_view arrow() { return " -> "; }
static constexpr std::string_view delimiter() { return "\""; }
static constexpr std::string_view apostrophe() { return "'"; }
static constexpr std::string_view LBraceEndL() { return "{\n"; }
static constexpr std::string_view RBraceEndL() { return "}\n"; }
static constexpr std::string_view semicolonEndL() { return ";\n"; }

/// A tracker to determine which libraries should be imported for a given
/// program.
struct ImportTracker {
public:
  /// Returns the list of library names used for in this program.
  /// E.g. if `primitives/core.futil` is used, returns { "core" }.
  llvm::SmallSet<StringRef, 4> getLibraryNames(ProgramOp program) {
    program.walk([&](ComponentOp component) {
      for (auto &op : *component.getBody()) {
        if (!isa<CellInterface>(op) || isa<InstanceOp>(op))
          // It is not a primitive.
          continue;
        usedLibraries.insert(getLibraryFor(&op));
      }
    });
    return usedLibraries;
  }

private:
  /// Returns the library name for a given Operation Type.
  StringRef getLibraryFor(Operation *op) {
    StringRef library;
    TypeSwitch<Operation *>(op)
        .Case<MemoryOp, RegisterOp, NotLibOp, AndLibOp, OrLibOp, XorLibOp,
              AddLibOp, SubLibOp, GtLibOp, LtLibOp, EqLibOp, NeqLibOp, GeLibOp,
              LeLibOp, LshLibOp, RshLibOp, SliceLibOp, PadLibOp>(
            [&](auto op) { library = "core"; })
        .Case<SgtLibOp, SltLibOp, SeqLibOp, SneqLibOp, SgeLibOp, SleLibOp,
              SrshLibOp>([&](auto op) { library = "binary_operators"; })
        /*.Case<>([&](auto op) { library = "math"; })*/
        .Default([&](auto op) {
          llvm_unreachable("Type matching failed for this operation.");
        });
    return library;
  }

  /// Maintains a unique list of libraries used throughout the lifetime of the
  /// tracker.
  llvm::SmallSet<StringRef, 4> usedLibraries;
};

//===----------------------------------------------------------------------===//
// Emitter
//===----------------------------------------------------------------------===//

/// An emitter for Calyx dialect operations to .futil output.
struct Emitter {
  Emitter(llvm::raw_ostream &os) : os(os) {}
  LogicalResult finalize();

  // Indentation
  raw_ostream &indent() { return os.indent(currentIndent); }
  void addIndent() { currentIndent += 2; }
  void reduceIndent() {
    assert(currentIndent >= 2 && "Unintended indentation wrap");
    currentIndent -= 2;
  }

  // Program emission
  void emitProgram(ProgramOp op);

  /// Import emission.
  void emitImports(ProgramOp op) {
    auto emitImport = [&](StringRef library) {
      // Libraries share a common relative path:
      //   primitives/<library-name>.futil
      os << "import " << delimiter() << "primitives/" << library << period()
         << "futil" << delimiter() << semicolonEndL();
    };

    for (StringRef library : importTracker.getLibraryNames(op))
      emitImport(library);
  }

  // Component emission
  void emitComponent(ComponentOp op);
  void emitComponentPorts(ComponentOp op);

  // Instance emission
  void emitInstance(InstanceOp op);

  // Wires emission
  void emitWires(WiresOp op);

  // Group emission
  void emitGroup(GroupInterface group);

  // Control emission
  void emitControl(ControlOp control);

  // Assignment emission
  void emitAssignment(AssignOp op);

  // Enable emission
  void emitEnable(EnableOp enable);

  // Register emission
  void emitRegister(RegisterOp reg);

  // Memory emission
  void emitMemory(MemoryOp memory);

  // Emits a library primitive with template parameters based on all in- and
  // output ports.
  // e.g.:
  //   $f.in0, $f.in1, $f.in2, $f.out : calyx.std_foo "f" : i1, i2, i3, i4
  // emits:
  //   f = std_foo(1, 2, 3, 4);
  void emitLibraryPrimTypedByAllPorts(Operation *op);

  // Emits a library primitive with a single template parameter based on the
  // first input port.
  // e.g.:
  //   $f.in0, $f.in1, $f.out : calyx.std_foo "f" : i32, i32, i1
  // emits:
  //   f = std_foo(32);
  void emitLibraryPrimTypedByFirstInputPort(Operation *op);

private:
  /// Used to track which imports are required for this program.
  ImportTracker importTracker;

  /// Emit an error and remark that emission failed.
  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    encounteredError = true;
    return op->emitError(message);
  }

  /// Emit an error and remark that emission failed.
  InFlightDiagnostic emitOpError(Operation *op, const Twine &message) {
    encounteredError = true;
    return op->emitOpError(message);
  }

  /// Helper function for emitting a Calyx section. It emits the body in the
  /// following format:
  /// {
  ///   <body>
  /// }
  template <typename Func>
  void emitCalyxBody(Func emitBody) {
    os << space() << LBraceEndL();
    addIndent();
    emitBody();
    reduceIndent();
    indent() << RBraceEndL();
  }

  /// Emits a Calyx section.
  template <typename Func>
  void emitCalyxSection(StringRef sectionName, Func emitBody,
                        StringRef symbolName = "") {
    indent() << sectionName;
    if (!symbolName.empty())
      os << space() << symbolName;
    emitCalyxBody(emitBody);
  }

  /// Helper function for emitting combinational operations.
  template <typename CombinationalOp>
  void emitCombinationalValue(CombinationalOp op, StringRef logicalSymbol) {
    auto inputs = op.inputs();
    os << LParen();
    for (size_t i = 0, e = inputs.size(); i != e; ++i) {
      emitValue(inputs[i], /*isIndented=*/false);
      if (i + 1 == e)
        continue;
      os << space() << logicalSymbol << space();
    }
    os << RParen();
  }

  /// Emits the value of a guard or assignment.
  void emitValue(Value value, bool isIndented) {
    if (auto blockArg = value.dyn_cast<BlockArgument>()) {
      // Emit component block argument.
      StringAttr portName = getPortInfo(blockArg).name;
      (isIndented ? indent() : os) << portName.getValue();
      return;
    }

    auto definingOp = value.getDefiningOp();
    assert(definingOp && "Value does not have a defining operation.");

    TypeSwitch<Operation *>(definingOp)
        .Case<CellInterface>([&](auto cell) {
          // A cell port should be defined as <instance-name>.<port-name>
          (isIndented ? indent() : os)
              << cell.instanceName() << period() << cell.portName(value);
        })
        .Case<hw::ConstantOp>([&](auto op) {
          // A constant is defined as <bit-width>'<base><value>, where the base
          // is `b` (binary), `o` (octal), `h` hexadecimal, or `d` (decimal).
          APInt value = op.value();

          (isIndented ? indent() : os)
              << std::to_string(value.getBitWidth()) << apostrophe() << "d";
          // We currently default to the decimal representation.
          value.print(os, /*isSigned=*/false);
        })
        .Case<comb::AndOp>([&](auto op) { emitCombinationalValue(op, "&"); })
        .Case<comb::OrOp>([&](auto op) { emitCombinationalValue(op, "|"); })
        .Case<comb::XorOp>([&](auto op) {
          // The XorOp is a bit different, since the Combinational dialect uses
          // it to represent binary not.
          if (!op.isBinaryNot()) {
            emitOpError(op, "Only supporting Binary Not for XOR.");
            return;
          }
          // The LHS is the value to be negated, and the RHS is a constant with
          // all ones (guaranteed by isBinaryNot).
          os << exclamationMark();
          emitValue(op.inputs()[0], /*isIndented=*/false);
        })
        .Default(
            [&](auto op) { emitOpError(op, "not supported for emission"); });
  }

  /// Emits a port for a Group.
  template <typename OpTy>
  void emitGroupPort(GroupInterface group, OpTy op, StringRef portHole) {
    assert((isa<GroupGoOp>(op) || isa<GroupDoneOp>(op)) &&
           "Required to be a group port.");
    indent() << group.symName().getValue() << LSquare() << portHole << RSquare()
             << equals();
    if (op.guard()) {
      emitValue(op.guard(), /*isIndented=*/false);
      os << questionMark();
    }
    emitValue(op.src(), /*isIndented=*/false);
    os << semicolonEndL();
  }

  /// Recursively emits the Calyx control.
  template <typename OpTy>
  void emitCalyxControl(OpTy controlOp) {
    // Check to see if this is a stand-alone EnableOp.
    if (isa<EnableOp>(controlOp)) {
      emitEnable(cast<EnableOp>(controlOp));
      return;
    }
    for (auto &&bodyOp : *controlOp.getBody()) {
      TypeSwitch<Operation *>(&bodyOp)
          .template Case<SeqOp>([&](auto op) {
            emitCalyxSection("seq", [&]() { emitCalyxControl(op); });
          })
          .template Case<ParOp>([&](auto op) {
            emitCalyxSection("par", [&]() { emitCalyxControl(op); });
          })
          .template Case<IfOp, WhileOp>([&](auto op) {
            indent() << (isa<IfOp>(op) ? "if " : "while ");
            emitValue(op.cond(), /*isIndented=*/false);
            if (auto groupName = op.groupName(); groupName.hasValue())
              os << " with " << groupName.getValue();
            emitCalyxBody([&]() { emitCalyxControl(op); });
          })
          .template Case<EnableOp>([&](auto op) { emitEnable(op); })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside control.");
          });
    }
  }

  /// The stream we are emitting into.
  llvm::raw_ostream &os;

  /// Whether we have encountered any errors during emission.
  bool encounteredError = false;

  /// Current level of indentation. See `indent()` and
  /// `addIndent()`/`reduceIndent()`.
  unsigned currentIndent = 0;
};

} // end anonymous namespace

LogicalResult Emitter::finalize() { return failure(encounteredError); }

/// Emit an entire program.
void Emitter::emitProgram(ProgramOp op) {
  for (auto &bodyOp : *op.getBody()) {
    if (auto componentOp = dyn_cast<ComponentOp>(bodyOp))
      emitComponent(componentOp);
    else
      emitOpError(&bodyOp, "Unexpected op");
  }
}

/// Emit a component.
void Emitter::emitComponent(ComponentOp op) {
  indent() << "component " << op.getName();

  // Emit the ports.
  emitComponentPorts(op);
  os << space() << LBraceEndL();
  addIndent();
  WiresOp wires;
  ControlOp control;

  // Emit cells.
  emitCalyxSection("cells", [&]() {
    for (auto &&bodyOp : *op.getBody()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<WiresOp>([&](auto op) { wires = op; })
          .Case<ControlOp>([&](auto op) { control = op; })
          .Case<InstanceOp>([&](auto op) { emitInstance(op); })
          .Case<RegisterOp>([&](auto op) { emitRegister(op); })
          .Case<MemoryOp>([&](auto op) { emitMemory(op); })
          .Case<hw::ConstantOp>([&](auto op) { /*Do nothing*/ })
          .Case<SliceLibOp, PadLibOp>(
              [&](auto op) { emitLibraryPrimTypedByAllPorts(op); })
          .Case<LtLibOp, GtLibOp, EqLibOp, NeqLibOp, GeLibOp, LeLibOp, SltLibOp,
                SgtLibOp, SeqLibOp, SneqLibOp, SgeLibOp, SleLibOp, AddLibOp,
                SubLibOp, ShruLibOp, RshLibOp, SrshLibOp, LshLibOp, AndLibOp,
                NotLibOp, OrLibOp, XorLibOp>(
              [&](auto op) { emitLibraryPrimTypedByFirstInputPort(op); })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside component");
          });
    }
  });

  emitWires(wires);
  emitControl(control);
  reduceIndent();
  os << RBraceEndL();
}

/// Emit the ports of a component.
void Emitter::emitComponentPorts(ComponentOp op) {
  // To avoid the native compiler adding each of the required ports twice,
  // add the @<port-name> attribute here. This is a quick-fix solution.
  // Eventually we want to add attributes directly to component arguments.
  // See: https://github.com/llvm/circt/issues/1666
  llvm::SmallVector<StringRef> requiredPorts = {"go", "clk", "done", "reset",
                                                "done"};
  auto requiresNativeCompilerAttribute = [&](StringRef portName) {
    return llvm::find_if(requiredPorts, [&](StringRef requiredPort) {
             return requiredPort == portName;
           }) != requiredPorts.end();
  };

  auto emitPorts = [&](auto ports) {
    os << LParen();
    for (size_t i = 0, e = ports.size(); i < e; ++i) {
      const auto &port = ports[i];
      std::string name = port.name.getValue().str();
      if (requiresNativeCompilerAttribute(name)) {
        // @<port-name> <port-name>
        name.insert(0, "@");
        name += " ";
        name += port.name.getValue();
      }
      // We only care about the bit width in the emitted .futil file.
      auto bitWidth = port.type.getIntOrFloatBitWidth();
      os << name << colon() << bitWidth;

      if (i + 1 < e)
        os << comma();
    }
    os << RParen();
  };
  emitPorts(op.getInputPortInfo());
  os << arrow();
  emitPorts(op.getOutputPortInfo());
}

void Emitter::emitInstance(InstanceOp op) {
  indent() << op.instanceName() << equals() << op.componentName() << LParen()
           << RParen() << semicolonEndL();
}

void Emitter::emitRegister(RegisterOp reg) {
  size_t bitWidth = reg.inPort().getType().getIntOrFloatBitWidth();
  indent() << reg.instanceName() << equals() << "std_reg" << LParen()
           << std::to_string(bitWidth) << RParen() << semicolonEndL();
}

void Emitter::emitMemory(MemoryOp memory) {
  size_t dimension = memory.sizes().size();
  if (dimension < 1 || dimension > 4) {
    emitOpError(memory, "Only memories with dimensionality in range [1, 4] are "
                        "supported by the native Calyx compiler.");
    return;
  }
  indent() << memory.instanceName() << " = std_mem_d"
           << std::to_string(dimension) << LParen() << memory.width()
           << comma();
  for (Attribute size : memory.sizes()) {
    APInt memSize = size.cast<IntegerAttr>().getValue();
    memSize.print(os, /*isSigned=*/false);
    os << comma();
  }

  ArrayAttr addrSizes = memory.addrSizes();
  for (size_t i = 0, e = addrSizes.size(); i != e; ++i) {
    APInt addrSize = addrSizes[i].cast<IntegerAttr>().getValue();
    addrSize.print(os, /*isSigned=*/false);
    if (i + 1 == e)
      continue;
    os << comma();
  }
  os << RParen() << semicolonEndL();
}

/// Calling getName() on a calyx operation will return "calyx.${opname}". This
/// function returns whatever is left after the first '.' in the string,
/// removing the 'calyx' prefix.
static StringRef removeCalyxPrefix(StringRef s) { return s.split(".").second; }

void Emitter::emitLibraryPrimTypedByAllPorts(Operation *op) {
  auto cell = cast<CellInterface>(op);
  indent() << cell.instanceName() << equals()
           << removeCalyxPrefix(op->getName().getStringRef()) << LParen();
  llvm::interleaveComma(op->getResults(), os, [&](auto res) {
    os << std::to_string(res.getType().getIntOrFloatBitWidth());
  });
  os << RParen() << semicolonEndL();
}

void Emitter::emitLibraryPrimTypedByFirstInputPort(Operation *op) {
  auto cell = cast<CellInterface>(op);
  unsigned bitwidth = cell.inputPorts()[0].getType().getIntOrFloatBitWidth();
  indent() << cell.instanceName() << equals()
           << removeCalyxPrefix(op->getName().getStringRef()) << LParen()
           << bitwidth << RParen() << semicolonEndL();
}

void Emitter::emitAssignment(AssignOp op) {

  emitValue(op.dest(), /*isIndented=*/true);
  os << equals();
  if (op.guard()) {
    emitValue(op.guard(), /*isIndented=*/false);
    os << questionMark();
  }
  emitValue(op.src(), /*isIndented=*/false);
  os << semicolonEndL();
}

void Emitter::emitWires(WiresOp op) {
  emitCalyxSection("wires", [&]() {
    for (auto &&bodyOp : *op.getBody()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<GroupInterface>([&](auto op) { emitGroup(op); })
          .Case<AssignOp>([&](auto op) { emitAssignment(op); })
          .Case<hw::ConstantOp, comb::AndOp, comb::OrOp, comb::XorOp>(
              [&](auto op) { /* Do nothing. */ })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside wires section");
          });
    }
  });
}

void Emitter::emitGroup(GroupInterface group) {
  auto emitGroupBody = [&]() {
    for (auto &&bodyOp : *group.getBody()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<AssignOp>([&](auto op) { emitAssignment(op); })
          .Case<GroupDoneOp>([&](auto op) { emitGroupPort(group, op, "done"); })
          .Case<GroupGoOp>([&](auto op) { emitGroupPort(group, op, "go"); })
          .Case<hw::ConstantOp, comb::AndOp, comb::OrOp, comb::XorOp>(
              [&](auto op) { /* Do nothing. */ })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside group.");
          });
    }
  };
  auto prefix = Twine(isa<CombGroupOp>(group) ? "comb " : "") + "group";
  emitCalyxSection(prefix.str(), emitGroupBody, group.symName().getValue());
}

void Emitter::emitEnable(EnableOp enable) {
  indent() << enable.groupName() << semicolonEndL();
}

void Emitter::emitControl(ControlOp control) {
  emitCalyxSection("control", [&]() { emitCalyxControl(control); });
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Emit the specified Calyx circuit into the given output stream.
mlir::LogicalResult circt::calyx::exportCalyx(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  Emitter emitter(os);
  for (auto &op : *module.getBody()) {
    op.walk([&](ProgramOp program) {
      emitter.emitImports(program);
      emitter.emitProgram(program);
    });
  }
  return emitter.finalize();
}

void circt::calyx::registerToCalyxTranslation() {
  static mlir::TranslateFromMLIRRegistration toCalyx(
      "export-calyx",
      [](ModuleOp module, llvm::raw_ostream &os) {
        return exportCalyx(module, os);
      },
      [](mlir::DialectRegistry &registry) {
        registry
            .insert<calyx::CalyxDialect, comb::CombDialect, hw::HWDialect>();
      });
}
