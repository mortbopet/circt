
#ifndef CIRCT_TXVF_UTILS_H
#define CIRCT_TXVF_UTILS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;

namespace circt {
namespace txvf {

class DialectSimBackplane;

std::string printAnyValueWithType(mlir::Type type, Any &value);

class SimMemory {
public:
  SimMemory(size_t size, Type elementType) { initStore(size, elementType); }
  SimMemory(mlir::MemRefType memrefType) {
    ArrayRef<int64_t> shape = memrefType.getShape();
    int64_t allocationSize = 1;
    for (int64_t dim : shape)
      allocationSize *= dim;

    initStore(allocationSize, memrefType.getElementType());
  }

  Optional<Any> read(unsigned addr) {
    if (addr < store.size())
      return store[addr];
    return {};
  }

  /// todo(mortbopet): write tracing
  LogicalResult write(unsigned addr, Any val) {
    if (addr < store.size()) {
      store[addr] = val;
      return success();
    }
    return failure();
  }
  size_t size() const { return store.size(); }

private:
  void initStore(size_t size, Type elementType) {
    Any defaultValue =
        llvm::TypeSwitch<Type, Any>(elementType)
            .Case<mlir::IntegerType>([&](auto) {
              unsigned width = elementType.getIntOrFloatBitWidth();
              return APInt(width, 0);
            })
            .Case<mlir::FloatType>([&](auto) { return APFloat(0.0); })
            .Default([&](auto) {
              assert(false && "Unhandled element type");
              return APInt();
            });
    store.resize(size, defaultValue);
  }

  SmallVector<Any> store;
};

using SimMemoryPtr = std::shared_ptr<SimMemory>;

class Stream {

  /// Construct a stream with two endpoints; the dialect simulator must be able
  /// to determine endpoints.
};

/// todo(mortbopet): description
struct SimContext {
  /// todo(mortbopet): Handshake ops' executableOpsInterface is hardcoded to
  /// accept a value mapping, so for now, just let it access the value map
  /// directly. Should obviously be fixed.
  friend class HandshakeInterpreter;

public:
  /// Instruction iterator.
  mlir::Block::iterator instrIter;

  /// Maintain the set of memories reference-able in this scope.
  SmallVector<SimMemoryPtr> memories;

  ArrayRef<Any> getReturnValues() { return returnValues; }

  /// User pointers for simulators to inject custom info into the context.
  llvm::DenseMap<StringRef, Any> userPtrs;

  /// Yields control from this interpreter, placing return values into
  /// returnValues.
  /// todo(mortbopet): some check to verify that # of rvals is equal to required
  /// # of rvals for the simulated callable.
  void yield(ArrayRef<Any> rvals) {
    assert(returnValues.size() == 0 && "return values already set");
    llvm::copy(rvals, std::back_inserter(returnValues));
    hasYielded = true;
  }

  bool yielded() { return hasYielded; }

  void setValue(Value ssaValue, Any value) {
    assert(value.hasValue() &&
           "setting null values not allowed, prone to errors");
    values[ssaValue] = value;
  }

  Any getValue(Value ssaValue) {
    auto it = values.find(ssaValue);
    assert(it != values.end() &&
           "No sim value registered for the requested SSA value");
    return it->second;
  }

  bool hasValue(Value ssaValue) {
    return values.find(ssaValue) != values.end();
  }

  void eraseValue(Value ssaValue) {
    assert(hasValue(ssaValue) && "No sim value registered for the SSA value");
    values.erase(ssaValue);
  }

private:
  bool hasYielded = false;

  /// Return values of the current callable scope.
  SmallVector<Any> returnValues;

  /// Argument values of the current callable scope.
  SmallVector<Any> argValues;

  /// Maintain a mapping between SSA values in the current callable scope and
  /// their value in the simulation.
  llvm::DenseMap<Value, Any> values;
};

using SimContextPtr = std::shared_ptr<SimContext>;

/// todo(mortbopet): document this. Why all the fields? why keep a refernce to
/// the source context in the transaction?
struct Transaction {
  /// Transaction values. We cannot have a 'Value(Any value)' constructor since
  /// this will interfere with the copy constructor (due to Any mapping to...
  /// anything!).
  struct Value {
    Value(Any value, Type typeHint) : value(value), typeHint(typeHint) {}
    Any value;
    Type typeHint;
  };

  enum class ValueType { MemoryLike, ValueLike };
  Transaction() {}
  Transaction(const Transaction &other){};
  Transaction(SimContextPtr &&ctx) : context(ctx) {}
  SimContextPtr context;
  StringRef callee; // symbol
  SmallVector<Transaction::Value> args;
  SmallVector<Transaction::Value> results;
  Type transactorType;
};

/// todo(mortbopet): description
class Transactor {
public:
  Transactor(DialectSimBackplane *backplane) : backplane(backplane) {}
  virtual ~Transactor() = default;

  virtual Transaction encode(SimContextPtr &simctx, ArrayRef<Any> io) = 0;

protected:
  DialectSimBackplane *backplane = nullptr;
};

class EntryTransactor : public Transactor {
public:
  using Transactor::Transactor;
  virtual SimContextPtr decode(Operation *target, Transaction &transaction) = 0;
  virtual LogicalResult print(llvm::raw_fd_ostream &out, Operation *target,
                              Transaction &transaction) = 0;
};

class CallTransactor : public Transactor {
public:
  using Transactor::Transactor;
  virtual void decode(std::vector<Any> &outs, Operation *target,
                      Transaction &transaction) = 0;
};

} // namespace txvf
} // namespace circt

#endif // CIRCT_TXVF_UTILS_H
