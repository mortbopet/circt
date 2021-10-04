#ifndef CIRCT_TXVF_TESTLIB_H
#define CIRCT_TXVF_TESTLIB_H

#include "mlir/IR/Dialect.h"

#include "SimUtils.h"

using namespace llvm;
using namespace mlir;

namespace circt {
namespace txvf {

using TestLibFunc = std::function<LogicalResult(Transaction &)>;

class TestLib {
public:
  TestLib();
  void addFunction(StringRef identifier, const TestLibFunc &);
  Optional<TestLibFunc> get(StringRef identifier);

private:
  llvm::DenseMap<StringRef, TestLibFunc> functions;
};

} // namespace txvf
} // namespace circt

#endif // CIRCT_TXVF_TESTLIB_H
