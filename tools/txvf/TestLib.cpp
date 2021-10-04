#include "TestLib.h"
namespace circt {
namespace txvf {

static LogicalResult mlirPrint(Transaction &transaction) {
  for (auto arg : transaction.args)
    if (arg.typeHint)
      outs() << printAnyValueWithType(arg.typeHint, arg.value);

  return success();
}

TestLib::TestLib() { addFunction("mlirPrint", mlirPrint); }

void TestLib::addFunction(StringRef identifier, const TestLibFunc &func) {
  assert(functions.count(identifier) == 0 &&
         "Multiple definitions of test library symbol");
  functions[identifier] = func;
}

Optional<TestLibFunc> TestLib::get(StringRef identifier) {
  auto it = functions.find(identifier);
  if (it == functions.end())
    return {};
  return {it->second};
}

} // namespace txvf
} // namespace circt
