#include "SimUtils.h"

namespace circt {
namespace txvf {

std::string printAnyValueWithType(mlir::Type type, Any &value) {
  std::stringstream out;
  if (type.isa<mlir::IntegerType>() || type.isa<mlir::IndexType>()) {
    out << any_cast<APInt>(value).getSExtValue();
    return out.str();
  } else if (type.isa<mlir::FloatType>()) {
    out << any_cast<APFloat>(value).convertToDouble();
    return out.str();
  } else if (type.isa<mlir::NoneType>()) {
    return "none";
  } else {
    llvm_unreachable("Unknown result type!");
  }
}

} // namespace txvf
} // namespace circt
