//===- HandshakeToAsync.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower the Handshake  dialect to
// Async dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HANDSHAKETOASYNC_H_
#define CIRCT_CONVERSION_HANDSHAKETOASYNC_H_

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
std::unique_ptr<mlir::Pass> createHandshakeToAsyncPass();
} // namespace circt

#endif // MLIR_CONVERSION_HANDSHAKETOASYNC_H_
