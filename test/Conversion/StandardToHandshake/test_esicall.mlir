// RUN: circt-opt -lower-std-to-handshake -split-input-file %s | FileCheck %s


// CHECK: hw.module.extern @add(%arg0: !esi.channel<i32>, %arg1: !esi.channel<i32>) -> (out0: !esi.channel<i32>)
// CHECK: handshake.func @main(%arg0: i32, %arg1: i32, %arg2: i1, %arg3: none, ...) -> (i32, none) {
// CHECK:   %0 = "handshake.merge"(%arg0) : (i32) -> i32
// CHECK:   %1 = "handshake.merge"(%arg1) : (i32) -> i32
// CHECK:   %2 = "handshake.merge"(%arg2) : (i1) -> i1
// CHECK:   "handshake.sink"(%2) : (i1) -> ()
// CHECK:   %3 = handshake.call @add(%0, %1) : (i32, i32) -> i32
// CHECK:   handshake.return %3, %arg3 : i32, none
// CHECK: }

func @add(%arg0 : i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}

func @main(%arg0 : i32, %arg1 : i32, %cond : i1) -> i32 {
  %0 = call @add(%arg0, %arg1) {ESI} : (i32, i32) -> i32
  return %0 : i32
}
