// RUN: hlstool %s | FileCheck %s

// CHECK-LABEL: module mac(
// CHECK:         input         a_valid,
// CHECK:         input  [31:0] a_data,
// CHECK:         input         b_valid,
// CHECK:         input  [31:0] b_data,
// CHECK:         input         c_valid,
// CHECK:         input  [31:0] c_data,
// CHECK:         input         arg3_valid, arg4_ready, arg5_ready, clock, reset,
// CHECK:         output        a_ready, b_ready, c_ready, arg3_ready, arg4_valid,
// CHECK:         output [31:0] arg4_data,
// CHECK:         output        arg5_valid
func @mac(%a : i32, %b : i32, %c : i32) -> i32 attributes {argNames = ["a", "b", "c"]} {
  %m = arith.muli %b, %c : i32
  %res = arith.addi %a, %b : i32
  return %res : i32
}